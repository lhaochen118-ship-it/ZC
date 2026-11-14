#!/usr/bin/env python3
"""Plan 5 W3 (docs/18): the judge. After the solver S sampled its programs for a question, the writer / judge model W (LoRA-W on
the solver's merged weights; zero-shot = the solver's weights) reads the question, the shop's context and the DISTINCT candidate
programs with what they returned, and says which one answers the question: <choice>k</choice> (k = candidate number, 0 = none,
U = the data does not record what the question needs). No hand-written rule reads the question; the policy below only combines
the judge's samples with the vote.

Shared by inference (this file's job, src/predict_flow.py phase C) and the training-data factory (training/data/w_judge.py): the
SAME prompt is built in both.

    candidates(log_row)                       -> distinct answer groups of a solver log row (src/hybrid.py finish()'s log)
    question_block(question, as_of, cands, mode) -> the per-question part of the prompt (after the shop prefix)
    parse(text)                               -> int k | 0 | "U" | None
    decide(row, cands, choices, **policy)     -> (final answer, confidence, how)

GPU job (run_kaggle.py <split> OUT/x --script judge.py --model M --logs F [--units F] --variants list,list+think,point,point+think):
  one variant per GPU at the same time (user 14/11); writes <out>/<variant>/verdicts.jsonl (+ submission.jsonl under the default
  policy). Offline scoring / policy grid: tests/replay_judge.py."""
import os, re, sys, json, glob, gzip, time, pickle, argparse, subprocess
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MAX_CAND = 6                    # candidates shown (by votes); the rest of the groups are rare one-sample answers
CODE_CHARS = 700                # a candidate's program as shown (toolkit programs are ~100 characters; pandas escapes are cut)
RES_CHARS = 220                 # a candidate's returned answer as shown

TASK = ("# TASK\nDo not write a new program. The numbered CANDIDATES above were written for the QUESTION; each shows what it "
        "returned. Read the question (the quantity asked, the period, the filters, the rulebook terms it uses, whether the shop's "
        "data records it), then write exactly:\n<plan>quantity | period | filters the question asks for</plan>\n<choice>k</choice>\n"
        "k = the number of the candidate whose program answers the question exactly; 0 when none of them does; U when none does "
        "because the shop's data does not record what the question needs.")


def _norm(s):
    import unicodedata
    s = unicodedata.normalize("NFC", str(s)).lower()
    s = "".join(c if (c.isalnum() or c.isspace()) else " " for c in s)
    return re.sub(r"\s+", " ", s).strip()


def canon(a):
    """hashable key of an answer: answers equal under the organisers' matching rules share it (= agent.canon; kept here so the
    data factory does not import the agent)."""
    try:
        t, v = a["type"], a.get("answer")
        def el(x): return ("n", round(float(x), 2)) if isinstance(x, (int, float)) and not isinstance(x, bool) else ("s", _norm(x))
        if t in ("number", "percent"): return str((t, round(float(v), 2)))
        if t == "string": return str((t, _norm(v)))
        if t == "list": return str((t, tuple(el(x) for x in v)))
        if t == "table": return str((t, tuple(_norm(c) for c in v["columns"]), tuple(sorted(tuple(el(x) for x in r) for r in v["rows"]))))
        return str((t, json.dumps(v, ensure_ascii=False, sort_keys=True)))
    except Exception:
        return "?" + json.dumps(a, ensure_ascii=False, sort_keys=True, default=str)[:300]


def samples_of(row):
    lg = row.get("samples")
    return (lg.get("samples") or []) if isinstance(lg, dict) else []


def candidates(row, max_cand=MAX_CAND):
    """the distinct answers of a solver log row, most voted first (ties: first sampled): [{key, answer, code, votes, n}], and the
    number of samples tried. Votes as in hybrid.finish (a flagged sample weighs 0.5); code = the group's most frequent program."""
    S = samples_of(row); groups = {}
    for s in S:
        if not s.get("ok") or not isinstance(s.get("answer"), dict): continue
        k = canon(s["answer"])
        g = groups.setdefault(k, {"key": k, "answer": s["answer"], "codes": Counter(), "votes": 0.0, "n": 0, "first": len(groups)})
        t = (s.get("turns") or [{}])[-1]
        code = (t.get("code") or "").strip()
        if t.get("escape") == "sql": code = "<sql>\n" + code
        g["codes"][code] += 1; g["votes"] += 0.5 if s.get("flags") else 1.0; g["n"] += 1
    out = sorted(groups.values(), key=lambda g: (-g["votes"], g["first"]))[:max_cand]
    for g in out: g["code"] = g.pop("codes").most_common(1)[0][0]; g.pop("first")
    return out, len(S)


def show_answer(a):
    if not isinstance(a, dict): return str(a)[:RES_CHARS]
    if a.get("type") == "unanswerable": return "Unanswerable (the program says the data cannot answer)"
    s = json.dumps(a, ensure_ascii=False, default=str)
    return s if len(s) <= RES_CHARS else s[:RES_CHARS] + " …"


def show_code(code):
    c = (code or "").strip()
    return c if len(c) <= CODE_CHARS else c[:CODE_CHARS] + "\n# … (cut)"


def question_block(question, as_of, cands, mode="list", n_samples=0):
    """per-question part: # QUESTION, # CANDIDATES, # TASK. mode 'list': every candidate with its share of the solver's samples;
    'point': the given candidates without shares (the caller passes one candidate: a yes (1) / no (0, U) check)."""
    lines = [f"# QUESTION (today = {as_of})\n{question.strip()}", "",
             "# CANDIDATES (programs written for this question and what each returned)"]
    for i, c in enumerate(cands, 1):
        share = f" (written by {c['n']} of {n_samples} samples)" if mode == "list" and n_samples else ""
        lines.append(f"[{i}]{share}\n<code>\n{show_code(c['code'])}\n</code>\nreturned: {show_answer(c['answer'])}")
    return "\n".join(lines) + "\n\n" + TASK


def label_text(plan, choice):
    return f"<plan>{' '.join(str(plan).split()).replace('<', ' ')}</plan>\n<choice>{choice}</choice>"


_CH = re.compile(r"<choice>\s*\[?\s*([0-9]{1,2}|[Uu])\s*\]?\s*</choice>")
_CH2 = re.compile(r"choice\W{0,4}([0-9]{1,2}|[Uu])\b", re.I)


def parse(text, n=MAX_CAND):
    """the judge's choice: 1..n, 0, 'U'; None when the output has none (or a number outside the candidates)."""
    t = (text or "").split("</think>")[-1]
    m = _CH.findall(t) or _CH2.findall(t)
    if not m: return None
    v = m[-1]
    if v in ("U", "u"): return "U"
    k = int(v)
    return k if 0 <= k <= n else None


def vote_decision(cands, n_samples, theta=0.5, denom="all"):
    """the plain vote of hybrid.finish on the candidates (denom 'valid': share of the samples that ran)."""
    if not cands: return {"type": "abstain"}, 0.0
    d = max(1.0, float(n_samples if denom == "all" else sum(c["n"] for c in cands)))
    conf = cands[0]["votes"] / d
    return (cands[0]["answer"] if conf >= theta else {"type": "abstain"}), conf


def decide(cands, n_samples, choices, mode="list", tau=0.6, theta=0.5, denom="all", u_answers=True, keep_unanimous=0.0):
    """final answer from the vote and the judge's samples -> (answer, confidence, how).
    list : choices = the judge's samples for the whole list. The choice with share >= tau wins: a candidate -> its answer;
           0 -> abstain; U -> unanswerable (u_answers) or abstain. No such choice -> the vote.
    point: choices = {candidate index (0-based): [samples]} of yes (1) / no (0, 'U') checks. Score of a candidate = share of yes;
           the best candidate with score >= tau wins (ties: more votes); every candidate judged below 1 - tau -> abstain (or
           unanswerable when U has share >= tau on the top candidate and u_answers); else the vote.
    keep_unanimous > 0: a question whose samples all ran and agree keeps its answer unless the judge's share against it is
           >= keep_unanimous (a conservative override threshold for the ~90 % easy questions)."""
    va, vc = vote_decision(cands, n_samples, theta, denom)
    if not cands: return {"type": "abstain"}, 0.0, "no_candidate"
    unanimous = len(cands) == 1 and cands[0]["n"] == n_samples
    if mode == "list":
        ch = [c for c in (choices or []) if c is not None]
        if not ch: return va, vc, "vote(no_judge)"
        top, n = Counter(map(str, ch)).most_common(1)[0]; share = n / len(ch)
        need = max(tau, keep_unanimous) if unanimous and top != "1" else tau
        if share < need: return va, vc, "vote(judge_split)"
        if top == "U": return ({"type": "unanswerable", "answer": None} if u_answers else {"type": "abstain"}), share, "judge_U"
        k = int(top)
        if k == 0 or k > len(cands): return {"type": "abstain"}, share, "judge_none"
        return cands[k - 1]["answer"], share, "judge_pick" if k != 1 else "judge_top"
    sc, us = {}, {}
    for i, ch in (choices or {}).items():
        ch = [c for c in ch if c is not None]
        if not ch: continue
        sc[int(i)] = sum(1 for c in ch if c == 1) / len(ch); us[int(i)] = sum(1 for c in ch if c == "U") / len(ch)
    if not sc: return va, vc, "vote(no_judge)"
    best = max(sc, key=lambda i: (sc[i], cands[i]["votes"], -i))
    if sc[best] >= tau: return cands[best]["answer"], sc[best], "judge_pick" if best else "judge_top"
    rej = max(tau, keep_unanimous) if unanimous else tau
    if all(s <= 1 - rej for s in sc.values()) and len(sc) == len(cands):
        if us.get(0, 0) >= rej and u_answers: return {"type": "unanswerable", "answer": None}, us[0], "judge_U"
        return {"type": "abstain"}, 1 - max(sc.values()), "judge_none"
    return va, vc, "vote(judge_split)"


# ------------------------------------------------------------------ job
def load_logs(path):
    """solver log rows by id (the last line of an id wins). path: a logs.jsonl(.gz), or a directory holding one."""
    if os.path.isdir(path):
        c = sorted(glob.glob(os.path.join(path, "**", "logs.jsonl*"), recursive=True))
        if not c: sys.exit(f"[judge] no logs.jsonl under {path}")
        path = c[0]
    op = gzip.open if path.endswith(".gz") else open
    rows = {}
    with op(path, "rt", encoding="utf-8") as f:
        for l in f:
            if l.strip(): r = json.loads(l); rows[r["id"]] = r
    return rows


def find_file(name, extra=()):
    """a file given by path, or by name under the code tree's training/data/out (shipped with --train-data) / /kaggle/input."""
    if not name: return ""
    if os.path.exists(name): return name
    here = os.path.dirname(os.path.abspath(__file__))
    roots = list(extra) + [os.path.join(here, "training", "data", "out"), os.path.join(os.path.dirname(here), "training", "data", "out"), "/kaggle/input"]
    names = [name] + ([name[:-3]] if name.endswith(".gz") else [name + ".gz"])   # Kaggle unpacks .gz files of a dataset
    for r in roots:
        for nm in names:
            c = sorted(glob.glob(os.path.join(r, "**", nm), recursive=True))
            if c: return c[0]
    sys.exit(f"[judge] {name} not found (looked under {roots})")


def shop_contexts(split, bids, cache="", units_path="", prompt_mode="full", workers=4):
    """{bid: {"system", "business"}}: the solver's own prompt prefix of each shop (src/shopctx.py: schema card, README, rulebook
    changes, industry rules, + the W1 units block), built from the split's data (ingest + automap; a canon pickle of the agent's
    cache is used when there is one)."""
    import shopctx
    rb0 = open(os.path.join(split, "rulebook.md"), encoding="utf-8").read()
    units_all = []
    if units_path:
        import units as unitsmod
        units_all = unitsmod.load(units_path)
    out = {}
    def one(b):
        import ingest, automap
        rbp = os.path.join(split, "rulebooks", b + ".md")
        rb = open(rbp, encoding="utf-8").read() if os.path.isfile(rbp) else rb0
        bp = os.path.join(cache or "", b + ".pkl")
        biz = pickle.load(open(bp, "rb")) if cache and os.path.isfile(bp) else ingest.load_business(os.path.join(split, "data", b))
        C, notes, _ = automap.build(biz)
        notes = "\n".join(notes) if isinstance(notes, (list, tuple)) else (notes or "")
        ctx = shopctx.ShopCtx(split, b, biz, C, notes, automap.parse_readme(biz.readme).get("data_end"), rb, mode=prompt_mode)
        business = ctx.business
        if units_all:
            us = unitsmod.for_shop(units_all, b)[:unitsmod.MAX_UNITS]
            if us: business = business + "\n\n" + unitsmod.block(us)
        return b, {"system": ctx.system, "business": business}
    from concurrent.futures import ProcessPoolExecutor
    bids = sorted(bids)
    if workers > 1 and len(bids) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(bids))) as ex:
            futs = {b: ex.submit(_ctx_worker, split, b, cache, units_path, prompt_mode) for b in bids}
            for b, f in futs.items():
                try: out[b] = f.result()
                except Exception as e: print(f"[judge] {b}: no context ({type(e).__name__}: {e})", flush=True)
    else:
        for b in bids:
            try: out[b] = one(b)[1]
            except Exception as e: print(f"[judge] {b}: no context ({type(e).__name__}: {e})", flush=True)
    return out


def _ctx_worker(split, b, cache, units_path, prompt_mode):
    return shop_contexts(split, [b], cache, units_path, prompt_mode, workers=1)[b]


def build_requests(rows, ctxs, mode, k):
    """(meta, conversation, temperature, seed) per judge sample. list: k samples per question; point: k per candidate."""
    reqs, cands_of = [], {}
    for qi, (i, r) in enumerate(rows.items()):
        cands, n = candidates(r); cands_of[i] = (cands, n)
        ctx = ctxs.get(r.get("business_id"))
        if not cands or ctx is None: continue
        units = [(None, cands)] if mode == "list" else [(ci, [c]) for ci, c in enumerate(cands)]
        for ci, cs in units:
            qb = question_block(r["question"], r["as_of_date"], cs, mode, n)
            conv = [{"role": "system", "content": ctx["system"]}, {"role": "user", "content": ctx["business"] + "\n\n" + qb}]
            for j in range(k):
                reqs.append(({"id": i, "cand": ci, "j": j}, conv, 0.0 if j == 0 else 0.7, 100003 * (qi % 9973) + 131 * ((ci or 0) + 1) + j))
    return reqs, cands_of


def run_variant(a, variant, out):
    os.makedirs(out, exist_ok=True); t0 = time.time()
    mode = "point" if variant.startswith("point") else "list"; think = "+think" in variant
    rows = load_logs(find_file(a.logs))
    if os.environ.get("ZC_SMOKE") == "1" or a.subset: rows = dict(list(rows.items())[:a.subset or 12])
    ctxs = json.load(open(a.ctx, encoding="utf-8"))
    k = min(a.k, 3) if think else a.k                                   # thinking samples are ~10x the tokens: fewer of them
    reqs, cands_of = build_requests(rows, ctxs, mode, k)
    reqs.sort(key=lambda x: (rows[x[0]["id"]].get("business_id", ""), x[0]["id"]))      # one shop's prefix at a time (prefix cache)
    mt = 1280 if think else a.max_tokens
    if a.fake_responder:
        g = {}; exec(compile(open(a.fake_responder, encoding="utf-8").read(), a.fake_responder, "exec"), g)
        texts = [g["responder"](c) for _, c, _, _ in reqs]
    else:
        sys.argv = [sys.argv[0]]; import agent
        lora = find_file(a.lora) if a.lora else ""
        if lora and os.path.isfile(lora): lora = os.path.dirname(lora)
        eng = agent.Engine(argparse.Namespace(model=a.model, tp=1, max_len=a.max_len, think=think, eager=False, lora=lora, quant=a.quant,
                                              gpu_util=0.92, max_num_seqs=128, max_batched_tokens=16384, spec_mtp=0, prefix_budget=0))
        texts, ntok = eng.gen([c for _, c, _, _ in reqs], [t for _, _, t, _ in reqs], [s for _, _, _, s in reqs], mt, think=think)
        print(f"[judge] {variant}: {len(reqs)} requests, {ntok} tokens out, {time.time() - t0:.0f}s", flush=True)
    by = {}
    for (m, _, _, _), t in zip(reqs, texts): by.setdefault(m["id"], []).append((m, t))
    n_parsed = n_req = 0
    with open(os.path.join(out, "verdicts.jsonl"), "w", encoding="utf-8") as fv, open(os.path.join(out, "submission.jsonl"), "w", encoding="utf-8") as fs:
        for i, r in rows.items():
            cands, n = cands_of[i]
            outs = by.get(i, [])
            if mode == "list": choices = [parse(t, len(cands)) for _, t in outs]
            else:
                choices = {}
                for m, t in outs: choices.setdefault(m["cand"], []).append(parse(t, 1))
            flat = choices if mode == "list" else [c for v in choices.values() for c in v]
            n_req += len(flat); n_parsed += sum(1 for c in flat if c is not None)
            final, conf, how = decide(cands, n, choices, mode, tau=a.tau, denom=a.denom, u_answers=bool(a.u_answers), keep_unanimous=a.keep_unanimous)
            fv.write(json.dumps({"id": i, "business_id": r.get("business_id"), "mode": mode, "variant": variant, "n_samples": n,
                                 "cands": [{"answer": c["answer"], "code": c["code"][:CODE_CHARS], "votes": c["votes"], "n": c["n"]} for c in cands],
                                 "choices": choices, "texts": [t[-400:] for _, t in outs][:12], "final": final, "conf": conf, "how": how},
                                ensure_ascii=False, default=str) + "\n")
            fs.write(json.dumps({"id": i, **final}, ensure_ascii=False, default=str) + "\n")
    summ = {"variant": variant, "questions": len(rows), "requests": n_req, "parsed": n_parsed, "s": round(time.time() - t0)}
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print("JUDGE", json.dumps(summ), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("out"); ap.add_argument("--model", default=""); ap.add_argument("--logs", required=True)
    ap.add_argument("--units", default="", help="the W1 units of the solver run (cards_unit.jsonl): their block joins the shop prefix")
    ap.add_argument("--lora", default="", help="LoRA-W adapter directory (or a file name inside it); '' = the model as it is")
    ap.add_argument("--variants", default="list", help="comma list of list | point, each optionally +think: one GPU each, in parallel")
    ap.add_argument("--variant", default="", help="(internal) run this one variant in this process")
    ap.add_argument("--k", type=int, default=5, help="judge samples per question (list) / per candidate (point): 1 greedy + k-1 at 0.7")
    ap.add_argument("--quant", default="fp8"); ap.add_argument("--max-len", type=int, default=8192); ap.add_argument("--cache", default="")
    ap.add_argument("--ctx", default=""); ap.add_argument("--subset", type=int, default=0); ap.add_argument("--fake-responder", default="")
    ap.add_argument("--prompt", default="full"); ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--tau", type=float, default=0.6); ap.add_argument("--denom", default="all")
    ap.add_argument("--u-answers", type=int, default=1); ap.add_argument("--keep-unanimous", type=float, default=0.0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.variant: return run_variant(a, a.variant, a.out)
    t0 = time.time()
    rows = load_logs(find_file(a.logs))
    if os.environ.get("ZC_SMOKE") == "1" or a.subset: rows = dict(list(rows.items())[:a.subset or 12])
    units = find_file(a.units) if a.units else ""
    ctxs = shop_contexts(a.split, {r.get("business_id") for r in rows.values()}, a.cache, units, a.prompt, workers=os.cpu_count() or 4)
    if not ctxs: sys.exit("[judge] no shop context could be built")
    a.ctx = os.path.join(a.out, "ctx.json"); json.dump(ctxs, open(a.ctx, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[judge] {len(rows)} questions, {len(ctxs)} shop contexts ({time.time() - t0:.0f}s)", flush=True)
    variants = [v for v in a.variants.split(",") if v]
    import predict_flow
    devs = predict_flow.visible_gpus() if not a.fake_responder else []
    argv = [sys.executable, os.path.abspath(__file__), a.split]
    common = ["--model", a.model, "--logs", a.logs, "--ctx", a.ctx, "--k", str(a.k), "--quant", a.quant, "--max-len", str(a.max_len)] + \
             ["--max-tokens", str(a.max_tokens), "--tau", str(a.tau), "--denom", a.denom, "--u-answers", str(a.u_answers),
              "--keep-unanimous", str(a.keep_unanimous)] + (["--lora", a.lora] if a.lora else []) + (["--subset", str(a.subset)] if a.subset else []) + \
             (["--fake-responder", a.fake_responder] if a.fake_responder else [])
    procs, waiting = [], list(enumerate(variants))
    slots = devs or ["cpu"]; free = list(slots); rcs = {}
    while waiting or procs:                                             # one variant per GPU; more variants than GPUs queue up
        while waiting and free:
            n, v = waiting.pop(0); g = free.pop(0)
            env = dict(os.environ, **({"CUDA_VISIBLE_DEVICES": g} if g != "cpu" else {}))
            vo = os.path.join(a.out, v.replace("+", "_"))
            os.makedirs(vo, exist_ok=True)
            p = subprocess.Popen(argv + [vo, "--variant", v] + common, env=env, stdout=open(os.path.join(vo, "run.log"), "w"), stderr=subprocess.STDOUT)
            procs.append((p, v, g))
        time.sleep(0.2 if a.fake_responder else 5)
        for p, v, g in list(procs):
            if p.poll() is not None:
                rcs[v] = p.returncode; procs.remove((p, v, g)); free.append(g)
                tail = open(os.path.join(a.out, v.replace("+", "_"), "run.log"), encoding="utf-8", errors="replace").read()[-600:]
                print(f"[judge] variant {v} rc {p.returncode} on {g}:\n{tail}", flush=True)
    json.dump({"rcs": rcs, "s": round(time.time() - t0)}, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    print(f"[judge] done {rcs} in {time.time() - t0:.0f}s", flush=True)
    if any(rcs.values()) and not any(rc == 0 for rc in rcs.values()): sys.exit(1)


if __name__ == "__main__":
    main()
