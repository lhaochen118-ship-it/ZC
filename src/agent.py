#!/usr/bin/env python3
"""DataMate agent: messy business files -> clean tables -> LLM-written pandas code -> executed answer -> vote -> decide.

Pipeline per split (everything automatic, nothing keyed on business / question ids):
  1. ingest.load_business: header detection, Vietnamese numbers / dates, multi-file merge (cached per business)
  2. prompt = system + rulebook (shared prefix) + README + data card (per business) + question + as-of date
  3. vLLM generates K programs per question (1 greedy + K-1 sampled); each sets ANSWER = {"type", "answer"}
  4. programs run in a sandboxed worker pool (timeout); failures go back to the LLM with the error (repair rounds)
  5. answers are canonicalised with the organisers' matching rules and voted; low agreement -> abstain
  6. deterministic formatter (rounding, ints, dates) -> submission.jsonl (+ per-question logs)

Usage: agent.py <split_dir> <out_dir> --model <hf id or path> [--k 4] [--repairs 2] [--subset N] [--tau 0.5]
                [--cache <dir>] [--tp 2] [--max-len 24576] [--think] [--workers N]"""
import os, re, sys, json, time, math, signal, pickle, argparse, traceback, unicodedata
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ingest
from score import norm, judge

SYSTEM = """You are DataMate, an expert data analyst for Vietnamese small businesses. You answer a question by writing ONE Python program that computes the answer from the business's tables with pandas.

Environment available to your program (already loaded, do NOT read files):
- `T`: dict table_name -> pandas DataFrame, cleaned: numeric columns are float, date columns are datetime64, text is NFC-normalised. Every table has `_source` (file[sheet]) and maybe `_sheet`.
- `RAW`: same tables with the original cell strings (use only if the cleaned column looks wrong).
- `TODAY`: pandas Timestamp of the as-of date ("hôm nay"). Records dated after TODAY have not happened yet unless the rulebook says otherwise.
- `CAL`: solar-lunar calendar DataFrame with columns date (datetime64), weekday, lunar_day, lunar_month, lunar_year, lunar_leap, holiday.
- helpers: `parse_num(x)`, `parse_date(x)`, `norm_text(s)` (lowercase, no punctuation), `strip_accents(s)`, `match_names(query, candidates, k=5)` -> best matching candidate strings (accent/typo tolerant),
  `month_range(y, m)` -> (first_day, last_day), `lunar_to_solar(lunar_year, lunar_month, lunar_day)` -> Timestamp.
- pandas as `pd`, numpy as `np`, `re`, `math`, `datetime`.

Rules:
1. Follow the RULEBOOK exactly (definitions of revenue, valid orders, weeks, "tháng này", ranking, debt, cost, rounding, answer formats). Where the business README explains its own columns, statuses or unit conversions, use it.
2. Inspect values you rely on (statuses, units, names) programmatically rather than guessing spellings; status labels differ between shops (e.g. "Xong", "Đã hoàn thành", "ok" = completed).
3. Finish by assigning `ANSWER = {"type": <one of number|percent|string|date|boolean|list|table|unanswerable>, "answer": <value>}`.
   - number/percent: a Python number (money = integer VND; other averages/percent rounded to 2 decimals per the rulebook). percent 12.4% -> 12.4.
   - string: the name exactly as written in the data. date: "YYYY-MM-DD" (or "YYYY-MM" for month questions). boolean: True/False.
   - list: Python list in the order the question asks. table: {"columns": [...], "rows": [[...], ...]} with the entity name first.
   - unanswerable: when the data does not contain what is needed (rulebook section on unanswerable questions), with "answer": None.
4. You may print() short diagnostics. Output only one ```python code block, no explanation."""


def build_messages(rulebook, biz, q):
    user = (f"# RULEBOOK\n{rulebook}\n\n# BUSINESS README\n{biz.readme.strip()}\n\n# DATA CARD (tables in T)\n{CARDS[q['business_id']]}\n\n"
            f"# QUESTION\nas_of_date (TODAY): {q['as_of_date']}\nquestion: {q['question']}\n\nWrite the program.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = CODE_RE.findall(text)
    if m: return max(m, key=len)
    if "ANSWER" in text: return text
    return None


# ------------------------------------------------------------------ sandboxed execution (worker pool)
_BIZ = {}; _CTX = {}


def _init_worker(cache_dir, split_dir):
    _CTX["cache"], _CTX["split"] = cache_dir, split_dir
    cal = pd.read_csv(os.path.join(split_dir, "calendar.csv"))
    cal["date"] = pd.to_datetime(cal["date"]); _CTX["cal"] = cal


def _get_biz(bid):
    if bid not in _BIZ:
        if len(_BIZ) > 6: _BIZ.pop(next(iter(_BIZ)))
        _BIZ[bid] = load_biz(_CTX["cache"], _CTX["split"], bid)
    return _BIZ[bid]


def month_range(y, m):
    s = pd.Timestamp(int(y), int(m), 1); return s, s + pd.offsets.MonthEnd(0)


def match_names(query, candidates, k=5):
    import difflib
    cands = list(dict.fromkeys(str(c) for c in candidates if c is not None and str(c) != "nan"))
    qn = ingest.strip_accents(norm_text(query))
    keyed = {ingest.strip_accents(norm_text(c)): c for c in cands}
    exact = [c for kk, c in keyed.items() if kk == qn]
    sub = [c for kk, c in keyed.items() if qn and (qn in kk or kk in qn) and c not in exact]
    fuzzy = [keyed[x] for x in difflib.get_close_matches(qn, list(keyed), n=k, cutoff=0.5)]
    return list(dict.fromkeys(exact + sub + fuzzy))[:k]


def norm_text(s): return ingest.norm_text(s)


class _Timeout(Exception): pass


def _alarm(*_): raise _Timeout()


def run_code(task):
    """task = (key, bid, as_of, code, timeout) -> (key, result dict)"""
    key, bid, as_of, code, timeout = task
    import io, contextlib, datetime
    t0 = time.time(); buf = io.StringIO()
    try:
        biz = _get_biz(bid)
        cal = _CTX["cal"]
        def lunar_to_solar(ly, lm, ld):
            r = cal[(cal.lunar_year == int(ly)) & (cal.lunar_month == int(lm)) & (cal.lunar_day == int(ld)) & (~cal.lunar_leap.astype(str).str.lower().eq("true"))]
            return r.date.iloc[0] if len(r) else pd.NaT
        g = {"T": {k: v.copy() for k, v in biz.tables.items()}, "RAW": biz.raw, "TODAY": pd.Timestamp(as_of), "CAL": cal.copy(),
             "pd": pd, "np": np, "re": re, "math": math, "datetime": datetime, "parse_num": ingest.parse_num, "parse_date": ingest.parse_date,
             "norm_text": norm_text, "strip_accents": ingest.strip_accents, "match_names": match_names, "month_range": month_range,
             "lunar_to_solar": lunar_to_solar, "__name__": "__datamate__"}
        signal.signal(signal.SIGALRM, _alarm); signal.alarm(timeout)
        try:
            with contextlib.redirect_stdout(buf): exec(compile(code, "<program>", "exec"), g)
        finally: signal.alarm(0)
        if "ANSWER" not in g: return key, {"ok": False, "error": "program did not assign ANSWER", "stdout": buf.getvalue()[-1500:], "t": time.time() - t0}
        ans = fmt_answer(g["ANSWER"])
        return key, {"ok": True, "answer": ans, "stdout": buf.getvalue()[-1500:], "t": time.time() - t0}
    except _Timeout:
        return key, {"ok": False, "error": f"timeout after {timeout}s", "stdout": buf.getvalue()[-800:], "t": time.time() - t0}
    except Exception as e:
        tb = traceback.format_exc().splitlines()
        tb = [l for l in tb if "<program>" in l or not l.startswith("  File")][-8:]
        return key, {"ok": False, "error": "\n".join(tb)[-1500:], "stdout": buf.getvalue()[-800:], "t": time.time() - t0}


# ------------------------------------------------------------------ formatting / canonical keys
TYPES = {"number", "percent", "string", "date", "boolean", "list", "table", "unanswerable"}


def _py(v):
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): v = float(v)
    if isinstance(v, (np.bool_,)): return bool(v)
    if isinstance(v, (pd.Timestamp,)): return v.strftime("%Y-%m-%d")
    if isinstance(v, float):
        if not math.isfinite(v): raise ValueError("non-finite number in answer")
        v = round(v, 2); return int(v) if v == int(v) else v
    return v


def fmt_answer(a):
    if not isinstance(a, dict) or a.get("type") not in TYPES: raise ValueError(f"ANSWER must be a dict with a valid type, got {a!r}"[:300])
    t, v = a["type"], a.get("answer")
    if t == "unanswerable": return {"type": t, "answer": None}
    if v is None: raise ValueError("answer is None (use type unanswerable if the data cannot answer)")
    if isinstance(v, (pd.Series, np.ndarray)): v = list(v)
    if t in ("number", "percent"):
        if isinstance(v, (list, tuple)) and len(v) == 1: v = v[0]
        v = _py(v) if not isinstance(v, str) else _py(float(ingest.parse_num(v)))
        if not isinstance(v, (int, float)) or isinstance(v, bool): raise ValueError("number answer is not numeric")
    elif t == "string":
        if isinstance(v, (list, tuple)) and len(v) == 1: v = v[0]
        v = str(v).strip()
    elif t == "date":
        v = pd.Timestamp(v).strftime("%Y-%m-%d") if not (isinstance(v, str) and re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", v)) else v
    elif t == "boolean":
        v = bool(v)
    elif t == "list":
        if not isinstance(v, (list, tuple)): v = [v]
        v = [_py(x) if not isinstance(x, str) else x for x in v]
    elif t == "table":
        if isinstance(v, pd.DataFrame): v = {"columns": [str(c) for c in v.columns], "rows": v.values.tolist()}
        v = {"columns": [str(c) for c in v["columns"]], "rows": [[_py(x) if not isinstance(x, str) else x for x in r] for r in v["rows"]]}
    json.dumps(v)
    return {"type": t, "answer": v}


def canon(a):
    """Hashable key: answers equal under the organisers' matching rules get the same key."""
    t, v = a["type"], a.get("answer")
    def el(x): return ("n", round(float(x), 2)) if isinstance(x, (int, float)) and not isinstance(x, bool) else ("s", norm(x))
    if t in ("number", "percent"): return (t, round(float(v), 2))
    if t == "string": return (t, norm(v))
    if t == "list": return (t, tuple(el(x) for x in v))
    if t == "table": return (t, tuple(norm(c) for c in v["columns"]), tuple(sorted(tuple(el(x) for x in r) for r in v["rows"])))
    return (t, json.dumps(v, ensure_ascii=False, sort_keys=True))


# ------------------------------------------------------------------ data
CARDS = {}


def load_biz(cache_dir, split_dir, bid):
    p = os.path.join(cache_dir, bid + ".pkl") if cache_dir else None
    if p and os.path.isfile(p): return pickle.load(open(p, "rb"))
    biz = ingest.load_business(os.path.join(split_dir, "data", bid))
    if p: os.makedirs(cache_dir, exist_ok=True); pickle.dump(biz, open(p, "wb"), protocol=4)
    return biz


def select_subset(qs, subset):
    byb = defaultdict(list)
    for q in sorted(qs, key=lambda q: q["id"]): byb[q["business_id"]].append(q)
    per = max(1, subset // len(byb)); return [q for b in sorted(byb) for q in byb[b][:per]]


def shard_of(qs, n):
    """business -> shard, greedy balance of question counts (deterministic)."""
    cnt = Counter(q["business_id"] for q in qs); load = [0] * n; out = {}
    for b, c in sorted(cnt.items(), key=lambda x: (-x[1], x[0])):
        j = load.index(min(load)); out[b] = j; load[j] += c
    return out


def run_dp(a):
    """One agent process per GPU (CUDA_VISIBLE_DEVICES=i), then merge submissions / logs and score."""
    import subprocess, torch
    n = min(a.dp, max(1, torch.cuda.device_count())); t0 = time.time()
    argv = [x for x in sys.argv[1:]]
    i = argv.index("--dp"); del argv[i:i + 2]
    argv[1] = a.out                                                    # positional: split, out
    cache = a.cache or os.path.join(a.out, "cache"); os.makedirs(a.out, exist_ok=True)
    nw = str(max(1, (a.workers or os.cpu_count() or 4) // n))
    procs = []
    for g in range(n):
        args = [sys.executable, os.path.abspath(__file__), a.split, os.path.join(a.out, f"shard{g}")] + argv[2:] + \
               ["--shard", f"{g}/{n}", "--cache", cache, "--tp", "1", "--workers", nw]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        procs.append(subprocess.Popen(args, env=env, stdout=open(os.path.join(a.out, f"shard{g}.log"), "w"), stderr=subprocess.STDOUT))
    rcs = [p.wait() for p in procs]
    for g in range(n): print(f"--- shard{g} rc={rcs[g]}\n" + "".join(open(os.path.join(a.out, f"shard{g}.log")).readlines()[-12:]), flush=True)
    for f in ("submission.jsonl", "logs.jsonl"):
        with open(os.path.join(a.out, f), "w", encoding="utf-8") as w:
            for g in range(n):
                p = os.path.join(a.out, f"shard{g}", f)
                if os.path.isfile(p): w.write(open(p, encoding="utf-8").read())
    print(f"DP DONE {n} shards in {time.time() - t0:.0f}s", flush=True)
    ans_path = os.path.join(a.split, "answers.jsonl")
    if os.path.isfile(ans_path):
        import score
        qs = [json.loads(l) for l in open(os.path.join(a.split, "questions.jsonl"), encoding="utf-8") if l.strip()]
        if a.subset: qs = select_subset(qs, a.subset)
        ids = {q["id"] for q in qs}
        refs = {k: v for k, v in score.load(ans_path).items() if k in ids}
        res, _ = score.score(refs, score.load(os.path.join(a.out, "submission.jsonl")))
        print("SCORE", json.dumps({k: res[k] for k in ("PA", "n", "correct", "wrong", "abstain") if k in res}), flush=True)
        json.dump(res, open(os.path.join(a.out, "score.json"), "w"), ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("out")
    ap.add_argument("--model", default="Qwen/Qwen3-14B-AWQ"); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--repairs", type=int, default=2); ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.5); ap.add_argument("--cache", default="")
    ap.add_argument("--tp", type=int, default=0); ap.add_argument("--max-len", type=int, default=24576)
    ap.add_argument("--think", action="store_true"); ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=2048); ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--dp", type=int, default=1, help="data parallel: one model copy per GPU, businesses split across them")
    ap.add_argument("--shard", default="", help="internal: i/N")
    a = ap.parse_args()
    if a.dp > 1 and not a.shard: return run_dp(a)
    t_start = time.time(); os.makedirs(a.out, exist_ok=True)
    cache = a.cache or os.path.join(a.out, "cache")
    qs = [json.loads(l) for l in open(os.path.join(a.split, "questions.jsonl"), encoding="utf-8") if l.strip()]
    if a.subset: qs = select_subset(qs, a.subset)                      # stratified: first N/B questions of each business
    if a.shard:                                                        # keep this shard's businesses (balanced by question count)
        i, n = map(int, a.shard.split("/")); qs = [q for q in qs if shard_of(qs, n)[q["business_id"]] == i]
    rulebook = open(os.path.join(a.split, "rulebook.md"), encoding="utf-8").read()
    bids = sorted({q["business_id"] for q in qs})

    # 1. ingest (parallel, cached)
    from concurrent.futures import ProcessPoolExecutor
    nw = a.workers or max(1, (os.cpu_count() or 4))
    os.makedirs(cache, exist_ok=True)
    todo = [b for b in bids if not os.path.isfile(os.path.join(cache, b + ".pkl"))]
    with ProcessPoolExecutor(nw) as ex: list(ex.map(ingest._cache_one, [(a.split, cache, b) for b in todo]))
    bizs = {b: load_biz(cache, a.split, b) for b in bids}
    for b, z in bizs.items(): CARDS[b] = z.card()
    print(f"ingest {len(bids)} businesses {time.time() - t_start:.0f}s", flush=True)

    # 2. LLM
    from vllm import LLM, SamplingParams
    import torch
    tp = a.tp or max(1, torch.cuda.device_count())
    llm = LLM(model=a.model, tensor_parallel_size=tp, dtype="half" if torch.cuda.get_device_capability()[0] < 8 else "auto",
              max_model_len=a.max_len, gpu_memory_utilization=0.92, enable_prefix_caching=True, seed=0)
    tok = llm.get_tokenizer()
    def render(msgs):
        kw = {"enable_thinking": a.think} if "qwen3" in a.model.lower() else {}
        try: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        except TypeError: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print(f"model loaded {time.time() - t_start:.0f}s", flush=True)

    import multiprocessing as mp
    pool = mp.get_context("fork").Pool(nw, initializer=_init_worker, initargs=(cache, a.split), maxtasksperchild=200)

    # state per (question, sample)
    convs = {}; results = defaultdict(list); logs = defaultdict(list)
    for q in qs:
        base = build_messages(rulebook, bizs[q["business_id"]], q)
        for s in range(a.k): convs[(q["id"], s)] = list(base)
    qmap = {q["id"]: q for q in qs}
    pending = sorted(convs, key=lambda k: (qmap[k[0]]["business_id"], k))
    for rnd in range(a.repairs + 1):
        if not pending: break
        t0 = time.time()
        prompts = [render(convs[k]) for k in pending]
        sps = [SamplingParams(temperature=0 if k[1] == 0 else a.temp, top_p=1.0 if k[1] == 0 else 0.95, max_tokens=a.max_tokens,
                              seed=1000 * rnd + k[1]) for k in pending]
        outs = llm.generate(prompts, sps, use_tqdm=False)
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        tasks, nxt = [], []
        for k, o in zip(pending, outs):
            text = o.outputs[0].text; code = extract_code(text)
            convs[k] = convs[k] + [{"role": "assistant", "content": text}]
            if code is None:
                logs[k[0]].append({"s": k[1], "round": rnd, "error": "no code block"}); nxt.append(k)
                convs[k].append({"role": "user", "content": "No ```python block found. Output only the program."}); continue
            tasks.append((k, qmap[k[0]]["business_id"], qmap[k[0]]["as_of_date"], code, a.timeout))
        t1 = time.time()
        codes = {t[0]: t[3] for t in tasks}
        for k, r in pool.imap_unordered(run_code, tasks, chunksize=4):
            code = codes[k]
            logs[k[0]].append({"s": k[1], "round": rnd, "code": code, **{x: r[x] for x in r if x != "t"}, "exec_s": round(r["t"], 2)})
            if r["ok"]: results[k[0]].append(r["answer"])
            elif rnd < a.repairs:
                convs[k].append({"role": "user", "content": f"The program failed:\n{r['error']}\nstdout:\n{r.get('stdout', '')[-600:]}\nFix it and output the full corrected program."})
                nxt.append(k)
        print(f"round {rnd}: {len(pending)} gens {ntok} tok in {t1 - t0:.0f}s ({ntok / max(t1 - t0, 1e-9):.0f} tok/s), exec {time.time() - t1:.0f}s, "
              f"{len(nxt)} to repair", flush=True)
        pending = sorted(nxt, key=lambda k: (qmap[k[0]]["business_id"], k))
    pool.close()

    # 3. vote + decide + write
    sub = open(os.path.join(a.out, "submission.jsonl"), "w", encoding="utf-8")
    logf = open(os.path.join(a.out, "logs.jsonl"), "w", encoding="utf-8")
    for q in qs:
        rs = results.get(q["id"], [])
        votes = Counter(canon(r) for r in rs); rep = {canon(r): r for r in rs}
        if votes:
            top, n = votes.most_common(1)[0]; conf = n / a.k
            final = rep[top] if conf >= a.tau else {"type": "abstain"}
        else: conf, final = 0.0, {"type": "abstain"}
        row = {"id": q["id"], **final}
        sub.write(json.dumps(row, ensure_ascii=False) + "\n")
        logf.write(json.dumps({"id": q["id"], "question": q["question"], "business_id": q["business_id"], "final": final, "conf": conf,
                               "votes": [[str(k), v] for k, v in votes.most_common()], "attempts": logs[q["id"]]}, ensure_ascii=False, default=str) + "\n")
    sub.close(); logf.close()
    print(f"DONE {len(qs)} questions in {time.time() - t_start:.0f}s", flush=True)

    ans_path = os.path.join(a.split, "answers.jsonl")
    if os.path.isfile(ans_path):                                       # training split: score now (and per tau from the logs)
        import score
        refs = {k: v for k, v in score.load(ans_path).items() if k in qmap}
        res, _ = score.score(refs, score.load(os.path.join(a.out, "submission.jsonl")))
        print("SCORE", json.dumps({k: res[k] for k in ("PA", "n", "correct", "wrong", "abstain") if k in res}), flush=True)
        json.dump(res, open(os.path.join(a.out, "score.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
