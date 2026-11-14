#!/usr/bin/env python3
"""Evaluation sets of docs/08_MASTER_PLAN.md §3 run through the hybrid flow (Kaggle job via run_kaggle.py --script eval_flow.py,
or locally). The ids of tests/splits/<set>.txt (originals, '<id>~p<k>' paraphrases, '<id>~accentfree') become ONE split
(each shop mapped once): data/<shop> linked from the train or public split, the public rulebook (training shops: their own
training rulebook in rulebooks/<shop>.md), the shared calendar; then agent.py --flow hybrid on it. Scoring is offline (tests/score_sets.py: a variant is scored
against its original question's reference answer).

Usage: eval_flow.py <split: public_test|train dir> <out> --sets quick400[,pub_full,...] [--para DIR] [-- agent flags...]
  --para DIR: holds para_pub.jsonl / para_train.jsonl (src/paraphrase_llm.py; Kaggle: <acc>/zc-para; else $ZC_PARA_DIR)"""
import os, sys, json, glob, random, shutil, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def find_split(name, hint):
    """the split directory (holding questions.jsonl) named name, near hint or under the data roots."""
    roots = [os.path.dirname(os.path.dirname(os.path.abspath(hint)))] + [os.environ.get("ZC_DATA", ""), os.path.join(os.path.dirname(HERE), "data", "raw")]
    for r in roots:
        if not r: continue
        for d in glob.glob(os.path.join(r, "**", name), recursive=True):
            if os.path.isfile(os.path.join(d, "questions.jsonl")): return d
    return None


def split_ids(name):
    for d in (os.path.join(os.path.dirname(HERE), "tests", "splits"), os.path.join(HERE, "refs", "splits")):
        p = os.path.join(d, name + ".txt")
        if os.path.isfile(p): return [l.strip() for l in open(p) if l.strip()]
    sys.exit(f"no split file for set {name}")


def build(sets, out_split, split_hint, para_dir):
    pub, tr = find_split("public_test", split_hint), find_split("train", split_hint)
    Q, src = {}, {}
    for sp in (pub, tr):
        if not sp: continue
        for l in open(os.path.join(sp, "questions.jsonl"), encoding="utf-8"):
            if l.strip(): q = json.loads(l); Q[q["id"]] = q; src[q["business_id"]] = sp
    for s in sets:                                                     # a set may ship its own questions (<set>.questions.jsonl,
        for d in (os.path.join(os.path.dirname(HERE), "tests", "splits"), os.path.join(HERE, "refs", "splits")):   # e.g. teacher
            p = os.path.join(d, s + ".questions.jsonl")                 # questions of the training shops, track M selector data)
            if os.path.isfile(p):
                for l in open(p, encoding="utf-8"):
                    if l.strip(): q = json.loads(l); Q.setdefault(q["id"], q)
                break
    para = {}
    for f in ("para_pub.jsonl", "para_train.jsonl"):
        d = para_dir or os.environ.get("ZC_PARA_DIR", "")                # local runs (preflight smoke): ZC_PARA_DIR
        p = os.path.join(d, f) if d else ""
        if not (p and os.path.isfile(p)):
            p = next(iter(sorted(glob.glob("/kaggle/input/**/" + f, recursive=True))), "")
        if p:
            for r in map(json.loads, open(p, encoding="utf-8")):
                if r.get("ok"): para[f"{r['id']}~p{r['k']}"] = r["paraphrase"]
    from perturb import perturb_question
    ids = []
    for s in sets: ids += [i for i in split_ids(s) if i not in ids]
    rows, miss = [], []
    for i in ids:
        if "~" in i and i in Q:                                         # a set's own variant question (<set>.questions.jsonl,
            rows.append(dict(Q[i])); continue                          # e.g. tr_newterm '<id>~n0'): its text as given
        base, _, var = i.partition("~")
        q = Q.get(base)
        if q is None: miss.append(i); continue
        if not var: text = q["question"]
        elif var.startswith("p"):
            text = para.get(i)
            if text is None: miss.append(i); continue
        elif var == "accentfree": text = perturb_question(q["question"], "accentfree", random.Random(0))
        else: miss.append(i); continue
        rows.append(dict(q, id=i, question=text))
    os.makedirs(os.path.join(out_split, "data"), exist_ok=True)
    for b in sorted({r["business_id"] for r in rows}):
        dst = os.path.join(out_split, "data", b)
        if not os.path.exists(dst): os.symlink(os.path.abspath(os.path.join(src[b], "data", b)), dst)
    shutil.copy(os.path.join(pub or tr, "calendar.csv"), out_split)
    shutil.copy(os.path.join(pub or tr, "rulebook.md"), out_split)    # the public rulebook (not a superset of the training one:
    if pub and tr:                                                      # it drops 3 training rows, docs/15 §4.3) -> each training
        os.makedirs(os.path.join(out_split, "rulebooks"), exist_ok=True)  # shop keeps the training rulebook (hybrid.rulebook_for)
        for b in sorted({r["business_id"] for r in rows if src.get(r["business_id"]) == tr}):
            shutil.copy(os.path.join(tr, "rulebook.md"), os.path.join(out_split, "rulebooks", b + ".md"))
    extra_rows = {}                                                     # a set may add rulebook rows for some shops (tr_newterm:
    for s in sets:                                                      # new business terms defined in the shop's rulebook, as a
        for d in (os.path.join(os.path.dirname(HERE), "tests", "splits"), os.path.join(HERE, "refs", "splits")):   # new industry
            p = os.path.join(d, s + ".rulebook_rows.jsonl")             # of the private test would)
            if os.path.isfile(p):
                for l in open(p, encoding="utf-8"):
                    if l.strip(): r = json.loads(l); extra_rows.setdefault(r["business_id"], []).append(r["row"])
                break
    for b, rws in extra_rows.items():
        if b not in src: continue
        os.makedirs(os.path.join(out_split, "rulebooks"), exist_ok=True)
        dst = os.path.join(out_split, "rulebooks", b + ".md")
        base = open(dst, encoding="utf-8").read() if os.path.isfile(dst) else open(os.path.join(src[b], "rulebook.md"), encoding="utf-8").read()
        with open(dst, "w", encoding="utf-8") as f:
            f.write(base.rstrip("\n") + "\n\n## Thuật ngữ riêng của cửa hàng\n| Thuật ngữ | Định nghĩa |\n|---|---|\n" + "\n".join(dict.fromkeys(rws)) + "\n")
    with open(os.path.join(out_split, "questions.jsonl"), "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"eval split: {len(rows)} questions of {sets} ({len(miss)} ids without text: {miss[:5]})", flush=True)
    return len(rows)


def main():
    argv = sys.argv[1:]; extra = []
    if "--" in argv: i = argv.index("--"); argv, extra = argv[:i], argv[i + 1:]
    if "--fake-responder" in argv:                                      # preflight appends it at the end (outside '--')
        i = argv.index("--fake-responder"); extra += argv[i:i + 2]; del argv[i:i + 2]
    split, out = argv[0], argv[1]
    sets = argv[argv.index("--sets") + 1].split(",") if "--sets" in argv else ["quick400"]
    para = argv[argv.index("--para") + 1] if "--para" in argv else ""
    rest = [x for j, x in enumerate(argv[2:]) if x not in ("--sets", "--para") and (j == 0 or argv[2:][j - 1] not in ("--sets", "--para"))]
    sp = os.path.join(out, "eval_split")
    n = build(sets, sp, split, para)
    if not n: sys.exit(f"eval_flow: no question of {sets} has a text here (paraphrase sets need --para / $ZC_PARA_DIR / a zc-para dataset)")
    if os.environ.get("ZC_SMOKE") == "1" and "--subset" not in extra: extra += ["--subset", "16"]
    dl = float(os.environ.get("ZC_DEADLINE") or 0)                     # Kaggle's 12 h session (make_kernel.py): unanswered
    if dl and "--hard-deadline" not in extra + rest:                    # questions abstained 10 min before the kernel's deadline,
        extra += ["--hard-deadline", str(int(dl - 600))]                # so the run writes its answers (no fast mode: same results)
    def opt(name):
        return rest[rest.index(name) + 1] if name in rest and rest.index(name) + 1 < len(rest) else ""
    if opt("--config") and "--units" not in rest + extra:               # plan 4 W: the model writes this split's rule units first
        import predict_flow                                             # (from the eval split's rulebook / shop data, no answer)
        fake = extra[extra.index("--fake-responder") + 1] if "--fake-responder" in extra else ""
        wm = opt("--model")                                             # the writer = the solver's merged weights (as in the image:
        try:                                                            # one 9.65B model fits the 15B budget, two do not); "units_model":
            if json.load(open(opt("--config"), encoding="utf-8")).get("units_model") == "base": wm = os.environ.get("ZC_BASE_MODEL") or wm   # "base" for studies
        except Exception: pass
        uv = predict_flow.prepare_units(opt("--config"), sp, out, wm, lambda *m: print(*m, flush=True),
                                        os.environ.get("ZC_UNITS_FAKE", ""), opt("--cache"))
        if uv: extra += ["--units", uv]
        if "--links" not in rest + extra:                                # plan 5 W2: the model links each question to rulebook rows
            kv = predict_flow.prepare_link(opt("--config"), sp, out, opt("--model"), lambda *m: print(*m, flush=True),
                                           os.environ.get("ZC_LINK_FAKE", "") or fake, opt("--cache"))
            if kv: extra += ["--links", kv]
    cmd = [sys.executable, os.path.join(HERE, "agent.py"), sp, os.path.join(out, "run")] + rest + extra
    rc = subprocess.run(cmd).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()
