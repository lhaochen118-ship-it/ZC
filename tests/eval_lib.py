#!/usr/bin/env python3
"""Check canonical tables + dm metrics against the train answers, without an LLM.

For every business with a reference compile program (src/refs/compile/<bid>.py) the program is executed, the
canonical tables are checked, and every question that qparse recognises is answered by the metric library and
scored with the organisers' rules. Usage: eval_lib.py <split_dir> <cache_dir> [bid ...] [--show N]"""
import os, sys, json, pickle, glob, time
from collections import Counter, defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import pandas as pd, numpy as np
import dm, qparse, score, ingest

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "refs", "compile")


def namespace(biz):
    g = {"T": {k: v.copy() for k, v in biz.tables.items()}, "RAW": biz.raw, "pd": pd, "np": np, "re": __import__("re"),
         "parse_num": ingest.parse_num, "parse_date": ingest.parse_date}
    g.update({k: getattr(dm, k) for k in ("finalize", "map_status", "canon_status", "key", "is_walkin", "money", "round_half_up", "readme_conversions", "conversions_table", "frame")}); g["tbl"] = lambda *c, **kw: dm.find_table(g["T"], *c, **kw); g["README"] = biz.readme
    return g


def build(biz, bid):
    g = namespace(biz); exec(compile(open(os.path.join(REF, bid + ".py")).read(), bid, "exec"), g)
    return g["C"], g.get("NOTES", "")


def main():
    argv = sys.argv[1:]
    if "--show" in argv: i = argv.index("--show"); show = int(argv[i + 1]); del argv[i:i + 2]
    else: show = 5
    a = [x for x in argv if not x.startswith("--")]
    split, cache = a[0], a[1]
    dm.set_calendar(pd.read_csv(os.path.join(split, "calendar.csv")))
    qs = [json.loads(l) for l in open(os.path.join(split, "questions.jsonl"))]
    refs = score.load(os.path.join(split, "answers.jsonl"))
    bids = a[2:] or sorted(os.path.basename(f)[:-3] for f in glob.glob(os.path.join(REF, "*.py")))
    tot = Counter(); bytpl = defaultdict(Counter)
    for bid in bids:
        biz = pickle.load(open(os.path.join(cache, bid + ".pkl"), "rb"))
        t0 = time.time(); C, notes = build(biz, bid); errs = dm.check(C)
        bq = [q for q in qs if q["business_id"] == bid]; res = Counter(); bad = []
        for q in bq:
            shop = dm.Shop(C, q["as_of_date"])
            pr = qparse.parse(q["question"], q["as_of_date"], shop)
            if not pr: res["unparsed"] += 1; continue
            tpl, kw = pr
            try: ans = qparse.run_template(shop, tpl, kw, q["as_of_date"])
            except dm.Unanswerable: ans = {"type": "unanswerable", "answer": None}
            except Exception as e: ans = {"type": "error", "answer": f"{type(e).__name__}: {e}"}
            ok = score.judge(ans, refs[q["id"]]) == "correct"
            res["ok" if ok else "wrong"] += 1; bytpl[tpl]["ok" if ok else "wrong"] += 1
            if not ok: bad.append((tpl, {k: str(v) for k, v in kw.items()}, q["as_of_date"], q["question"][:90], refs[q["id"]].get("answer"), ans.get("answer")))
        tot.update(res)
        print(f"{bid}: {len(C['orders'])} orders {errs or ''} | ok {res['ok']} wrong {res['wrong']} unparsed {res['unparsed']} ({time.time() - t0:.0f}s)")
        for b in bad[:show]: print("   ", b)
    print("TOTAL", dict(tot), {k: dict(v) for k, v in bytpl.items()})


if __name__ == "__main__":
    main()
