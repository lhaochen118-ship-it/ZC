#!/usr/bin/env python3
"""Check canonical tables + dm metrics against the train answers, without an LLM.

For every business with a reference compile program (src/refs/compile/<bid>.py) the program is executed, the
canonical tables are checked, and every question that qparse recognises is answered by the metric library and
scored with the organisers' rules. Usage: eval_lib.py <split_dir> <cache_dir> [bid ...] [--show N]"""
import os, re, sys, json, pickle, glob, time
from collections import Counter, defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import pandas as pd, numpy as np
import dm, qparse, score, ingest


def fmt(a):
    """numbers as the organisers expect (numpy -> python, 2-decimal rounding)."""
    if a is None: return {"type": "error", "answer": "no ANSWER"}
    import agent
    return agent.fmt_answer(a)

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
    dump = open(argv[argv.index("--dump") + 1], "w") if "--dump" in argv else None
    if dump is not None: i = argv.index("--dump"); del argv[i:i + 2]
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
        end = re.search(r"có đến hết (\d{1,2})/(\d{1,2})/(\d{4})", biz.readme or "")
        meta = {"data_end": pd.Timestamp(int(end.group(3)), int(end.group(2)), int(end.group(1)))} if end else {}
        for q in bq:
            shop = dm.Shop(C, q["as_of_date"], meta)
            pr = qparse.parse(q["question"], q["as_of_date"], shop)
            if not pr: res["unparsed"] += 1; continue
            tpl, code = pr
            try: ans = dm.run_program(code, shop, q["as_of_date"]); ans = fmt(ans)
            except Exception as e: ans = {"type": "error", "answer": f"{type(e).__name__}: {e}"[:120]}
            ok = score.judge(ans, refs[q["id"]]) == "correct"
            res["ok" if ok else "wrong"] += 1; bytpl[tpl]["ok" if ok else "wrong"] += 1
            if not ok: bad.append((tpl, code[10:120], q["as_of_date"], q["question"][:100], refs[q["id"]].get("answer"), ans.get("answer")))
            elif dump is not None: dump.write(json.dumps({"id": q["id"], "business_id": bid, "question": q["question"], "as_of_date": q["as_of_date"], "tpl": tpl, "code": code}, ensure_ascii=False) + "\n")
        tot.update(res)
        print(f"{bid}: {len(C['orders'])} orders {errs or ''} | ok {res['ok']} wrong {res['wrong']} unparsed {res['unparsed']} ({time.time() - t0:.0f}s)")
        for b in bad[:show]: print("   ", b)
    print("TOTAL", dict(tot), {k: dict(v) for k, v in bytpl.items()})


if __name__ == "__main__":
    main()
