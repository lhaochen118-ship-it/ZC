#!/usr/bin/env python3
"""src/precheck.py: no error on any checked program (src/refs/answer_examples.jsonl, every training shop, entity values of
its automap tables, its hints), and the wrong programs of each rule are caught (unknown function, wrong keyword, too many
arguments, entity not in the data, wrong by=, import / open, syntax, no ANSWER).
Usage: test_precheck.py <train_split> <scratch> [--warn-max 0.02]"""
import os, sys, json, pickle, time
from collections import defaultdict, Counter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "training", "data"))
os.environ.setdefault("ZC_ADJ_ROWS", "move")
import precheck, api_spec, grammar


def main():
    split, scratch = sys.argv[1], sys.argv[2]
    warn_max = float(sys.argv[sys.argv.index("--warn-max") + 1]) if "--warn-max" in sys.argv else 0.02
    import common
    canon_dir = os.path.join(scratch, "canon"); cache = os.environ.get("CACHE") or os.path.join(scratch, "cache")
    spec = api_spec.build_api_spec()
    ex = [json.loads(l) for l in open(os.path.join(ROOT, "src", "refs", "answer_examples.jsonl"), encoding="utf-8")]
    by = defaultdict(list)
    for r in ex: by[r["business_id"]].append(r)
    t0 = time.time(); errs, warns = [], Counter(); n = 0; first = None
    for b in sorted(by):
        c = common.ctx(split, cache, canon_dir, b, "compact")
        ev = c.entity_values
        for r in by[b]:
            h = c.hints(r["question"], r["as_of_date"])
            iss = precheck.precheck(r["code"], spec, h, ev, None, r["as_of_date"]); n += 1
            e = precheck.errors(iss)
            if e: errs.append((r["id"], r["code"], e))
            for i in iss:
                if i["level"] == "warn": warns[i["rule"]] += 1
        if first is None: first = (c, by[b][0])
    for x in errs[:10]: print("FALSE ERROR", x)
    nw = sum(warns.values())
    print(f"checked programs: {n}, false errors {len(errs)}, warnings {nw} {dict(warns)} ({time.time() - t0:.0f}s)")
    c, r = first; ev = c.entity_values; item = ev["item"][0]
    bad = {"unknown function": 'ANSWER = {"type": "number", "answer": revenu(month(2025, 1))}',
           "wrong keyword": 'ANSWER = {"type": "number", "answer": revenue(month(2025, 1), product="x")}',
           "too many args": 'ANSWER = {"type": "number", "answer": cancel_rate(month(2025, 1), 3, 4)}',
           "entity": 'ANSWER = {"type": "number", "answer": revenue(month(2025, 1), item="Sản phẩm không tồn tại XYZ")}',
           "by": 'ANSWER = {"type": "list", "answer": top_items(month(2025, 1), 3, by="sales")}',
           "import": 'import os\nANSWER = {"type": "number", "answer": 1}',
           "open": 'ANSWER = {"type": "number", "answer": len(open("/etc/passwd").read())}',
           "syntax": 'ANSWER = {"type": "number", "answer": revenue(month(2025, 1)}',
           "no answer": 'x = revenue(month(2025, 1))'}
    missed = [k for k, code in bad.items() if not precheck.errors(precheck.precheck(code, spec, {}, ev, None, r["as_of_date"]))]
    ok_item = precheck.errors(precheck.precheck(f'ANSWER = {{"type": "number", "answer": revenue(month(2025, 1), item={item!r})}}', spec, {}, ev, None, r["as_of_date"]))
    print("missed bad programs:", missed, "| real item flagged:", ok_item)
    ok = not errs and not missed and not ok_item and nw <= warn_max * n
    print("test_precheck " + ("PASS" if ok else "FAIL")); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
