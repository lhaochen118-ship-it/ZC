#!/usr/bin/env python3
"""src/grammar.py (docs/08 §5.4 acceptance): for every training shop, every checked program of that shop
(src/refs/answer_examples.jsonl) wrapped in the output format is accepted by the shop's grammar (xgrammar, compiled once per
shop); the grammar is built in < 2 s per shop; and it rejects what it must (an unknown function, a wrong keyword, an item that
is not in the data, an import, a missing ANSWER, free text outside the tags).
Usage: test_grammar.py <train_split> <scratch> [--shops biz_101,...]"""
import os, sys, json, time
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "training", "data"))
import grammar, api_spec


def main():
    split, S = sys.argv[1], sys.argv[2]
    only = sys.argv[sys.argv.index("--shops") + 1].split(",") if "--shops" in sys.argv else None
    import common
    cache, canon = os.environ.get("CACHE") or os.path.join(S, "cache"), os.path.join(S, "canon")
    spec = api_spec.build_api_spec()
    by = defaultdict(list)
    for l in open(os.path.join(ROOT, "src", "refs", "answer_examples.jsonl"), encoding="utf-8"):
        r = json.loads(l); by[r["business_id"]].append(r)
    tot = ok = 0; slow = []; fails = []; neg_missed = []
    for b in sorted(by):
        if only and b not in only: continue
        biz, C, notes, de = common.canon(split, cache, canon, b)
        ev = grammar.entity_values(C)
        t = time.time(); g = grammar.build_grammar(spec, ev); tb = time.time() - t
        if tb > 2: slow.append((b, round(tb, 2)))
        chk = grammar.Checker(g)
        for r in by[b]:
            tot += 1
            if chk(grammar.wrap(r["code"], grammar.auto_plan(r["code"]))): ok += 1
            else: fails.append((b, r["code"]))
        if b == sorted(by)[0] or (only and b == only[0]):
            item = ev["item"][0]
            bad = {"function": 'ANSWER = {"type": "number", "answer": revenu(month(2025, 1))}',
                   "keyword": 'ANSWER = {"type": "number", "answer": revenue(month(2025, 1), product="x")}',
                   "entity": 'ANSWER = {"type": "number", "answer": revenue(month(2025, 1), item="Không có hàng này XYZ")}',
                   "import": 'import os\nANSWER = {"type": "number", "answer": 1}',
                   "no answer": 'x = revenue(month(2025, 1))'}
            for k, code in bad.items():
                if chk(grammar.wrap(code, "x")): neg_missed.append(k)
            if chk("Here is the answer: 5"): neg_missed.append("free text")
            assert chk(grammar.wrap(f'ANSWER = {{"type": "number", "answer": revenue(month(2025, 1), item={item!r})}}', "x")), "a real item rejected"
            assert chk("<plan>x</plan>\n<python>\nimport math\nANSWER = {'type': 'number', 'answer': 1}\n</python>"), "python escape rejected"
    for f in fails[:10]: print("REJECTED", f)
    print(f"accepted {ok}/{tot}; slow builds {slow}; negatives accepted {neg_missed}")
    good = ok == tot and not slow and not neg_missed
    print("test_grammar " + ("PASS" if good else "FAIL")); sys.exit(0 if good else 1)


if __name__ == "__main__":
    main()
