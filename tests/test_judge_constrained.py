#!/usr/bin/env python3
"""v2 A: the constrained judge format (judge.CHOICE_RE, variants 'list+c' / 'point+c') accepts the trained label format exactly
(label_text; every WJR1 training label when the data is present) and parse() reads it."""
import os, re, sys, gzip, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, os.path.join(ROOT, "src"))
import judge


def main():
    for plan, ch in (("revenue | month | item", 2), ("n_orders | last_week | shift", 0), ("x", "U")):
        t = judge.label_text(plan, ch)
        assert re.fullmatch(judge.CHOICE_RE, t), t
        assert judge.parse(t, 6) == ch, (t, judge.parse(t, 6))
    assert judge.parse("<plan>a</plan>\n<choice>7</choice>", 6) is None, "out-of-range choice"
    assert not re.fullmatch(judge.CHOICE_RE, "Let me think about candidate 1 ...")
    f = os.path.join(ROOT, "Process", "train_data", "p5", "wj_list_r.jsonl.gz")
    if os.path.isfile(f):
        bad = sum(1 for l in gzip.open(f, "rt", encoding="utf-8") if not re.fullmatch(judge.CHOICE_RE, json.loads(l)["messages"][-1]["content"].strip()))
        assert bad == 0, f"{bad} trained labels outside the constrained format"
    print("test_judge_constrained PASS")


if __name__ == "__main__":
    main()
