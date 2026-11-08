#!/usr/bin/env python3
"""src/tools.py (docs/08 §5.5): deterministic, outputs within ~300 tokens (900 characters), and T1 find() resolves the cases it
is for: accent-free, regional word ('cá tràu' = 'cá lóc', 'xây trát' = 'xây tô'), name prefix, typo; T3 period(), T9 has_data(),
T7 check(), T8 help(), T2 values(), T6 explain() (filter trace).
Usage: test_tools.py <train_split> <scratch>"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "training", "data"))
import tools


def main():
    split, S = sys.argv[1], sys.argv[2]
    import common
    cache, canon = os.environ.get("CACHE") or os.path.join(S, "cache"), os.path.join(S, "canon")
    rb = open(os.path.join(split, "rulebook.md"), encoding="utf-8").read()
    def T(b, day="2026-03-01"):
        c = common.ctx(split, cache, canon, b, "compact"); sh = c.shop(day)
        return tools.bind_tools(sh, day, rb), sh
    t, _ = T("biz_125")
    cases = [("canh chua ca loc", "Canh chua cá lóc", "no_accent"), ("canh chua cá tràu", "Canh chua cá lóc", "regional"),
             ("Canh chau cá lóc", "Canh chua cá lóc", "typo")]
    for text, want, how in cases:
        r = t["find"]("item", text)
        assert r and r[0] == want and r.info[0][1] == how, (text, r)
        assert repr(r) == repr(t["find"]("item", text)), "not deterministic"
    t2, sh2 = T("biz_113")
    r = t2["find"]("item", "cát xây trát"); assert r and r[0] == "Cát xây tô", r
    r = t2["find"]("item", "Gạch lát nền Prime"); assert r and r[0].startswith("Gạch lát nền Prime") and r.info[0][1] == "abbrev", r
    p = t["period"]("tuần trước"); assert str(p.start.date()) == "2026-02-16" and str(p.end.date()) == "2026-02-22", p
    try: t["period"]("hồi đó"); raise AssertionError("period() read nonsense")
    except ValueError as e: assert "tháng 7/2025" in str(e)
    assert "NOT recorded" in t["has_data"]("giới tính")
    assert t["check"]({"type": "list", "answer": ["a", "a"]}) == ["duplicates in list"]
    assert "revenue(" in str(t["help"]("revenue"))
    outs = [repr(t["find"]("item", "ca")), str(t["values"]("lines", "item_name")), str(t["schema"]()), str(t["peek"]("orders", 5)),
            str(t["rule"]("doanh thu")), str(t["help"]("top_items"))]
    assert all(len(o) <= 1800 for o in outs), [len(o) for o in outs]
    assert all(len(o) <= 900 for o in outs[:2] + outs[3:]), [len(o) for o in outs]
    ex = str(t["explain"]('ANSWER = {"type": "number", "answer": revenue(month(2025, 6), item="Canh chua cá lóc")}'))
    assert "item='Canh chua cá lóc' ->" in ex and "ANSWER" in ex, ex
    print("find / period / has_data / check / help / explain OK; output lengths", [len(o) for o in outs])
    print("test_tools PASS")


if __name__ == "__main__":
    main()
