#!/usr/bin/env python3
"""WP9 regression tests (docs/08 WP9: one test per fixed error group, from training shops, never public questions):
- unit words: 'bán được bao nhiêu ly / mấy gói' links the shop's unit of measure in the hints (kind base_unit), only after a
  counting word ('lý do' / 'ly' elsewhere is not linked);
- debt_aging accepts flat edges ([7, 15] upper bounds, [0, 8, 16] lower bounds) as well as (lo, hi) pairs.
Usage: test_wp9_fixes.py <train_split> <scratch>"""
import os, sys, glob, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "training", "data"))


def main():
    split, S = sys.argv[1], sys.argv[2]
    import common, dm
    cache, canon = os.environ.get("CACHE") or os.path.join(S, "cache"), os.path.join(S, "canon")
    c = common.ctx(split, cache, canon, "biz_111", "compact")
    h = c.hints("Hôm qua bán được bao nhiêu ly?", "2025-06-14")
    u = [e for e in h["entities"] if e["kind"] in ("base_unit", "unit")]
    assert u and u[0]["kind"] == "base_unit" and dm.key(u[0]["data_value"]) == "ly", h["entities"]
    h = c.hints("hom qua ban dc may ly vay", "2025-06-14")
    assert any(e["kind"] == "base_unit" for e in h["entities"]), h["entities"]
    h = c.hints("Lý do khách huỷ đơn tháng trước là gì?", "2025-06-14")
    assert not any(e["kind"] in ("base_unit", "unit") for e in h["entities"]), h["entities"]
    n = 0
    for p in sorted(glob.glob(os.path.join(canon, "*.pkl"))):
        can = pickle.load(open(p, "rb")); C = can["C"]
        day = str(C["orders"].order_date.max().date())
        api = dm.api(dm.Shop(C, day, {"data_end": can.get("data_end")}), day)
        try: ref = api["debt_aging"](day)
        except Exception: continue
        if not sum(ref): continue
        assert api["debt_aging"](day, bins=[7, 15]) == ref and api["debt_aging"](day, bins=[0, 8, 16]) == ref, (p, ref)
        assert api["debt_aging"](day, bins=[(0, 7), (8, 15), (16, None)]) == ref
        n += 1
        if n >= 2: break
    assert n, "no shop with debts in the canonical tables"
    print(f"unit hints ok; debt_aging flat edges = pairs on {n} shops")
    print("test_wp9_fixes PASS")


if __name__ == "__main__":
    main()
