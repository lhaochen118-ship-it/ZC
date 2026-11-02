#!/usr/bin/env python3
"""Generic fixes of mapping / answer program execution (no GPU, no data files):
dedent of code blocks, splicing of repair fragments, closest-name hints on KeyError, fillna(None) as a no-op, the soft
content checks (canon.secondary_checks) and their ranking, stock from lots when there is no snapshot.
Usage: test_mapping_fixes.py"""
import os, sys, types
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import pandas as pd
import agent, canon, dm

bad = []
def check(name, cond, info=""):
    if not cond: bad.append(name); print("FAIL", name, info)

# 1. a code block indented as a whole runs
c = agent.extract_code("Here:\n```python\n    # mapping\n    a = 1\n    C = {'orders': a}\n```\n")
check("dedent", c is not None and c.startswith("# mapping\na = 1"), repr(c))

# 2. a repair that sends only the fixed statements is spliced into the previous program
prev = 'h = T["hd"]\norders = frame(h, order_id="Ma", revenue="TongTien")\norders["valid"] = True\nC = {"orders": orders}\nNOTES = "x"'
frag = 'orders = frame(h, order_id="Ma", revenue="Tien hang")\npayments = frame(h, order_id="Ma", amount="Da tra")'
sp = agent.splice(prev, frag)
check("splice replaces", sp is not None and "Tien hang" in sp and "TongTien" not in sp and sp.index("payments") < sp.index("C = "), sp)
check("splice keeps the rest", sp is not None and "orders['valid'] = True" in sp and "NOTES = 'x'" in sp, sp)
check("splice: complete program untouched", agent.splice(prev, 'h = T["hd"]\norders = frame(h)\nC = {"orders": orders}') is None)
check("splice: syntax error untouched", agent.splice(prev, "orders = frame(h,") is None)
sp2 = agent.splice(prev, 'ls = frame(T["lo"], order_id="Ma", lot_id="Lo")\nC["lot_sales"] = ls')
check("splice: C extension after C", sp2 is not None and sp2.index("C = ") < sp2.index("C['lot_sales'] = ls") and sp2.index("ls = frame") < sp2.index("C['lot_sales']"), sp2)
check("splice: answer fragment", (agent.splice("x = revenue(month(2025, 1))\nANSWER = {'type': 'number', 'answer': x}",
                                                "x = revenue(month(2025, 2))") or "").count("2025, 2") == 1)

# 3. closest real names on a KeyError
biz = types.SimpleNamespace(tables={"HoaDon": pd.DataFrame({"Tên KH": ["a"], "Mã bill": ["b"]})}, raw={}, readme="")
h = canon.key_hint(KeyError("Ten khach hang"), biz) + canon.key_hint(KeyError("Ten KH"), biz)
check("key hint", "'Tên KH' (column of HoaDon)" in h, h)
check("key hint: table", "(table)" in canon.key_hint(KeyError("Hoa don"), biz))
r = canon.run_mapping('t = T["HoaDon"]\nx = t["Ten KH"]', biz)
check("key hint in the error", not r["ok"] and "closest names in the data: 'Tên KH'" in r["error"], r.get("error"))

# 4. fillna(None) is a no-op, other fills still work
s = pd.Series([1.0, None]); d = pd.DataFrame({"a": [None, 2.0]})
check("fillna None series", s.fillna(None).isna().sum() == 1)
check("fillna None frame", d.fillna(None).isna().sum().sum() == 1)
check("fillna value", s.fillna(0).tolist() == [1.0, 0.0] and d.fillna(5)["a"].tolist() == [5.0, 2.0])

# 5. soft checks: revenue against a 'Tiền hàng' column, lots without lot_sales; none on a right mapping
n = 120
raw = pd.DataFrame({"Mã bill": [f"B{i:04d}" for i in range(n)], "Tiền hàng": [100.0] * n,
                    "TongTien": [100.0 + (20.0 if i % 5 == 0 else 0.0) for i in range(n)], "Ngày": pd.date_range("2025-01-01", periods=n)})
biz = types.SimpleNamespace(tables={"hd": raw}, raw={}, readme="Dữ liệu có đến hết 30/04/2025.")
def C_of(rev_col):
    o = pd.DataFrame({"order_id": raw["Mã bill"], "order_date": raw["Ngày"], "valid": True, "revenue": raw[rev_col]})
    return {"orders": o, "lines": pd.DataFrame({"order_id": raw["Mã bill"], "line_amount": raw[rev_col]})}
w_bad, w_ok = canon.secondary_checks(C_of("TongTien"), biz), canon.secondary_checks(C_of("Tiền hàng"), biz)
check("revenue warning", any(x.startswith("orders.revenue differs from the column 'Tiền hàng'") for x in w_bad), w_bad)
check("no warning on the right column", w_ok == [], w_ok)
C = C_of("Tiền hàng"); C["lots"] = pd.DataFrame({"lot_id": ["L1"], "item_code": ["X"], "expiry": ["2026-01-01"], "qty_in_base": [10]})
check("lots without lot_sales", any("lot_sales" in x for x in canon.secondary_checks(C, biz)))
# ranking: a wrong revenue column is ranked below any candidate without it; a missing secondary table costs little
rs = canon.rank_score
check("rank: revenue", rs({"coverage": 30, "warnings": w_bad}) < rs({"coverage": 5, "warnings": []}))
check("rank: lots", rs({"coverage": 26, "warnings": ["C['lots'] is mapped without C['lot_sales']"]}) > rs({"coverage": 19, "warnings": []}))

# 6. stock from lots when there is no stock snapshot / ledger (lot-based pharmacies)
orders = pd.DataFrame({"order_id": ["O1", "O2"], "order_date": pd.to_datetime(["2025-03-01", "2025-03-05"]), "status": ["hoàn thành"] * 2,
                       "valid": True, "revenue": [10.0, 20.0]})
lines = pd.DataFrame({"order_id": ["O1", "O2"], "order_date": orders["order_date"], "status": ["hoàn thành"] * 2, "valid": True,
                      "item_code": ["X", "X"], "item_name": ["Thuốc X"] * 2, "qty": [2, 3], "qty_base": [2, 3], "line_amount": [10.0, 20.0]})
lots = pd.DataFrame({"lot_id": ["L1", "L2"], "item_code": ["X", "X"], "expiry": ["2026-01-01", "2027-01-01"], "qty_in_base": [4, 10],
                     "receipt_date": pd.to_datetime(["2025-01-01", "2025-02-01"])})
ls = pd.DataFrame({"order_id": ["O1", "O2"], "lot_id": ["L1", "L1"], "qty_base": [2, 2], "returned_base": [0, 0]})
sh = dm.Shop({"orders": orders, "lines": lines, "items": lines[["item_code", "item_name"]].drop_duplicates(), "lots": lots, "lot_sales": ls},
             today="2025-03-10")
check("stock from lots", sh.stock_at("Thuốc X", "2025-03-10") == 10 and sh.stock_at("X", "2025-03-02") == 12,
      (sh.stock_at("Thuốc X", "2025-03-10"), sh.stock_at("X", "2025-03-02")))
try: sh.stock_at("Thuốc Y", "2025-03-10"); check("stock of an item without lots is unanswerable", False)
except dm.Unanswerable: pass

print(f"{len(bad)} failures")
print("PASS" if not bad else "FAIL")
