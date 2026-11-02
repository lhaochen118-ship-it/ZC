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
C2 = C_of("Tiền hàng"); C2["purchases"] = pd.DataFrame({"receipt_id": ["P1"], "receipt_date": ["2025-02-01"], "amount": [10.0]})
C2["stock"] = pd.DataFrame({"item_code": ["X", "Y"] * 40, "qty_base": range(80)}); C2["conversions"] = pd.DataFrame(columns=["item", "unit", "factor"])
w2 = canon.secondary_checks(C2, biz)
check("purchases headers", any("C['purchases'] has no item_code" in x for x in w2), w2)
check("stock ledger as snapshot", any("rows per item" in x for x in w2), w2)
check("empty conversions", any("C['conversions'] is empty" in x for x in w2), w2)
# alias columns of the toolkit's tables are renamed; missing toolkit columns / a non-amount paid_at_purchase are flagged
C3 = {"deliveries": pd.DataFrame({"order_id": ["O1"], "delivery_date": ["2025-01-02"], "delivery_batch": [1]}),
      "payments": pd.DataFrame({"order_id": ["O1"], "pay_time": pd.to_datetime(["2025-01-03 10:00"]), "amount": [5.0]})}
canon.normalize_aliases(C3)
check("alias trip_no", "trip_no" in C3["deliveries"].columns and "delivery_batch" not in C3["deliveries"].columns)
check("pay_date from pay_time", "pay_date" in C3["payments"].columns)
C4 = C_of("Tiền hàng"); C4["orders"]["paid_at_purchase"] = pd.Timestamp("2025-01-01"); C4["deliveries"] = pd.DataFrame({"order_id": ["O1"], "delivery_date": ["2025-01-02"]})
C4["payments"] = pd.DataFrame({"order_id": ["B0001"], "pay_date": ["2025-01-05"], "amount": [1.0]})
w4 = canon.secondary_checks(C4, biz)
check("missing trip_no", any("C['deliveries'] lacks ['trip_no']" in x for x in w4), w4)
check("paid_at_purchase is a time", any("paid_at_purchase must be the AMOUNT" in x for x in w4), w4)
check("payments without total_payable", any("orders has no total_payable" in x for x in w4), w4)
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

# 7. a person named differently from the data: full name ~ nickname, name ~ name + phone; never two people
pdf = pd.DataFrame({"staff": ["Đức", "Phúc", None], "customer_name": ["Từ Hải Thanh 0359470831", "Lê An", "Trần An"]})
mp = lambda col, v: pdf[sh._match_person(pdf, (col,), v)][col].tolist()
check("person: nickname", mp("staff", "Lương Chí Đức") == ["Đức"] and mp("staff", "đức") == ["Đức"], mp("staff", "Lương Chí Đức"))
check("person: phone", mp("customer_name", "Từ Hải Thanh") == ["Từ Hải Thanh 0359470831"])
check("person: ambiguous / unknown", mp("customer_name", "An") == [] and mp("customer_name", "Nguyễn Văn Bình") == [])

# 8. rule parser: general patterns found on the train oracle (template chosen for a question; None = no rule answer)
import qparse
tpl = lambda q: (qparse.parse(q, "2026-07-31") or (None,))[0]
for q, want in [("coi gium thang 8/2025 dthu binh quan moi order la bao nhieu", "aov"),
                ("tháng này doanh thu các đơn bán tại quầy là bao nhiêu, doanh thu các đơn bán qua zalo là bao nhiêu", None),
                ("thang 2/2026 co bao nhieu mat hang xuat hien trong it nhat 107 don hang", "items_in_n_orders"),
                ("cho hoi 3 mat hang hot nhat tuan roi la nhung mat hang nao", "top_items"),
                ("tháng 6/2026 có bao nhiêu ngày đạt doanh thu từ 12.000.000 đồng trở lên", "n_days_rev_ge"),
                ("tháng 6/2026 có bao nhiêu bill có doanh thu từ 1.000.000 đồng trở lên", "n_orders_rev_ge"),
                ("thang 3/2026 co bao nhieu don bi huy vi giao tre", None),
                ("tỉ lệ đơn huỷ tháng 3/2026 của các đơn trả bằng ví điện tử là bao nhiêu %", "cancel_rate"),
                ("doanh thu tháng 7/2025 là bao nhiêu", "revenue"), ("tháng 7/2025 có bao nhiêu đơn", "n_orders"),
                ("thang 6/2025 mat hang Espresso (double) ban duoc tong cong bao nhieu ly", None)]:
    check(f"qparse: {q[:40]}", tpl(q) == want, tpl(q))
# 9. fee lines are not items
fl = pd.DataFrame({"item_name": ["Phí phục vụ", "Phin cà phê", "Bia"], "item_group": ["phí dịch vụ", "dụng cụ", "bia"]})
check("fee lines", dm.fee_lines(fl).tolist() == [True, False, False])

print(f"{len(bad)} failures")
print("PASS" if not bad else "FAIL")
