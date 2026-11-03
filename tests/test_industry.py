#!/usr/bin/env python3
"""Industry toolkit (src/industry.py) on small synthetic shops: phone shop (devices, model lines, trade-ins, payable,
instalments in the debts, warranty), repair shop (tickets, jobs, vehicles, mechanics, commission), mother & baby (money off,
members, diaper packs). Every expected value is computed by hand from the public rulebook's definitions."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import pandas as pd
import dm, industry
from industry import model_line, brand_of, split_vehicle, vehicle_lines, job_of, job_query, split_jobs, device_kind

fails = 0


def check(name, got, want):
    global fails
    ok = got == want
    if not ok: fails += 1
    print(("ok   " if ok else "FAIL ") + f"{name}: {got!r}" + ("" if ok else f" (want {want!r})"))


# ---------------------------------------------------------------- helpers
check("model_line GB", model_line("iPhone 16 Pro Max 256GB Đen"), "iPhone 16 Pro Max")
check("model_line no GB", model_line("Nokia 105 4G"), "Nokia 105 4G")
check("brand iPhone", brand_of("iPhone 13 128GB"), "Apple")
check("brand first word", brand_of("Samsung Galaxy A56 5G 256GB"), "Samsung")
check("split_vehicle", split_vehicle("wave alpha 50K2-994.07"), ("wave alpha", "50K2-994.07"))
check("split_vehicle no plate", split_vehicle("Vision"), ("Vision", None))
vl = vehicle_lines(["Air Blade", "airblade", "AB", "Wave Alpha", "alpha", "wave α", "Vision", "Vision 110", "Exciter", "Ex", "Lead"])
check("vehicle lines AB", len({vl["Air Blade"], vl["airblade"], vl["AB"]}), 1)
check("vehicle lines alpha", len({vl["Wave Alpha"], vl["alpha"], vl["wave α"]}), 1)
check("vehicle lines distinct", len({vl["Air Blade"], vl["Wave Alpha"], vl["Vision"], vl["Exciter"], vl["Lead"]}), 5)
check("vehicle lines Ex", vl["Ex"] == vl["Exciter"], True)
check("ambiguous short name joins nothing", len(set(vehicle_lines(["Wave", "Wave Alpha", "Wave RSX"]).values())), 3)
check("job plain nhớt = engine oil", job_of("nhớt castrol"), "thay nhớt")
check("job nhớt hộp số = gear oil", job_of("nhớt hộp số"), "thay nhớt láp")
check("job plain ruột = new tube", job_of("ruột"), "thay ruột")
check("job vá ruột", job_of("vá ruột"), "vá ruột")
check("job bulb is not sửa điện", job_of("thay bóng đèn"), "thay bóng đèn")
check("job battery is not sửa điện", job_of("bình gs"), "thay ắc quy")
check("job làm điện", job_of("làm điện"), "sửa điện")
check("query dầu láp", job_query("thay dầu láp"), "thay nhớt láp")
check("split_jobs", split_jobs("bdtq + vs nồi, thay nhớt và rửa xe"), ["bdtq", "vs nồi", "thay nhớt", "rửa xe"])
check("device kinds", [device_kind(x) for x in ("Máy mới", "máy cũ", "Phụ kiện", "PK")], ["máy mới", "máy cũ", "phụ kiện", "phụ kiện"])

# ---------------------------------------------------------------- phone shop
lines = pd.DataFrame([
    # order, date, item, group, qty, amount, imei
    ("1", "2025-07-01", "iPhone 16 128GB Đen", "máy mới", 1, 20000000, "111"),
    ("1", "2025-07-01", "Ốp lưng silicon iPhone", "phụ kiện", 1, 100000, None),
    ("2", "2025-07-02", "iPhone 13 128GB Trắng", "máy cũ", 1, 8000000, "222"),
    ("3", "2025-07-03", "Samsung Galaxy A56 5G 256GB", "máy mới", 2, 16000000, "333"),
    ("4", "2025-07-04", "iPhone 16 256GB Xanh", "máy mới", 1, 23000000, "444"),     # returned
    ("5", "2025-08-05", "Củ sạc nhanh 20W Type-C", "phụ kiện", 3, 600000, None),
], columns=["order_id", "order_date", "item_name", "item_group", "qty", "line_amount", "imei"])
orders = pd.DataFrame({"order_id": ["1", "2", "3", "4", "5"], "order_date": ["2025-07-01", "2025-07-02", "2025-07-03", "2025-07-04", "2025-08-05"],
                       "customer_name": ["An", "Bình", "Chi", "Dũng", None], "status": ["hoàn thành", "hoàn thành", "hoàn thành", "trả hàng", "hoàn thành"],
                       "order_discount": [100000, 0, 0, 0, 0], "tradein": [3000000, 0, 0, 1000000, 0],
                       "total_payable": [17000000, 8000000, 16000000, 22000000, 600000]})
O, L = dm.finalize(lines, orders)
C = {"orders": O, "lines": L,
     "instalments": pd.DataFrame({"order_id": ["3"], "buy_date": ["2025-07-03"], "customer_name": ["Chi"], "item_name": ["Samsung Galaxy A56 5G 256GB"],
                                  "price": [16000000], "prepaid": [4000000], "n_terms": [6], "per_term": [2000000], "due_day": [5]}),
     "instalment_payments": pd.DataFrame({"order_id": ["3", "3"], "pay_date": ["2025-08-05", "2025-09-10"], "amount": [2000000, 2000000]}),
     "payments": pd.DataFrame({"order_id": ["1", "2", "3", "4", "5"], "pay_date": ["2025-07-01", "2025-07-02", "2025-07-03", "2025-07-04", "2025-08-05"],
                               "amount": [17000000, 5000000, 4000000, 22000000, 600000]}),
     "warranty": pd.DataFrame({"date": ["2025-07-20", "2025-08-01", "2025-08-02"], "item_name": ["iPhone 16", "iPhone 13", "iPhone 16"],
                               "imei": ["111", "222", "111"]})}
S = dm.Shop(C, "2025-09-30")
jul = dm.month(2025, 7)
check("devices_sold Jul (accessories, returned order out)", S.devices_sold(jul), 4)
check("devices_sold new", S.devices_sold(jul, kind="máy mới"), 3)
check("devices_sold used", S.devices_sold(jul, kind="cũ"), 1)
check("devices_sold brand Apple", S.devices_sold(jul, brand="Apple"), 2)
check("devices_sold line", S.devices_sold(jul, line="iPhone 16"), 1)
check("top_lines", S.top_lines(jul, 2), ["Samsung Galaxy A56 5G", "iPhone 13"])
check("device_share new by revenue", S.device_share(jul, kind="máy mới"), dm.pct(36000000, 44000000))
check("avg_device_price", S.avg_device_price(jul), 11000000)
check("tradein_total (returned order out)", S.tradein_total(jul), 3000000)
check("tradein_share", S.tradein_share(jul), dm.pct(1, 3))
check("payable_total", S.payable_total(jul), 41000000)
check("avg_payable", S.avg_payable(jul), dm.money(41000000 / 3))
# instalment: 6 x 2,000,000 = 12,000,000; due 5 Aug, 5 Sep, 5 Oct ...; paid 2m on 5 Aug, 2m on 10 Sep
check("instalment_remaining 30/09", S.instalment_remaining("2025-09-30"), 8000000)
check("instalment_remaining 06/09 (before the 2nd payment)", S.instalment_remaining("2025-09-06"), 10000000)
check("instalment_overdue 06/09 (Sep term due 5/9 unpaid)", S.instalment_overdue("2025-09-06"), 2000000)
check("instalment_overdue 05/09 (due today is not overdue)", S.instalment_overdue("2025-09-05"), 0)
check("n_overdue_customers 06/09", S.n_overdue_customers("2025-09-06"), 1)
check("terms_due 05/09 incl. today", S.terms_due("2025-09-05", customer="Chi"), 2)
# debts: order 2 owes 3m (debt book); order 3 owes its unpaid instalments (not 16m - 4m paid at purchase twice)
check("debt_total incl. instalments", S.debt_total("2025-09-30"), 3000000 + 8000000)
check("n_debtors", S.n_debtors("2025-09-30"), 2)
check("top_debtors", S.top_debtors("2025-09-30", 1)["rows"], [["Chi", 8000000]])
check("debt_flow Aug: instalment paid", S.debt_flow(dm.month(2025, 8))[1], 2000000)
check("warranty_visits Q3", S.warranty_visits(dm.quarter(2025, 3)), 3)
check("warranty_rate Jul", S.warranty_rate(jul), dm.pct(1, 4))
check("top_warranty_lines Aug", S.top_warranty_lines(dm.month(2025, 8), 1), ["iPhone 13"])
check("devices_with_warranty (sold Jul, by 30/09)", S.devices_with_warranty(jul, "2025-09-30"), 2)

# ---------------------------------------------------------------- repair shop
rows = [("2025-07-01", "Đạt", "Air Blade 59A1-111.11", "thay nhớt + vá ruột", 150000, None),
        ("2025-07-01", "a Đạt", "AB 59A1-111.11", "nhớt hộp số", 80000, None),
        ("2025-07-02", "A.Phúc", "Wave Alpha 59B2-222.22", "sửa điện, thay bóng đèn", 200000, None),
        ("2025-07-03", "Phúc", "alpha 59B2-222.22", "bh vá", 0, "bảo hành"),
        ("2025-07-03", "Đạt", "Vision 59C3-333.33", "ruột", 120000, None),
        ("2025-07-09", "Phúc", "vision 110 59C3-334.44", "rửa xe", 30000, None)]
t = pd.DataFrame(rows, columns=["date", "staff", "xe", "work", "money", "note"])
vp = t["xe"].map(split_vehicle)
lines = pd.DataFrame({"order_id": [f"r{i}" for i in range(len(t))], "order_date": t["date"], "staff": t["staff"], "item_name": t["work"],
                      "line_amount": t["money"], "qty": 1, "vehicle": [v for v, _ in vp], "plate": [p for _, p in vp], "work": t["work"],
                      "warranty": t["note"].eq("bảo hành"), "note": t["note"]})
O, L = dm.finalize(lines, dedupe=False)
C = {"orders": O, "lines": L, "staff": pd.DataFrame({"staff": ["Hà Loan Đạt", "Đoàn Bảo Phúc"], "commission_pct": [30, 20]})}
S = dm.Shop(C, "2025-07-31")
check("n_tickets incl. warranty", S.n_tickets(jul), 6)
check("n_tickets paid", S.n_tickets(jul, paid=True), 5)
check("n_tickets of a mechanic (every spelling)", S.n_tickets(jul, staff="Hà Loan Đạt"), 3)
check("job thay nhớt (gear oil out)", S.job_tickets(jul, "thay nhớt"), 1)
check("job dầu láp", S.job_tickets(jul, "thay dầu láp"), 1)
check("job vá ruột (warranty ticket out)", S.job_tickets(jul, "vá ruột"), 1)
check("job thay ruột (plain ruột)", S.job_tickets(jul, "thay ruột"), 1)
check("job sửa điện", S.job_tickets(jul, "sửa điện"), 1)
check("avg_ticket (paid tickets)", S.avg_ticket(jul), dm.money(580000 / 5))
check("warranty_rate", S.warranty_rate(jul), dm.pct(1, 5))
check("n_vehicles (plates, warranty incl.)", S.n_vehicles(jul), 4)
check("n_vehicles of a line (AB = Air Blade)", S.n_vehicles(jul, vehicle="Air Blade"), 1)
check("regular vehicles (>= 2 paid tickets)", S.n_vehicles(jul, min_paid=2), 1)
check("staff_stats tickets", dict(S.staff_stats(jul, "tickets")), {"Hà Loan Đạt": 3, "Đoàn Bảo Phúc": 3})
check("staff_stats n_days", dict(S.staff_stats(jul, "n_days")), {"Hà Loan Đạt": 2, "Đoàn Bảo Phúc": 3})
check("top_staff revenue", S.top_staff(jul, 1, "revenue"), ["Hà Loan Đạt"])
check("commission table", S.commission(jul)["rows"], [["Hà Loan Đạt", 105000], ["Đoàn Bảo Phúc", 46000]])
check("commission one mechanic (given name)", S.commission(jul, staff="Phúc"), 46000)
check("top_vehicles revenue", len(S.top_vehicles(jul, 1)), 1)

# ---------------------------------------------------------------- mother & baby
lines = pd.DataFrame([("1", "2026-05-01", "Tã quần Bobby size XL 44 miếng", "bỉm tã", 2, 600000),
                      ("1", "2026-05-01", "Sữa Similac số 4 900g", "sữa bột", 1, 500000),
                      ("2", "2026-05-02", "Tã dán Huggies size M 60 miếng", "bỉm tã", 3, 900000),
                      ("3", "2026-05-03", "Tã quần Bobby size M 60 miếng", "bỉm tã", 1, 300000)],
                     columns=["order_id", "order_date", "item_name", "item_group", "qty", "line_amount"])
orders = pd.DataFrame({"order_id": ["1", "2", "3"], "order_date": ["2026-05-01", "2026-05-02", "2026-05-03"], "order_discount": [10000, 0, 20000], "points_discount": [0, 5000, 0], "shipping_fee": [0, 15000, 0],
                       "status": ["hoàn thành", "hoàn thành", "huỷ"]})
items = pd.DataFrame({"item_name": ["Tã quần Bobby size XL 44 miếng", "Tã dán Huggies size M 60 miếng", "Tã quần Bobby size M 60 miếng", "Sữa Similac số 4 900g"],
                      "brand": ["Bobby", "Huggies", "Bobby", "Similac"], "size": ["XL", "M", "M", "Số 4"]})
O, L = dm.finalize(lines, orders)
C = {"orders": O, "lines": L, "items": items,
     "customers": pd.DataFrame({"customer_id": ["K1", "K2", "K3"], "registered": ["2026-04-30", "2026-05-10", "2026-05-31"]})}
S = dm.Shop(C, "2026-05-31")
may = dm.month(2026, 5)
check("voucher_total (cancelled out)", S.voucher_total(may), 10000)
check("points_total", S.points_total(may), 5000)
check("shipping_total", S.shipping_total(may), 15000)
check("new_members", S.new_members(may), 2)
check("pack_share size M", S.pack_share(may, size="M"), dm.pct(3, 5))
check("pack_share brand", S.pack_share(may, brand="Bobby"), dm.pct(2, 5))
check("size_qty Số 4", S.size_qty(may, "Số 4"), 1)
check("top_brands diapers", S.top_brands(may, 1, group="bỉm tã"), ["Huggies"])

print("FAILED" if fails else "PASS", fails)
sys.exit(1 if fails else 0)
