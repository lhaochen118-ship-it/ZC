#!/usr/bin/env python3
"""Robustness test of the generic ingestion on SYNTHETIC messy files (formats not taken from the competition data):
every variant encodes the same clean table in a different messy way; ingest must recover the clean values.
Usage: test_ingest_robust.py [scratch_dir]"""
import os, sys, csv, shutil, tempfile, datetime as dt
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import ingest

rng = np.random.default_rng(0)
N = 60
CLEAN = pd.DataFrame({
    "code": [f"0{100 + i}" for i in range(N)],                                   # leading-zero codes must stay text
    "date": [pd.Timestamp(2025, 1, 1) + pd.Timedelta(days=int(d)) for d in rng.integers(0, 500, N)],
    "qty": rng.integers(1, 20, N).astype(float),
    "price": (rng.integers(5, 2000, N) * 1000).astype(float),
    "rate": np.round(rng.uniform(-20, 40, N), 2),
    "name": [f"Sản phẩm {chr(65 + i % 26)}{i}" for i in range(N)],
})

VI_MONTH = lambda d: f"Ngày {d.day} tháng {d.month} năm {d.year}"


def money_variants(v, k):
    s = f"{int(v):,}"
    return [s.replace(",", "."), s, s.replace(",", ".") + " đ", "VND " + s, s.replace(",", " "), f"{int(v) // 1000}k",
            (f"{int(v) // 1_000_000}tr{int(v) % 1_000_000 // 1000:03d}" if v >= 1_000_000 else s.replace(",", ".") + "đ")][k % 7]


def date_variants(d, k):
    return [d.strftime("%d/%m/%Y"), d.strftime("%Y-%m-%d"), d.strftime("%d-%m-%Y"), d.strftime("%d.%m.%Y"), VI_MONTH(d),
            f"{d.day} thg {d.month} {d.year}", d.strftime("%Y/%m/%d"), d.to_pydatetime()][k % 8]


def rows_of(variant):
    """header + body rows (lists of cells) for a messy variant."""
    hdr = ["Mã hàng", "Ngày bán", "Số lượng", "Đơn giá", "Tỉ lệ", "Tên hàng"]
    body = []
    for i, r in CLEAN.iterrows():
        k = i if variant != "uniform" else 0
        qty = f"{int(r.qty)} cái" if k % 3 == 0 else (int(r.qty) if k % 3 == 1 else str(int(r.qty)))
        rate = (f"{r.rate:.2f}".replace(".", ",") + "%") if k % 2 else r.rate
        if variant == "paren_neg" and r.rate < 0: rate = f"({abs(r.rate):.2f})"
        body.append([r.code, date_variants(r.date, k), qty, money_variants(r.price, k), rate, r["name"]])
    return hdr, body


def write_variant(d, name, variant):
    hdr, body = rows_of(variant)
    p = os.path.join(d, name)
    if name.endswith(".csv"):
        enc = "utf-16" if variant == "utf16" else "utf-8-sig"
        sep = ";" if variant in ("semicolon", "utf16") else ","
        with open(p, "w", encoding=enc, newline="") as f:
            w = csv.writer(f, delimiter=sep)
            w.writerow(hdr)
            for r in body: w.writerow([x.strftime("%d/%m/%Y") if isinstance(x, dt.datetime) else x for x in r])
        return p
    import openpyxl
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Tháng 1"
    if variant in ("title_rows", "page_headers", "two_row_header"):
        ws.append(["BÁO CÁO BÁN HÀNG"]); ws.append(["Từ ngày 01/01/2025 đến 31/12/2026"]); ws.append([])
    if variant == "two_row_header":
        ws.append(["Thông tin hàng", None, "Bán ra", None, None, None]); ws.append(["Mã hàng", "Ngày bán", "Số lượng", "Đơn giá", "Tỉ lệ", "Tên hàng"])
    else: ws.append(hdr)
    for i, r in enumerate(body):
        ws.append(r)
        if variant == "page_headers" and i % 25 == 24: ws.append(hdr)                 # header repeated every page
    if variant in ("title_rows", "page_headers", "two_row_header"):
        ws.append(["Tổng cộng", None, None, "999.999.999", None, None]); ws.append(["Ghi chú: số liệu nhập tay"])
    wb.save(p); return p


def check_dup_subheaders():
    """group row (merged cells) over repeated sub-headers: Nhập/Xuất x Số lượng/Thành tiền."""
    import openpyxl
    d = tempfile.mkdtemp(); p = os.path.join(d, "kho.xlsx"); wb = openpyxl.Workbook(); ws = wb.active
    ws.append(["THẺ KHO"]); ws.append([None, None, "Nhập", None, "Xuất", None])
    ws.append(["Mã hàng", "Ngày", "Số lượng", "Thành tiền", "Số lượng", "Thành tiền"])
    for i in range(30): ws.append([f"SP{i:03d}", f"{i % 28 + 1:02d}/03/2025", i, f"{i * 1000:,}", i + 1, f"{(i + 1) * 2000:,}"])
    wb.save(p); biz = ingest.load_business(d); shutil.rmtree(d)
    (name, df), = biz.tables.items(); errs = []
    for c, want in [("Nhập - Số lượng", list(range(30))), ("Xuất - Thành tiền", [(i + 1) * 2000 for i in range(30)])]:
        if c not in df.columns: errs.append(f"missing column {c!r} (have {[x for x in df.columns if not x.startswith('_')]})")
        elif list(df[c]) != want: errs.append(f"{c} values wrong")
    return errs


def check_decimal(sep):
    """weights with decimals: '1.500' must be 1.5 when the column clearly uses '.' as the decimal separator."""
    d = tempfile.mkdtemp(); p = os.path.join(d, "can.csv")
    vals = [1.5, 2.25, 0.75, 1.5, 12.0, 3.125, 0.5, 1.75] * 4
    txt = ["1.5", "2.25", "0.75", "1.500", "12", "3.125", "0.5", "1.75"] * 4
    if sep == ",": txt = [t.replace(".", ",") for t in txt]
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";"); w.writerow(["Mã", "Khối lượng (kg)", "Thành tiền"])
        for i, t in enumerate(txt): w.writerow([f"H{i}", t, f"{(i + 1) * 1500:,}".replace(",", "." if sep == "," else ",")])
    biz = ingest.load_business(d); shutil.rmtree(d)
    (name, df), = biz.tables.items(); errs = []
    got = list(df["Khối lượng (kg)"])
    if got != vals: errs.append(f"weights {got[:8]} != {vals[:8]}")
    if list(df["Thành tiền"]) != [(i + 1) * 1500 for i in range(len(txt))]: errs.append(f"money {list(df['Thành tiền'])[:4]}")
    return errs


def check(variant, fname):
    d = tempfile.mkdtemp(); write_variant(d, fname, variant)
    biz = ingest.load_business(d); shutil.rmtree(d)
    (name, df), = biz.tables.items()
    cols = [c for c in df.columns if not c.startswith("_") and not c.endswith("__unit")]
    errs = []
    if len(df) != N: errs.append(f"rows {len(df)} != {N}")
    want = {"Mã hàng": ("code", "text"), "Ngày bán": ("date", "datetime"), "Số lượng": ("qty", "number"),
            "Đơn giá": ("price", "number"), "Tỉ lệ": ("rate", "number"), "Tên hàng": ("name", "text")}
    for c in cols:
        key = next((k for k in want if k in c), None)
        if key is None: continue
        src, typ = want[key]; got_t = biz.meta[name]["types"].get(c)
        if got_t != typ: errs.append(f"{c}: type {got_t} != {typ}"); continue
        if len(df) != N: continue
        a, b = df[c].reset_index(drop=True), CLEAN[src]
        if typ == "number": bad = ~np.isclose(a.astype(float), b.astype(float))
        elif typ == "datetime": bad = pd.to_datetime(a).dt.normalize() != b
        else: bad = a.astype(str) != b.astype(str)
        if bad.any(): errs.append(f"{c}: {int(bad.sum())} wrong, e.g. got {a[bad].iloc[0]!r} want {b[bad].iloc[0]!r}")
    return errs


if __name__ == "__main__":
    cases = [("uniform", "a.xlsx"), ("mixed", "a.xlsx"), ("mixed", "a.csv"), ("semicolon", "a.csv"), ("utf16", "a.csv"),
             ("title_rows", "a.xlsx"), ("page_headers", "a.xlsx"), ("two_row_header", "a.xlsx"), ("paren_neg", "a.xlsx")]
    fails = 0
    for v, f in cases:
        e = check(v, f); fails += bool(e)
        print(f"{'PASS' if not e else 'FAIL'} {v:15s} {f:7s} {'; '.join(e)[:300]}")
    for nm, fn in [("dup_subheaders", check_dup_subheaders), ("decimal_dot", lambda: check_decimal(".")), ("decimal_comma", lambda: check_decimal(","))]:
        e = fn(); fails += bool(e); cases.append(nm)
        print(f"{'PASS' if not e else 'FAIL'} {nm:23s} {'; '.join(e)[:300]}")
    print(f"{len(cases) - fails}/{len(cases)} variants recovered")
