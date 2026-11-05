#!/usr/bin/env python3
"""Deterministic mapping of one business's raw tables onto the canonical tables (dm.CANON_DOC), without an LLM.

Why: the LLM-written mapping programs were the main source of error and of run-to-run variance (the same business got
3 different mappings in 3 runs; the public score moved +-0.03 from that alone). The data packages are exports of one
canonical model (orders, lines, catalogue, purchases, payments, stock ...) written with a closed vocabulary of column
names and README descriptions ("mã đơn", "thành tiền sau giảm", "số lượng đã đổi ra đơn vị cơ bản" ...). Inverting
that export is a deterministic problem:

  1. README parsing: column descriptions per file / sheet (`## Cột trong ...`), bold-name definitions in the
     conventions ("**Còn nợ** là số khách còn nợ của đơn"), the data end date, unit conversions.
  2. Column roles: every raw column gets a canonical role from (a) its README description, (b) its name (an
     accent-free lexicon learned from the training mappings + generic Vietnamese words), (c) its values (dates with
     a time of day, status / payment / channel vocabularies, codes, money).
  3. Table kinds: sales lines, order headers, catalogue, conversions, customers, purchases, payments, stock, lots ...
     from the roles they carry.
  4. Assembly: the same toolkit calls the reference programs make (frame / finalize / conversions_table), plus
     convention handlers switched on by README phrases or data patterns (negative quantities = returns, status
     written in notes, toppings in the dish cell, voucher / points / shipping pseudo-lines, abbreviation sheets ...).

Nothing here is keyed on a business id, a file name of a particular business or a question. Validation:
tests/eval_split.py (answers of the rule programs on train and public), tests/test_automap.py."""
import os, re, sys, math, fnmatch, unicodedata
from collections import OrderedDict, defaultdict, Counter
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dm, ingest


# ------------------------------------------------------------------ text helpers
def fold(s):
    """accent-free, lowercase, punctuation -> space ('Mã H.hóa' -> 'ma h hoa', 'SL (ĐV cơ bản)' -> 'sl dv co ban')."""
    s = unicodedata.normalize("NFD", str(s)).replace("đ", "d").replace("Đ", "D")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%]+", " ", s)).strip()


def fold_name(c):
    """column name -> folded words; CamelCase names of database exports are split ('MaHoaDon' -> 'ma hoa don')."""
    c = str(c)
    if re.fullmatch(r"[A-Za-z0-9_]+", c) and re.search(r"[a-z][A-Z]|[A-Z]{2}[a-z]", c):
        c = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", c)
    return fold(c.replace("_", " "))


# ------------------------------------------------------------------ 1. README
def parse_readme(text):
    """-> dict(sections=[(file_pattern, sheet|None, {col: desc})], defs={col: desc} (bold definitions anywhere),
    conventions=[bullet text], intro=str, data_end=Timestamp|None)."""
    text = str(text or "")
    out = {"sections": [], "defs": {}, "conventions": [], "intro": "", "data_end": None, "qualified": {}}
    m = re.search(r"có đến hết (\d{1,2})/(\d{1,2})/(\d{4})", text)
    if m: out["data_end"] = pd.Timestamp(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    cur, in_conv, intro = None, False, []
    for line in text.split("\n"):
        h = re.match(r"^##\s+(.+?)\s*$", line)
        if h:
            title = h.group(1); cur = None; in_conv = fold(title).startswith("quy uoc")
            mm = re.match(r"Cột trong\s+`([^`]+)`(?:\s*,\s*sheet\s+`([^`]+)`)?", title)
            if mm:
                cur = {"file": mm.group(1), "sheet": mm.group(2), "cols": OrderedDict()}
                out["sections"].append(cur)
            continue
        if line.startswith("# "): continue
        b = re.match(r"^\s*[-*]\s*\*\*(.+?)\*\*\s*:\s*(.+?)\s*$", line)
        if cur is not None and b:
            cur["cols"][b.group(1).strip()] = b.group(2).strip(); continue
        if in_conv and re.match(r"^\s*[-*]\s+", line):
            out["conventions"].append(re.sub(r"^\s*[-*]\s+", "", line).strip())
        elif in_conv and out["conventions"] and line.startswith("  "):
            out["conventions"][-1] += "\n" + line.strip()
        elif not in_conv and cur is None and not line.startswith("-") and line.strip():
            intro.append(line.strip())
    out["intro"] = "\n".join(intro)
    # bold definitions: '**Tồn cuối kỳ** là tồn cuối ...', '`chi_tiet_don.TriGia` là thành tiền của dòng'
    for conv in out["conventions"] + intro:
        for mm in re.finditer(r"\*\*([^*]+?)\*\*\s*(?:\([^)]*\)\s*)?là\s+([^;.]+(?:\([^)]*\))?[^;.]*)", conv):
            out["defs"].setdefault(mm.group(1).strip(), mm.group(2).strip())
        for mm in re.finditer(r"`(\w+)\.(\w+)`\s*(?:\([^)]*\)\s*)?là\s+([^;.]+(?:\([^)]*\))?[^;.]*)", conv):
            out["qualified"][(mm.group(1), mm.group(2))] = mm.group(3).strip()
    return out


def _file_matches(pattern, src_file):
    """README file pattern ('BanHang_T<tháng>_<năm>.xlsx') against an ingested source file name."""
    p = fold(re.sub(r"<[^>]+>", "*", pattern)).replace(" ", ""); s = fold(src_file).replace(" ", "")
    p = re.sub(r"\*+", "*", p)
    return fnmatch.fnmatch(s, p) or fnmatch.fnmatch(s, p.replace("xlsx", "*").replace("csv", "*"))


def table_descriptions(biz, name, rd):
    """{column: description} for one ingested table: its README section(s) + bold definitions of its columns."""
    df = biz.tables[name]; srcs = list(biz.meta[name]["sources"])
    out = {}
    for sec in rd["sections"]:
        for src in srcs:
            f, sheet = (src.split(":")[0], None) if ":" in src and "[" not in src else (re.sub(r"\[.*$", "", src), (re.search(r"\[(.*)\]$", src) or [None, None])[1])
            if _file_matches(sec["file"], f) and (sec["sheet"] is None or fold(sec["sheet"]) == fold(sheet or "")):
                for c, d in sec["cols"].items():
                    if c in df.columns: out.setdefault(c, d)
                break
    for c in df.columns:
        if c in rd["defs"] and c not in out: out[c] = rd["defs"][c]
        for (t, cc), d in rd["qualified"].items():
            if cc == c and (t == name or any(src.endswith(":" + t) for src in srcs)): out[c] = d
    return out


# ------------------------------------------------------------------ 2. column roles
# README description (folded) -> role. First match wins: specific phrases first. Roles of side tables carry a prefix
# that the table kind resolves (e.g. 'qty' in a purchase table is the received quantity).
DESC_RULES = [
    (r"^cot trong", "empty"),
    (r"so thu tu don .*trong so|so thu tu don hoa don|^mot don$|^mot hoa don$|^mot phieu$", "order_id"),
    (r"so thu tu dong", "line_no"),
    (r"^ma don nhap|^ma phieu nhap", "receipt_id"),
    (r"^ma phieu thu|^ma thanh toan", "payment_id"),
    (r"^ma don|^so hoa don|^ma hoa don|^ma bill|^ma chung tu", "order_id"),
    (r"^ma nha cung cap|^nha cung cap", "supplier"),
    (r"^ma nhan vien", "staff"),
    (r"^ma khach", "customer_id"),
    (r"^ma hang|^ma san pham|^ma sku|^ma vach", "item_code"),
    (r"^so lo|^ma lo", "lot_id"),
    (r"^ten khach|khach thanh vien|^ten kh\b", "customer_name"),
    (r"^ho ten tho|^ten tho", "staff_name"),
    (r"^tho lam phieu|^nhan vien ban|^tho\b", "staff"),
    (r"^dong xe va bien so", "vehicle"),
    (r"^cac viec da lam|^viec da lam", "tasks"),
    (r"^ten goi khac", "alias_names"),
    (r"^ten o nha cua be", "baby_name"),
    (r"^hang ban|^ten hang|^ten mat hang|^mat hang\b|^ten san pham|^ten$", "item_name"),
    (r"^hang ap dung", "promo_items"),
    (r"^ten chuong trinh", "promo_name"),
    (r"^hang$", "brand"),
    (r"^so cua sua|size", "size"),
    (r"^quy doi don vi lon", "conversion_text"),
    (r"^nhom hang", "item_group"),
    (r"^nhom khach|^loai khach", "customer_group"),
    (r"^so luong da doi ra don vi co ban|^so luong quy doi", "qty_base"),
    (r"^so luong nhap \(?don vi co ban|^so luong nhap don vi co ban", "qty_in_base"),
    (r"^so luong ban lay tu lo", "lot_qty_base"),
    (r"^so luong khach tra lai", "returned_base"),
    (r"^so luong giao", "qty_base"),
    (r"^so luong nhap", "qty"),
    (r"^so luong", "qty"),
    (r"^don vi co ban|^dv goc|^dvt$", "base_unit"),
    (r"^don vi lon", "big_unit"),
    (r"^don vi nhap", "unit"),
    (r"^don vi tinh", "unit"),
    (r"^he so", "factor"),
    (r"^gia nhap mot don vi|^gia nhap", "unit_cost"),
    (r"^gia von mot don vi ban|^gia von luc ban", "unit_cost"),
    (r"^gia von", "cost"),
    (r"^gia ban hien tai|^gia hien tai", "price"),
    (r"^gia ban ap dung tu ngay|^gia moi", "new_price"),
    (r"^gia truoc lan doi|^gia cu", "old_price"),
    (r"^ngay bat dau ap dung", "price_date"),
    (r"^don gia", "unit_price"),
    (r"^thanh tien sau giam|^thanh tien cua dong|^thanh tien cac dong", "line_amount"),
    (r"^tien nhap cua ca dong", "amount"),
    (r"^tong tien phieu nhap|tong tien cua ca phieu nhap", "receipt_total"),
    (r"^tien hang cua don|tong tien hang cua (ca )?don|^tong tien hang", "gross"),
    (r"^tien giam cua ca dong|^tien giam cua dong", "line_discount"),
    (r"^giam gia cua ca don|^tien bot cho ca don", "order_discount"),
    (r"^giam gia", "discount"),
    (r"^phi giao hang", "shipping_fee"),
    (r"^tien thu may cu", "tradein"),
    (r"^so tien khach tra cho ca don|^khach da tra|^da thanh toan|^da thu", "paid_total"),
    (r"^khach con no|^so khach con no|^con no|^cong no|^so tien cua don chua thu|^so con no|^tien con no|chua thu tinh den", "debt_end"),
    (r"^khach can tra cho ca don|^so khach phai tra cho ca don|^tong tien don|^tong tien ca phieu|^tong tien cua ca phieu", "total_payable"),
    (r"^trang thai phieu nhap", "receipt_status"),
    (r"^trang thai", "status"),
    (r"^hinh thuc thanh toan cua don", "payment_method"),
    (r"^hinh thuc thanh toan", "method"),
    (r"^kenh ban", "channel"),
    (r"^thoi diem dat don|^gio dat don|^gio tao don", "order_time"),
    (r"^thoi diem thu", "pay_time"),
    (r"^ngay ban|^ngay sua", "order_date"),
    (r"^ngay nhap", "receipt_date"),
    (r"^ngay giao", "delivery_date"),
    (r"^ngay dang ky", "member_since"),
    (r"^ngay vao lam", "start_date"),
    (r"^ngay tao", "created"),
    (r"^tu ngay", "date_from"),
    (r"^den ngay", "date_to"),
    (r"^han su dung", "expiry"),
    (r"^han thanh toan", "due_date"),
    (r"^so ngay duoc no", "term_days"),
    (r"^dot giao", "trip_no"),
    (r"^bien so", "plate"),
    (r"^tai xe", "driver"),
    (r"^ton sau|^ton o dong cuoi", "stock_after"),
    (r"^ton kho|^ton cuoi", "stock_qty"),
    (r"^hoat chat", "active_ingredient"),
    (r"^thuoc ke don|^ke don", "prescription"),
    (r"^con kinh doanh", "active"),
    (r"^so dien thoai", "phone"),
    (r"^dia chi", "address"),
    (r"^ban$", "table_no"),
    (r"^khu vuc", "area"),
    (r"^so khach cua don", "guests"),
    (r"^imei", "imei"),
    (r"^ghi chu", "note"),
    (r"^phan tram hoa hong", "commission_pct"),
    (r"^diem con", "points"),
    (r"^so tien", "amount"),
    (r"^tong tien ca phieu", "line_amount"),
]

# folded column name -> role (exact names). Learned from the training reference mappings, plus generic words.
NAME_RULES = [
    # keys
    (r"ma don hang|ma don|ma hoa don|ma hd|so hd|so hoa don|ma bill|bill|so ct|ma chung tu|ma giao dich|ma gd|don so|order id|hd|ma phieu ban|mahoadon", "order_id"),
    (r"ma phieu nhap|ma don nhap|ma nhap hang|so phieu nhap|so phieu|ma phieu|maphieunhap", "receipt_id"),
    (r"so phieu|ma phieu", "order_id"),
    (r"ma phieu thu|ma thanh toan|ma tt|ma pt", "payment_id"),
    (r"ma phieu tra", "return_id"),
    (r"no|stt|so dong|so thu tu|thu tu|tt dong|dong|line|line no", "line_no"),
    (r"ma hang|ma hang hoa|ma h hoa|ma hh|ma san pham|ma sp|ma sku|sku|id sp|ma mat hang|ma mon|ma hang ban|ma san pham ban|barcode|ma vach|mahanghoa|ma sp ban", "item_code"),
    (r"ma khach|ma kh|ma khach hang|ma khach mua|ma kh mua|ma nguoi mua|ma the|customer id|makhachhang|so kh", "customer_id"),
    (r"ma nv|ma nv ban|ma nhan vien|nhan vien ban|nguoi ban|manhanvien|nhan vien|nv ban", "staff"),
    (r"ma ncc|ma nha cung cap|nha cung cap|ncc|manhacungcap", "supplier"),
    (r"ma lo|lo|so lo|malo|lo hang", "lot_id"),
    # dates
    (r"ngay|ngay ban|ngay dat hang|ngay hd|ngay xuat|ngay giao dich|ngay thang|date|ngay lap|ngay mua|ngayban|ngaylap|thoi gian|ngay chung tu|ngay sua", "order_date"),
    (r"gio|gio ban|gio tao don|thoi gian tao|thoi gian lap|thoigianlap|gio dat|thoi diem dat|time", "order_time"),
    (r"ngay nhap|ngay nhap kho|ngaynhapkho", "receipt_date"),
    (r"ngay thu|ngay tt|ngay gio thu|ngaygiothu|thoi gian thu|thoi diem thu|ngay tra", "pay_time"),
    (r"ngay giao|ngay giao hang|thoi gian giao|ngay cho", "delivery_date"),
    (r"han su dung|hsd|hansudung|han dung", "expiry"),
    (r"han thanh toan|han tra|ngay phai tra|ngay den han", "due_date"),
    (r"ngay ap dung|ngayapdung|ngay hieu luc", "price_date"),
    # items
    (r"ten hang|ten hang hoa|hang hoa|hang ban|hang|mat hang|ten mat hang|san pham|ten san pham|ten sp|mon|ten mon|product|item|chi tiet hang|noi dung|tengoi|ten goi|tenhang|ten hang ban|ten hang hoa ban", "item_name"),
    (r"nhom hang|nhom hang hoa|nhom|loai hang|loai san pham|danh muc|phan loai|danhmuc|nganh hang", "item_group"),
    (r"dvt|don vi tinh|don vi|dv|dv tinh|d vi|unit|donvitinh|donvi|dvt ban", "unit"),
    (r"dv goc|don vi co ban|dvt co ban|don vi goc|dv co ban|donvicoso|dv nho", "base_unit"),
    (r"dvt nhap|don vi nhap", "unit"),
    (r"he so|ti le quy doi|ty le quy doi|he so qd|bang|he so quy doi|hesoquydoi|quy doi", "factor"),
    (r"loai dv|don vi quy doi|dv lon|quy cach|dv nhap|don vi lon", "big_unit"),
    # quantities / money
    (r"sl|s l|slg|sluong|so luong|sl ban|so luong ban|so luong hang|so luong san pham|soluong|so luong mua|sl nhap|so luong nhap", "qty"),
    (r"sl le|so luong le|sl quy doi|so luong quy doi|quy ra dv le|sl co ban|sl dv co ban|sl quy ra le|slquydoi|so luong dvt co ban|sl thuc giao|so luong giao", "qty_base"),
    (r"sl nhap quy doi|so luong nhap quy doi|sl nhap dvt co ban|so luong nhap dvt co ban|sl nhap co ban", "qty_in_base"),
    (r"sl tra quy doi|so luong tra quy doi|sl tra dvt co ban|so luong tra dvt co ban|sl tra", "returned_base"),
    (r"don gia|don gia d|gia 1 dv|gia ban|gia|gia le|gia san pham|gia niem yet|gia don vi|giadonvi|price|dg|don gia ban|gia ban le", "unit_price"),
    (r"thanh tien|thanh tien d|tt|amount|tri gia|trigia|tien|cong|tien hang|tong|t tien|thanh tien sau giam|tien ban", "line_amount"),
    (r"giam d|giam|chiet khau|ck|giam gia|khuyen mai|chietkhau|tien giam", "discount"),
    (r"gia von|gia von luc ban|giavonlucban|gia von hien tai|giavonhang|gia von hang", "cost"),
    (r"gia nhap|don gia nhap|dongianhap|gia nhap don vi", "unit_cost"),
    (r"tong tien hang|tien hang truoc giam|gia tri hang|tong tien hang hoa|tongtienhang", "gross"),
    (r"giam gia hd|giam gia don|ck don|chiet khau don|bot", "order_discount"),
    (r"phi ship|phi giao hang|phi van chuyen|phivanchuyen|ship", "shipping_fee"),
    (r"thue|thue gtgt|vat|thuegtgt", "vat"),
    (r"tong thanh toan|tong tien|tongtien|phai thu|tong bill|tong don|tong sau giam gia|tongthanhtoan|can tra|tong phieu|khach can tra|tong cong", "total_payable"),
    (r"khach tra|da tra|da thu|khach da tra|da thanh toan|khachdatra|tien khach tra", "paid_total"),
    (r"con no|cong no|no|chua tra|du no|con phai thu|conphaithu", "debt_end"),
    (r"so tien|so tien thu|tien thu|sotienthu|gia tri", "amount"),
    (r"tong tien phieu|tong tien phieu nhap|tong gia tri nhap|tonggiatrinhap|can tra ncc", "receipt_total"),
    # labels
    (r"trang thai|trang thai don|trang thai don hang|tinh trang|tt don|tien do|xu ly|ket qua|status|trang thai xu ly|trang thai giao|tinhtrang|ghi chu tt", "status"),
    (r"trang thai phieu|trang thai phieu nhap", "receipt_status"),
    (r"khach|khach hang|ten khach|ten khach hang|ho ten khach|ten kh|khach hang ten|doi tac|ben mua|tenkh|khach mua|khach hang mua|khach le si", "customer_name"),
    (r"nhom khach hang|nhom khach|loai khach|loai khach hang|hang khach|kieu khach|hang", "customer_group"),
    (r"kenh|kenh ban|nguon|kenh dat|nguon don|ban qua|loai ban|kenhbanhang|kenh ban hang", "channel"),
    (r"thanh toan bang|phuong thuc tt|hinh thuc thanh toan|phuong thuc thanh toan|httt|pttt|tra bang|phuongthuctt|loai tt|phuong thuc|phuongthuc|hinh thuc tt", "payment_method"),
    (r"ghi chu|note|luu y|dien giai|mo ta|ghi chu don|noi dung", "note"),
    (r"sdt|so dt|so dien thoai|phone|dien thoai|dt|sodienthoai", "phone"),
    (r"dia chi|addr|address|diachilienhe", "address"),
    (r"ban|phong ban|so ban", "table_no"),
    (r"khu vuc", "area"),
    (r"so khach", "guests"),
    (r"ton cuoi|ton kho|ton thuc|ton cuoi ky|ton", "stock_qty"),
    (r"ton sau|ton sau gd|ton cuoi", "stock_after"),
    (r"ngay gio phat sinh|thoi gian phat sinh|thoi diem phat sinh", "event_time"),
    (r"loai giao dich|kieu chung tu|loai chung tu", "event_kind"),
    (r"bien so xe|bien so", "plate"),
    (r"luot giao|lan giao|dot giao|chuyen", "trip_no"),
    (r"lai xe|tai xe|nguoi cho|nguoi giao hang", "driver"),
    (r"ky han|so ngay no|han no ngay|han no", "term_days"),
    (r"hoat chat|thanh phan chinh|thanhphanchinh", "active_ingredient"),
    (r"thuoc ke don|ke don|ban theo don|bantheodon", "prescription"),
    (r"gia cu", "old_price"),
    (r"gia moi", "new_price"),
    (r"imei", "imei"),
    (r"so ky|so thang gop|so ky gop|ky han gop", "n_terms"),
    (r"gop thang|gop moi thang|tien gop thang|gop ky|tien moi ky|gop moi ky", "per_term"),
    (r"tra truoc|dat coc|tra lan dau", "prepaid"),
    (r"han gop ngay|han gop|ngay han gop|ngay gop hang thang", "due_day"),
    (r"ngay gop|ngay dong gop", "inst_pay_date"),
    (r"da gop|tien da gop|so tien gop", "inst_paid"),
    (r"ngay nhan|ngay nhan may|ngay tiep nhan", "received_date"),
    (r"loi|loi may|tinh trang loi|hong", "fault"),
    (r"gia thu|gia thu mua|tien thu may|gia thu may", "tradein_price"),
    (r"tru vao stt so ban|tru vao|tru vao don|stt so ban", "tradein_order"),
    (r"may thu|may cu thu", "tradein_item"),
    (r"may|ten may|dong may|model|may ban", "item_name"),
    (r"ngay tra|ngay khach tra", "pay_time"),
]
_DESC = [(re.compile(p), r) for p, r in DESC_RULES]
_NAME = [(re.compile(r"^(?:" + p + r")$"), r) for p, r in NAME_RULES]


def desc_role(desc):
    d = fold(desc)
    for rx, r in _DESC:
        if rx.search(d): return r
    return None


def name_roles(col):
    """every role whose name pattern matches the column name exactly (several: 'thành tiền' is a line amount in a
    sales table and an amount in a purchase table)."""
    n = fold_name(col)
    n = re.sub(r"^(?:\d+\s+)", "", n)
    return [r for rx, r in _NAME if rx.match(n)]


STATUS_WORDS = re.compile(r"\b(xong|hoan thanh|hoan tat|da giao|huy|tra hang|hoan hang|dang giao|dang van chuyen|khach tra|giao that bai|ok|done|completed|cancel)\b")
PAY_WORDS = re.compile(r"\b(tien mat|chuyen khoan|vi dien tu|cod|ghi no|cash|transfer|ewallet|debt|the|momo|thu ho)\b")
CHAN_WORDS = re.compile(r"\b(tai quay|store|zalo|shopee|lazada|tiktok|san tmdt|phone|facebook|website|grab|tai tiem|dien thoai|online|offline)\b")
WALKIN_WORDS = re.compile(r"^(khach le|le|kl|k l|khach vang lai|vang lai|guest)$")


def profile(s):
    """cheap value profile of one column."""
    nn = s.dropna()
    p = {"n": len(s), "fill": float(len(nn) / max(len(s), 1)), "dtype": "empty" if not len(nn) else
         "datetime" if pd.api.types.is_datetime64_any_dtype(s) else "number" if pd.api.types.is_numeric_dtype(s) else "text"}
    if not len(nn): return p
    smp = nn.head(3000)
    p["nunique"] = int(nn.nunique()); p["uniq"] = p["nunique"] / max(len(nn), 1)
    if p["dtype"] == "datetime":
        t = pd.to_datetime(smp, errors="coerce").dropna()
        p["has_time"] = float(((t.dt.hour != 0) | (t.dt.minute != 0)).mean()) if len(t) else 0.0
    elif p["dtype"] == "number":
        v = pd.to_numeric(smp, errors="coerce").dropna()
        p["int"] = float((v == v.round()).mean()) if len(v) else 0.0; p["min"] = float(v.min()) if len(v) else 0; p["max"] = float(v.max()) if len(v) else 0
        p["neg"] = float((v < 0).mean()) if len(v) else 0.0
    else:
        v = smp.astype(str).map(fold)
        p["status"] = float(v.map(lambda x: bool(STATUS_WORDS.search(x))).mean())
        p["pay"] = float(v.map(lambda x: bool(PAY_WORDS.search(x))).mean())
        p["chan"] = float(v.map(lambda x: bool(CHAN_WORDS.search(x))).mean())
        p["code"] = float(smp.astype(str).str.fullmatch(r"[A-Za-z]{1,6}[-_]?\d{2,}").mean())
        p["phone"] = float(smp.astype(str).str.replace(r"\D", "", regex=True).str.fullmatch(r"0?\d{9,10}").mean())
        p["walkin"] = float(v.map(lambda x: bool(WALKIN_WORDS.match(x))).mean())
        p["len"] = float(smp.astype(str).str.len().mean())
    return p


def column_roles(df, descs):
    """{column: [roles, best first]}: the README description decides when it is known; otherwise the name lexicon."""
    out = {}
    for c in df.columns:
        if str(c).startswith("_") or str(c).endswith("__unit"): continue
        rs = []
        if not df[c].notna().any(): out[c] = []; continue          # an empty column has no role
        d = descs.get(c)
        if d and fold(d) not in ("ten", "ten goi"):                  # a bare 'tên' depends on the table: the name decides
            r = desc_role(d)
            if r: rs.append(r)
        for r in name_roles(c):
            if r not in rs: rs.append(r)
        if not rs and d and fold(d) in ("ten", "ten goi"): rs.append("item_name")
        out[c] = rs
    return out


# ------------------------------------------------------------------ 3. table kinds
class TableInfo:
    """roles + profiles of one raw table."""
    def __init__(self, biz, name, rd):
        self.name, self.df = name, biz.tables[name]
        self.descs = table_descriptions(biz, name, rd)
        self.roles = column_roles(self.df, self.descs)
        self.prof = {c: profile(self.df[c]) for c in self.roles}
        self.kind, self.score = None, 0.0

    def cols(self, *roles, first=False, dtype=None):
        """columns carrying one of the roles (as their first role when first=True), README-described ones first."""
        out = []
        for c, rs in self.roles.items():
            hit = (rs[:1] if first else rs)
            if any(r in hit for r in roles) and (dtype is None or self.prof[c]["dtype"] in ((dtype,) if isinstance(dtype, str) else dtype)):
                out.append(c)
        rank = {r: i for i, r in enumerate(roles)}
        return sorted(out, key=lambda c: (0 if c in self.descs else 1, min(rank.get(r, 99) for r in self.roles[c]), self.roles[c].index(next(r for r in self.roles[c] if r in roles))))

    def col(self, *roles, first=False, dtype=None, exclude=()):
        for c in self.cols(*roles, first=first, dtype=dtype):
            if c not in exclude: return c
        return None

    def has(self, *roles, first=False, dtype=None): return self.col(*roles, first=first, dtype=dtype) is not None

    def uniq(self, c): return self.prof[c].get("uniq", 0.0) if c else 0.0

    def __repr__(self): return f"<{self.name}: {self.kind} {len(self.df)} rows>"


ITEM = ("item_code", "item_name")
DATE = ("datetime",)
NUM = ("number",)


def classify(ti):
    """the kind of a raw table from the roles of its columns (None = not used by the canonical model)."""
    n = len(ti.df); f = lambda *r, **k: ti.has(*r, **k)
    item = f(*ITEM, first=True)
    oid = ti.col("order_id", first=True) or ti.col("order_id")
    # industry books and side lists first (they also carry item / order columns)
    if f("vehicle") and f("tasks"): return "repair_tickets"
    if f("n_terms") and (f("per_term") or f("prepaid")): return "instalments"
    if f("tradein_price") and (f("tradein_order") or f("tradein_item")): return "tradein_book"
    if f("received_date") and (f("imei") or f("fault")): return "warranty_book"
    if not f("item_code", first=True) and f("debt_end") and f("paid_total", "amount") and f("pay_time", dtype=DATE) \
            and not ti.col("order_id", first=True) and not f("qty", "unit_price", first=True): return "debt_book"
    if f("promo_name") and (f("date_from") or f("date_to")): return "promotions"
    if f("commission_pct") or (f("staff_name") and (f("start_date") or f("commission_pct"))): return "staff_list"
    if f("lot_id") and f("expiry") and item and not f("receipt_date", first=True) and ti.uniq(ti.col("lot_id")) > 0.95 \
            and not (f("unit_cost", first=True) and f("qty", first=True) and not f("qty_in_base")): return "lots"
    if oid and f("lot_id") and (f("lot_qty_base", "qty", "qty_base", "returned_base")) and not f("item_name", first=True): return "lot_sales"
    if oid and f("delivery_date") and (f("trip_no") or f("driver") or f("plate")): return "deliveries"
    if oid and (f("due_date") or f("term_days")) and not item: return "terms"
    if item and (f("stock_after", first=True) or (f("stock_after") and f("event_time", "order_time", "order_date", dtype=DATE))): return "ledger"
    if item and f("price_date") and f("new_price", "old_price", "price", "unit_price"): return "prices"
    if f("item_code", "item_name") and (f("active_ingredient") or f("prescription")) and not f("item_group", "base_unit", "price", "cost", first=True) \
            and not oid: return "drug_info"
    purch = sum(map(bool, (f("receipt_id", first=True), f("receipt_date", first=True), f("supplier", first=True), f("receipt_total"),
                          f("receipt_status"), f("qty_in_base"), f("unit_cost", first=True) and not oid)))
    sales = sum(map(bool, (ti.col("order_id", first=True), f("order_date", first=True, dtype=DATE), f("status", first=True),
                          f("customer_name", "customer_id", first=True), f("line_amount", first=True), f("order_time", first=True))))
    if purch >= 2 and purch >= sales and (f("receipt_id") or f("receipt_date")):
        return "receipt_lines" if item else "receipt_headers"
    if item and (f("qty", "qty_base", "line_amount", "unit_price") and (oid or f("order_date", dtype=DATE))):
        if f("factor", dtype=NUM) and f("big_unit", "unit") and n <= 5000 and not oid: return "conversions"
        return "sales_lines"
    if oid and not item and f("amount", first=True) and (f("pay_time", "payment_id") or f("order_date", "order_time", dtype=DATE)) \
            and not f("total_payable", "status", "customer_name", "customer_id", "gross", first=True): return "payments"
    if oid and not item and (f("order_date", "order_time", dtype=DATE)) and (f("total_payable", "gross", "paid_total", "debt_end", "line_amount", "status")):
        return "orders"
    if item and f("factor", dtype=NUM) and (f("big_unit") or f("unit") or f("base_unit")): return "conversions"
    if item and f("stock_qty") and not f("item_group", "price", "cost", "unit_price", first=True): return "stock"
    cid = ti.col("customer_id", first=True)
    if cid and ti.uniq(cid) > 0.85 and not f("item_code", first=True) and (f("phone") or f("customer_group") or f("address") or f("member_since")):
        return "customers"
    if f("return_id") and oid: return "returns"
    tn = fold(ti.name).replace(" ", "")
    if not oid and not item and ("chiphi" in tn or "expense" in tn) and (f("amount", "line_amount", dtype=NUM) or any(p["dtype"] == "number" for p in ti.prof.values())):
        return "expenses"
    ic = ti.col("item_code", first=True) or ti.col("item_name", first=True) or ti.col(*ITEM)
    if ic and ti.uniq(ic) > 0.9 and n <= 20000 and f("item_group", "base_unit", "unit", "price", "unit_price", "cost", "stock_qty", "size", "brand",
                                                  "alias_names", "conversion_text"):
        return "catalogue"
    cc = ti.col("customer_id", first=True) or ti.col("customer_name", first=True)
    if cc and ti.uniq(cc) > 0.85 and (f("phone") or f("customer_group") or f("address") or f("member_since") or f("points")): return "customers"
    if ti.col("supplier") and ti.uniq(ti.col("supplier")) > 0.9 and n <= 500: return "suppliers"
    return None


# ------------------------------------------------------------------ 4. assembly helpers
def _num(s): return pd.to_numeric(s.map(lambda v: ingest.parse_num(v) if isinstance(v, str) else v), errors="coerce")


def _is_const_within(df, key, col):
    """share of groups (of key) on which col takes one value (blank cells ignored)."""
    g = df[[key, col]].dropna().groupby(key)[col].nunique()
    return float((g <= 1).mean()) if len(g) else 1.0


def _amount_identity(df, qty, price, amt, disc=None):
    """share of rows with amount == qty * price (- disc)."""
    x = pd.DataFrame({"q": _num(df[qty]), "p": _num(df[price]), "a": _num(df[amt])})
    if disc is not None: x["d"] = _num(df[disc]).fillna(0)
    x = x.dropna(subset=["q", "p", "a"]); x = x[(x["q"] > 0) & (x["p"] > 0)]
    if not len(x): return 0.0
    e = x["q"] * x["p"] - (x["d"] if disc is not None else 0)
    return float(((x["a"] - e).abs() <= 1 + 0.001 * x["a"].abs()).mean())


def _datetime_with_time(df, c):
    t = pd.to_datetime(df[c], errors="coerce").dropna().head(5000)
    return len(t) > 0 and float(((t.dt.hour != 0) | (t.dt.minute != 0)).mean()) > 0.3


PHONE_IN_NAME = re.compile(r"\s*[-–(,]?\s*\(?\+?(?:84|0)\d(?:[\s.\-]?\d){7,9}\)?\s*$")


def strip_phone(name):
    """'Triệu Linh An - 0327 988 481' / 'Tăng Hùng Hà (0986246709)' -> 'Triệu Linh An' / 'Tăng Hùng Hà' (None stays None)."""
    if name is None or (isinstance(name, float) and math.isnan(name)): return name
    s = PHONE_IN_NAME.sub("", str(name)).strip(" -–,")
    return s or str(name)


def phone_of(name):
    m = PHONE_IN_NAME.search(str(name)) if isinstance(name, str) else None
    return re.sub(r"\D", "", m.group(0))[-9:] if m else None


def canon_customers(names):
    """one display name per customer of a book that has no customer codes (rulebook §3: names that differ only in
    spelling are one customer, names that differ in letters or digits are two). A phone written after the name is
    part of the spelling ('An 0327988481' = 'An - 0327 988 481'); a name written with and without its phone is one
    customer when that name never appears with another phone. Display = the name without the phone."""
    v = names.dropna().astype(str)
    if not len(v): return names
    base = v.map(lambda x: dm.key(strip_phone(x))); ph = v.map(phone_of)
    phones = pd.DataFrame({"b": base, "p": ph}).dropna().groupby("b")["p"].nunique()
    disp = {}
    for b, grp in pd.DataFrame({"b": base, "v": v, "p": ph}).groupby("b"):
        if phones.get(b, 0) <= 1:                                     # one person: every spelling -> the bare name
            bare = grp.loc[grp["p"].isna(), "v"]
            nm = bare.value_counts().index[0] if len(bare) else strip_phone(grp["v"].value_counts().index[0])
            for x in grp["v"].unique(): disp[x] = nm
        else:                                                          # several phones: one customer per phone
            for p, g2 in grp.groupby(grp["p"].fillna("")):
                nm = g2["v"].value_counts().index[0]
                for x in g2["v"].unique(): disp[x] = nm
    return names.map(lambda x: disp.get(str(x), x) if isinstance(x, str) else x)


# ------------------------------------------------------------------ 5. assembly
def _frame(ti, fields):
    """frame() over the raw table with {canonical: column or None}."""
    return dm.frame(ti.df, **{k: v for k, v in fields.items() if v is not None})


def _bool(x):
    return str(x).strip().lower() in ("true", "1", "1.0", "có", "co", "x", "yes", "đúng", "dung", "kê đơn", "ke don")


CONV_TEXT = re.compile(r"1\s*([^\d=;,]+?)\s*=\s*([\d.,]+)\s*([^\d;,]+)")


def readme_unit_rules(rd):
    """conversions written in the README conventions in sentence form: '<A>, <B> bán theo bó, 1 bó = 10 cây' ->
    rows (item, unit, factor, base_unit) for every item listed."""
    rows = []
    for conv in rd["conventions"]:
        for m in re.finditer(r"(?:\*\*[^*]+\*\*\s*:\s*)?([^:\n]+?)\s+bán theo\s+(\w+)\s*,\s*1\s+(\w+)\s*=\s*([\d.,]+)\s+(\w+)", conv):
            names, unit, f, base = m.group(1), m.group(3), float(m.group(4).replace(",", ".")), m.group(5)
            for nm in re.split(r",\s*|\s+và\s+", names):
                nm = nm.strip(" .")
                if nm: rows.append((nm, unit, f, base))
    return pd.DataFrame(rows, columns=["item", "unit", "factor", "base_unit"])


def build_items(K, notes):
    """catalogue: item_code / item_name / item_group / base_unit / price / cost + attributes, drug info merged by code."""
    cats = K.get("catalogue", [])
    if not cats: return None
    ti = max(cats, key=lambda t: len(t.df))
    f = {"item_code": ti.col("item_code", first=True) or ti.col("item_code"), "item_name": ti.col("item_name", first=True) or ti.col("item_name"),
         "item_group": ti.col("item_group", first=True) or ti.col("item_group"),
         "base_unit": ti.col("base_unit", first=True) or ti.col("unit", "base_unit"),
         "price": ti.col("price", first=True) or ti.col("price", "unit_price", dtype=NUM),
         "cost": ti.col("cost", dtype=NUM) or ti.col("unit_cost", dtype=NUM),
         "size": ti.col("size"), "brand": ti.col("brand", first=True), "alias_names": ti.col("alias_names"),
         "conversion_text": ti.col("conversion_text"), "stock_qty": ti.col("stock_qty", dtype=NUM), "active": ti.col("active")}
    if f["item_code"] == f["item_name"]: f["item_name"] = None
    it = _frame(ti, f)
    if "item_code" in it: it["item_code"] = it["item_code"].astype(str).str.strip()
    it["unit_cost"] = it["cost"] if "cost" in it else np.nan
    for d in K.get("drug_info", []):
        k = d.col("item_code") or d.col("item_name"); kk = "item_code" if d.col("item_code") else "item_name"
        if kk not in it: continue
        key_ = it[kk].astype(str).str.strip()
        ai, pr = d.col("active_ingredient"), d.col("prescription")
        if ai: it["active_ingredient"] = key_.map(dict(zip(d.df[k].astype(str).str.strip(), d.df[ai])))
        if pr: it["prescription"] = key_.map(dict(zip(d.df[k].astype(str).str.strip(), d.df[pr].map(_bool))))
    for c in ("size", "brand"):
        if c in it: it[c] = it[c].where(it[c].astype(str).str.strip().ne(""), None)
    notes.append(f"catalogue from {ti.name} ({len(it)} items)")
    return it


def build_conversions(K, rd, items, biz):
    """unit conversions (item, unit, factor = base units per unit, base_unit): conversion tables (chained factors
    resolved), README bullets ('- X: 1 thùng = 24 lon'), README sentences ('X bán theo bó, 1 bó = 10 cây'),
    conversion text in the catalogue ('1 thùng = 4 bịch')."""
    parts = []
    for ti in K.get("conversions", []):
        df = ti.df
        ic = ti.col("item_code", first=True) or ti.col("item_name", first=True) or ti.col(*ITEM)
        uc = ti.col("big_unit", first=True) or ti.col("big_unit") or ti.col("unit")
        fc = ti.col("factor", dtype=NUM); bc = ti.col("base_unit", first=True)
        if not (ic and uc and fc): continue
        x = pd.DataFrame({"item": df[ic].astype(str).str.strip(), "unit": df[uc].astype(str).str.strip(), "factor": pd.to_numeric(df[fc], errors="coerce"),
                          "via": df[fc + "__unit"] if fc + "__unit" in df.columns else None, "base_unit": df[bc] if bc else None}).dropna(subset=["factor"])
        # chained factors: 'hộp' = 15 'vỉ' (the factor's own unit) -> 15 x factor(vỉ) base units
        for _ in range(3):
            fmap = {(dm.key(i), dm.key(u)): f for i, u, f in zip(x["item"], x["unit"], x["factor"])}
            def resolve(r):
                v = r["via"]
                if v is None or (isinstance(v, float) and math.isnan(v)) or not str(v).strip(): return r["factor"]
                if r["base_unit"] is not None and isinstance(r["base_unit"], str) and dm.key(v) == dm.key(r["base_unit"]): return r["factor"]
                inner = fmap.get((dm.key(r["item"]), dm.key(v)))
                return r["factor"] * inner if inner else r["factor"]
            new = x.apply(resolve, axis=1)
            done = (new == x["factor"]).all(); x["factor"] = new; x["via"] = None
            if done: break
        parts.append(x[["item", "unit", "factor", "base_unit"]])
    rc = dm.readme_conversions(biz.readme or "")
    if len(rc): parts.append(rc)
    ru = readme_unit_rules(rd)
    if len(ru): parts.append(ru)
    if items is not None and "conversion_text" in items:
        rows = []
        for nm, code, txt in zip(items.get("item_name", items.get("item_code")), items.get("item_code", items.get("item_name")), items["conversion_text"]):
            for m in CONV_TEXT.finditer(str(txt or "")):
                rows.append((nm, m.group(1).strip(), float(m.group(2).replace(",", ".")), m.group(3).strip()))
        if rows: parts.append(pd.DataFrame(rows, columns=["item", "unit", "factor", "base_unit"]))
    if not parts: return None
    cv = pd.concat(parts, ignore_index=True)
    cv = cv[cv["factor"].notna() & (cv["factor"] > 0)].drop_duplicates(["item", "unit"], keep="first").reset_index(drop=True)
    return cv if len(cv) else None


PSEUDO = re.compile(r"^(voucher|dung \d+ diem|phi ship|phi giao hang|giam gia hoa don|chiet khau hoa don|khuyen mai hoa don)\b")


def _status_from_note(note):
    """rulebook §8: a book without a status column writes the status in its notes; a note that names none = completed."""
    return note.map(lambda x: (dm.canon_status(x, None) or "hoàn thành") if isinstance(x, str) and x.strip() else None)


def _abbrev_table(biz, infos, names):
    """a lookup table of short names (a small table of two text columns: short -> full name, optional type): the
    short column covers most item names of the sales book, the other one is longer. -> (short col, full col, table)."""
    ks = set(names.dropna().astype(str).map(dm.key))
    best = None
    for ti in infos.values():
        df = ti.df
        if not 3 <= len(df) <= 3000 or ti.kind not in (None, "catalogue"): continue
        txt = [c for c in df.columns if not str(c).startswith("_") and ti.prof.get(c, {}).get("dtype") == "text" and df[c].notna().mean() > 0.8]
        for a in txt:
            ka = set(df[a].dropna().astype(str).map(dm.key))
            share = float(names.dropna().astype(str).map(dm.key).isin(ka).mean()) if len(names.dropna()) else 0.0
            if share < 0.5: continue
            for b in txt:
                if b == a: continue
                longer = float((df[b].astype(str).str.len() > df[a].astype(str).str.len() + 2).mean())
                kb = set(df[b].dropna().astype(str).map(dm.key))
                share_b = float(names.dropna().astype(str).map(dm.key).isin(kb).mean()) if len(names.dropna()) else 0.0
                if longer > 0.6 and len(kb) >= 0.4 * len(ka) and share_b < share / 2:
                    if best is None or share > best[0]: best = (share, a, b, ti)
    return best


TOPPING = re.compile(r"\s+\+\s+(?:(\d+)\s*(?:x\s*)?)?(.+?)\s*\(\s*([\d.,]+)\s*(?:k|đ|d|vnđ|vnd)?\s*\)\s*$", re.I)


def split_toppings(L, rd):
    """README: 'topping ghi chung ô với món, nối bằng " + " ... tiền topping ghi trong ngoặc và đã cộng vào thành tiền'
    -> one line for the dish (amount minus the toppings) and one line per topping (qty = its number of portions, default
    1; amount = the price in brackets). Public rulebook: the topping is its own item."""
    if not any(re.search(r"topping", c, re.I) and "+" in c for c in rd["conventions"]): return L
    if "item_name" not in L: return L
    rows, out = [], []
    for idx, r in L.iterrows():
        nm = r["item_name"]
        if not isinstance(nm, str) or " + " not in nm: out.append(r); continue
        parts = nm.split(" + "); dish = parts[0].strip(); tops = []
        for p in parts[1:]:
            m = re.match(r"^\s*(?:(\d+)\s*(?:x\s*)?)?(.+?)\s*\(\s*([\d.,]+)\s*(k|đ|d|vnđ|vnd)?\s*\)\s*$", p, re.I)
            if not m: tops = None; break
            amt = ingest.parse_num(m.group(3) + (m.group(4) or "đ"))
            tops.append((int(m.group(1) or 1), m.group(2).strip(), amt))
        if not tops: out.append(r); continue
        d = r.copy(); d["item_name"] = re.sub(r"\s{2,}", " ", dish)
        tot_t = sum(t[2] for t in tops)
        if pd.notna(d.get("line_amount")): d["line_amount"] = d["line_amount"] - tot_t
        if "qty_base" in d and pd.notna(d.get("qty_base")) and pd.notna(d.get("qty")): pass
        out.append(d)
        for q, tn, amt in tops:
            t = r.copy(); t["item_name"] = tn; t["qty"] = q; t["unit_price"] = amt / q if q else amt; t["line_amount"] = amt
            for c in ("item_code", "line_discount", "unit_cost"):
                if c in t: t[c] = np.nan if c != "item_code" else None
            if "qty_base" in t: t["qty_base"] = q
            if "unit" in t: t["unit"] = "phần"
            if "factor" in t: t["factor"] = 1.0
            out.append(t)
    return pd.DataFrame(out).reset_index(drop=True)


LINE_FIELDS = ("order_id", "line_no", "order_date", "order_time", "item_code", "item_name", "item_group", "unit", "qty", "qty_base",
               "unit_price", "line_discount", "line_amount", "unit_cost", "status", "note", "customer_id", "customer_name",
               "customer_group", "staff", "channel", "payment_method", "order_discount", "shipping_fee", "vat", "total_payable",
               "paid_total", "debt_end", "gross", "tradein", "phone", "address", "table_no", "area", "guests", "imei")
ORDER_FIELDS = ("order_id", "order_date", "order_time", "customer_id", "customer_name", "customer_group", "staff", "channel",
                "payment_method", "status", "order_discount", "shipping_fee", "vat", "total_payable", "paid_total", "debt_end", "gross",
                "note", "tradein", "phone", "address", "table_no", "area", "guests")


def pick_fields(ti, wanted, line_table):
    """canonical field -> raw column of one table (each raw column used once)."""
    used, f = set(), {}
    def take(name, *roles, dtype=None, first_only=False):
        c = None
        for first in ((True,) if first_only else (True, False)):
            c = next((x for x in ti.cols(*roles, first=first, dtype=dtype) if x not in used), None)
            if c: break
        if c: used.add(c); f[name] = c
        return c
    # ids and dates first (they disambiguate the rest)
    take("order_id", "order_id")
    if "line_no" in wanted: take("line_no", "line_no", dtype=NUM, first_only=True)
    dcols = [c for c in ti.cols("order_date", "order_time", dtype=DATE) if c not in used]
    tcol = next((c for c in ti.cols("order_time", dtype=DATE, first=True) if c not in used and _datetime_with_time(ti.df, c)), None) or \
        next((c for c in dcols if _datetime_with_time(ti.df, c)), None)
    dcol = next((c for c in ti.cols("order_date", dtype=DATE, first=True) if c != tcol), None) or next((c for c in dcols if c != tcol), None)
    if dcol: f["order_date"] = dcol; used.add(dcol)
    if tcol: f["order_time"] = tcol; used.add(tcol)
    if line_table:
        take("item_code", "item_code", first_only=True); take("item_name", "item_name", first_only=True)
        if "item_name" not in f: take("item_name", "item_name")
        if "item_code" not in f: take("item_code", "item_code")
        take("item_group", "item_group"); take("qty_base", "qty_base", dtype=NUM, first_only=True); take("qty", "qty", dtype=NUM)
        if "qty" not in f and "qty_base" not in f: take("qty", "qty_base", dtype=NUM)
        take("unit", "unit", first_only=True)
        if "unit" not in f: take("unit", "unit", "base_unit")
        take("unit_price", "unit_price", dtype=NUM, first_only=True)
        if "unit_price" not in f: take("unit_price", "unit_price", "price", dtype=NUM)
        take("line_amount", "line_amount", dtype=NUM, first_only=True)
        take("unit_cost", "unit_cost", "cost", dtype=NUM)
    take("status", "status", first_only=True)
    if "status" not in f:
        c = next((x for x in ti.cols("status") if x not in used and ti.prof[x].get("status", 0) > 0.5), None)
        if c: used.add(c); f["status"] = c
    take("customer_id", "customer_id", first_only=True); take("customer_name", "customer_name", first_only=True)
    take("customer_group", "customer_group", first_only=True); take("staff", "staff", first_only=True)
    take("channel", "channel", first_only=True); take("payment_method", "payment_method", first_only=True)
    if "payment_method" not in f:
        c = next((x for x in ti.cols("method", "payment_method") if x not in used and ti.prof[x].get("pay", 0) > 0.5), None)
        if c: used.add(c); f["payment_method"] = c
    for k in ("order_discount", "line_discount", "shipping_fee", "vat", "total_payable", "paid_total", "debt_end", "gross", "tradein"):
        take(k, k, dtype=NUM, first_only=True)
    # ambiguous 'discount' columns: a line discount when amount = qty x price - discount, else an order discount
    for c in [x for x in ti.cols("discount", dtype=NUM) if x not in used]:
        k = "order_discount"
        if line_table and f.get("qty") and f.get("unit_price") and f.get("line_amount"):
            if _amount_identity(ti.df, f["qty"], f["unit_price"], f["line_amount"], c) > max(0.9, _amount_identity(ti.df, f["qty"], f["unit_price"], f["line_amount"]) + 0.02):
                k = "line_discount"
        if k in f: continue
        f[k] = c; used.add(c)
    if not line_table and "total_payable" not in f:
        take("total_payable", "total_payable", "line_amount", dtype=NUM)
    for k, roles in (("note", ("note",)), ("phone", ("phone",)), ("address", ("address",)), ("table_no", ("table_no",)), ("area", ("area",)),
                     ("guests", ("guests",)), ("imei", ("imei",))):
        take(k, *roles)
    return {k: v for k, v in f.items() if k in wanted}


def build_sales(biz, infos, K, rd, items, cv, notes):
    S = max(K.get("sales_lines", []), key=lambda t: len(t.df), default=None)
    H = None
    if K.get("orders"):
        if S is not None and pick_fields(S, LINE_FIELDS, True).get("order_id"):
            sid = set(S.df[pick_fields(S, LINE_FIELDS, True)["order_id"]].dropna().astype(str).str.strip())
            H = max(K["orders"], key=lambda t: (len(set(t.df[pick_fields(t, ORDER_FIELDS, False).get("order_id")].dropna().astype(str).str.strip()) & sid)
                                                if pick_fields(t, ORDER_FIELDS, False).get("order_id") else 0, len(t.df)))
        else:
            H = max(K["orders"], key=lambda t: len(t.df))
    if S is None and H is None and K.get("repair_tickets"):
        return build_tickets(biz, K, notes)
    if S is None and H is None: raise ValueError("no sales table")
    hf = pick_fields(H, ORDER_FIELDS, False) if H is not None else {}
    if S is None:                                                     # invoice list only: revenue = tiền hàng of the order
        rev = H.col("gross", dtype=NUM, first=True) or H.col("gross", dtype=NUM) or H.col("line_amount", dtype=NUM)
        orders = _frame(H, hf)
        if "status" in hf: orders["status"] = dm.map_status(H.df[hf["status"]])
        lines = dm.frame(H.df, order_id=hf["order_id"], line_amount=rev) if rev else dm.frame(H.df, order_id=hf["order_id"], line_amount=hf.get("total_payable"))
        notes.append(f"invoice list only ({H.name}): no item lines; order revenue = {rev or hf.get('total_payable')}")
        if rev: orders["revenue"] = _num(H.df[rev])
        O, L = dm.finalize(lines, orders.drop(columns=[c for c in ("gross",) if c in orders]))
        return O, L, {"lines": None, "orders": H.name, "fields": hf}
    lf = pick_fields(S, LINE_FIELDS, True)
    df = S.df; LOOKUPS = []
    L = _frame(S, lf)
    # status: a status column, else the notes (rulebook §8)
    if "status" in lf: L["status"] = dm.map_status(df[lf["status"]])
    elif "note" in lf and "status" not in hf:
        L["status"] = _status_from_note(df[lf["note"]]); notes.append(f"status read from the notes column {lf['note']}")
    # pseudo-lines of an order (voucher, points used, shipping fee written as item lines): order-level amounts
    if "item_name" in L:
        k = L["item_name"].map(lambda x: fold(x) if isinstance(x, str) else "")
        ps = k.str.match(PSEUDO) & (L["qty"].isna() if "qty" in L else True)
        if ps.sum() > 0:
            amt = L.loc[ps, "line_amount"] if "line_amount" in L else pd.Series(0, index=L.index[ps])
            kind = k[ps].map(lambda x: "shipping_fee" if x.startswith(("phi ship", "phi giao")) else "points_discount" if x.startswith("dung") else "voucher_discount")
            oid = L.loc[ps, "order_id"].astype(str).str.strip() if "order_id" in L else None
            P = pd.DataFrame({"order_id": oid, "kind": kind, "amount": _num(amt).abs(), "name": L.loc[ps, "item_name"]})
            if "order_id" in L:
                ids = L["order_id"].where(L["order_id"].astype(str).str.strip().isin(["", "nan", "None"]) == False).ffill()
                P["order_id"] = ids[ps].astype(str).str.strip().values
            P = P.drop_duplicates(["order_id", "kind", "amount", "name"])     # rulebook §8: identical rows of one order = export error
            piv = P.pivot_table(index="order_id", columns="kind", values="amount", aggfunc="sum").fillna(0)
            pts = k[ps & k.str.startswith("dung")].str.extract(r"dung (\d+) diem")[0].astype(float)
            L = L[~ps].reset_index(drop=True)
            notes.append(f"{int(ps.sum())} pseudo-lines (voucher / points / shipping) moved to order-level amounts")
            PSEUDO_ORDERS = piv
        else: PSEUDO_ORDERS = None
    else: PSEUDO_ORDERS = None
    L = split_toppings(L, rd)
    # abbreviated item names + lookup sheet (short -> full name, type)
    ab = _abbrev_table(biz, infos, L["item_name"]) if "item_name" in L else None
    if ab:
        _, a, b, ti = ab; m = dict(zip(ti.df[a].astype(str).map(dm.key), ti.df[b])); L["item_name_raw"] = L["item_name"]
        L["item_name"] = L["item_name"].map(lambda x: m.get(dm.key(x), x) if isinstance(x, str) else x)
        other = [c for c in ti.df.columns if c not in (a, b) and not str(c).startswith("_") and ti.prof.get(c, {}).get("dtype") == "text"]
        for c in other[:1]:
            mm = dict(zip(ti.df[b].astype(str).map(dm.key), ti.df[c]))
            L["item_type"] = L["item_name"].map(lambda x: mm.get(dm.key(x)) if isinstance(x, str) else None)
            if "item_group" not in L: L["item_group"] = L["item_type"]
        notes.append(f"item names expanded through the lookup {ti.name}.{a} -> {b}"); LOOKUPS.append(ti.name)
    orders = None
    if H is not None:
        orders = _frame(H, hf)
        if "status" in hf: orders["status"] = dm.map_status(H.df[hf["status"]])
        elif "note" in hf and "status" not in lf: orders["status"] = _status_from_note(H.df[hf["note"]])
        if "gross" in orders: orders = orders.drop(columns=["gross"])
    if PSEUDO_ORDERS is not None and len(PSEUDO_ORDERS):
        if orders is None:
            ids = L["order_id"].astype(str).str.strip()
            for c in PSEUDO_ORDERS.columns: L[c] = ids.map(PSEUDO_ORDERS[c])
        else:
            for c in PSEUDO_ORDERS.columns: orders[c] = orders["order_id"].astype(str).str.strip().map(PSEUDO_ORDERS[c])
    if "gross" in L: L = L.drop(columns=["gross"])
    O, L = dm.finalize(L, orders, conversions=cv, items=items)
    for c in ("paid_total", "debt_end", "tradein", "phone", "address", "table_no", "area", "guests", "note", "imei_count"):
        if c in L.columns and c not in O.columns and "order_id" in L.columns:   # order-level cells of a flat book (finalize keeps
            first = L.dropna(subset=[c]).groupby(L["order_id"].astype(str))[c].first()   # only ORDER_COLS on the orders)
            O[c] = O["order_id"].astype(str).map(first)
    if PSEUDO_ORDERS is not None and len(PSEUDO_ORDERS):
        for c in PSEUDO_ORDERS.columns:
            if c not in O: O[c] = O["order_id"].map(PSEUDO_ORDERS[c])
            O[c] = O[c].fillna(0.0)
        if "voucher_discount" in O and "order_discount" not in O: O["order_discount"] = O["voucher_discount"]   # voucher money off
        if "points_discount" not in O: O["points_discount"] = 0.0                                                # (industry.voucher_total)
        if "shipping_fee" not in O and "shipping_fee" in PSEUDO_ORDERS: O["shipping_fee"] = O["order_id"].map(PSEUDO_ORDERS["shipping_fee"]).fillna(0)
    return O, L, {"lines": S.name, "orders": H.name if H is not None else None, "fields": lf, "order_fields": hf, "lookups": LOOKUPS}


PLATE = re.compile(r"\b(\d{2}\s?-?\s?[A-Z]{1,2}\d?\s?-?\s?\d{3}\.?\d{2}|\d{2}[A-Z]{1,2}\d?-?\d{4,5})\b")


def split_vehicle(v):
    """'wave alpha 50K2-994.07' -> ('wave alpha', '50K2-994.07')."""
    if not isinstance(v, str): return (None, None)
    m = PLATE.search(v.upper())
    if not m: return (v.strip(), None)
    return (v[:m.start()].strip(" -,") or None, m.group(1).replace(" ", ""))


def link_people(values, full_names):
    """nickname / given name written in a book ('A.Phúc', 'a Phúc', 'Phúc') -> the one full name of a staff list whose last
    words match (two candidates: unchanged)."""
    fk = [(n, dm.key(n).split()) for n in full_names if isinstance(n, str) and n.strip()]
    def one(x):
        if not isinstance(x, str): return x
        g = dm.person_key(re.sub(r"[.]", " ", x)).split()
        if not g: return x
        hit = [n for n, k in fk if k[-len(g):] == g]
        return hit[0] if len(hit) == 1 else x
    return values.map(one)


def build_tickets(biz, K, notes):
    """a service book (repair shop): every row is one ticket (rulebook: one row = one visit), ticket money = revenue;
    vehicle cell -> model + plate; warranty tickets (note 'bảo hành', money 0) kept as tickets with revenue 0."""
    ti = max(K["repair_tickets"], key=lambda t: len(t.df)); df = ti.df.reset_index(drop=True)
    g = ti.col
    dc = g("order_date", dtype=DATE); vc = g("vehicle"); tc = g("tasks"); sc = g("staff"); ac = g("line_amount", "total_payable", "amount", dtype=NUM)
    nc = g("note")
    if ac is None: ac = next((c for c in df.columns if ti.prof.get(c, {}).get("dtype") == "number"), None)
    oid = "T" + pd.Series(range(len(df))).astype(str)
    import industry
    mp = df[vc].map(industry.split_vehicle) if vc else pd.Series([(None, None)] * len(df))
    staff = df[sc] if sc else None
    for st in K.get("staff_list", []):
        fn = st.col("staff_name") or st.col("staff")
        if fn and staff is not None: staff = link_people(staff, st.df[fn].dropna().astype(str).unique()); notes.append(f"mechanics linked to the full names of {st.name}")
    amt = _num(df[ac]).fillna(0) if ac else pd.Series(0.0, index=df.index)
    note = df[nc] if nc else pd.Series([None] * len(df))
    warranty = note.map(lambda x: isinstance(x, str) and "bao hanh" in dm.key(x)) | ((amt == 0) & note.notna())
    lines = pd.DataFrame({"order_id": oid, "order_date": df[dc] if dc else pd.NaT, "item_name": "phiếu sửa xe", "qty": 1.0,
                          "line_amount": amt, "staff": staff, "note": note})
    O, L = dm.finalize(lines, dedupe=False)
    O["vehicle_cell"] = df[vc].values if vc else None
    O["vehicle"] = [m for m, _ in mp]; O["plate"] = [p for _, p in mp]
    O["work"] = df[tc].values if tc else None; O["warranty"] = warranty.values
    O["staff_raw"] = df[sc].values if sc else None
    notes.append(f"service book {ti.name}: one ticket per row ({len(O)} tickets, {int(warranty.sum())} warranty)")
    return O, L, {"lines": ti.name, "orders": None, "fields": {"order_date": dc, "vehicle": vc, "tasks": tc, "staff": sc, "amount": ac, "note": nc}}


def build_member_list(K):
    """C['customers']: the customer / member list (customer_id, customer_name, customer_group, phone, registered, points)."""
    for ti in K.get("customers", []):
        g = ti.col
        c = dm.frame(ti.df, customer_id=g("customer_id", first=True), customer_name=g("customer_name", first=True) or g("customer_name") or g("item_name"),
                     customer_group=g("customer_group", first=True), phone=g("phone"), registered=g("member_since", dtype=DATE), points=g("points", dtype=NUM))
        if "customer_id" in c: c["customer_id"] = c["customer_id"].astype(str).str.strip()
        return c
    return None


def debts_from_notes(O, rd, notes):
    """orders whose debt is written only in a free-text note (README: 'đơn ghi nợ thì ghi khách đã đưa / còn thiếu bao nhiêu',
    'ghi chú nợ ở Note là tình trạng của đơn tính đến hết ngày cuối dữ liệu'): debt_end = debt_from_note(note, total)."""
    if "note" not in O or "debt_end" in O or "paid_total" in O: return O
    said = any(re.search(r"\b(no|ghi no|con thieu|da dua)\b", fold(c)) and re.search(r"ghi chu|note", fold(c)) for c in rd["conventions"]) or \
        any(re.search(r"ghi no thi ghi|con thieu|ghi chu no", fold(d)) for sec in rd["sections"] for d in sec["cols"].values())
    if not said: return O
    tot = O["total_payable"] if "total_payable" in O else pd.Series(np.nan, index=O.index)
    de = [dm.debt_from_note(n, t) for n, t in zip(O["note"], tot)]
    if sum(1 for x in de if x is not None) < 5: return O
    O = O.copy(); O["debt_end"] = [x if x is not None else 0.0 for x in de]
    notes.append(f"debts read from the order notes ({sum(1 for x in de if x)} orders still owing at the end of the data)")
    return O


def build_customers(K, O):
    """customer names / groups from a customer list (by code); names that carry a phone made one per customer."""
    for ti in K.get("customers", []):
        ic = ti.col("customer_id", first=True) or ti.col("customer_id")
        nc = ti.col("customer_name", first=True) or ti.col("customer_name") or ti.col("item_name")
        gc = ti.col("customer_group", first=True) or ti.col("customer_group")
        if ic and "customer_id" in O and O["customer_id"].notna().any():
            ids = ti.df[ic].astype(str).str.strip(); oc = O["customer_id"].astype(str).str.strip()
            if oc[O["customer_id"].notna()].isin(set(ids)).mean() < 0.5: continue
            if nc and ("customer_name" not in O or O["customer_name"].isna().all()):
                O["customer_name"] = oc.map(dict(zip(ids, ti.df[nc]))).where(O["customer_id"].notna(), None)
            if gc and ("customer_group" not in O or O["customer_group"].isna().all()):
                O["customer_group"] = oc.map(dict(zip(ids, ti.df[gc]))).where(O["customer_id"].notna(), None)
            break
    if "customer_name" in O and O["customer_name"].notna().any() and ("customer_id" not in O or O["customer_id"].isna().all()):
        O["customer_name_raw"] = O["customer_name"]; O["customer_name"] = canon_customers(O["customer_name"])
    elif "customer_name" in O:
        O["customer_name"] = O["customer_name"].map(lambda x: strip_phone(x) if isinstance(x, str) and phone_of(x) else x)
    return O


def build_purchases(K, cv, items):
    L = K.get("receipt_lines", []); Hs = K.get("receipt_headers", [])
    if not L: return None
    ti = max(L, key=lambda t: len(t.df)); df = ti.df.copy()
    rid = ti.col("receipt_id", first=True) or ti.col("receipt_id")
    hdr = None
    if Hs and rid:
        h = max(Hs, key=lambda t: len(set(t.df[t.col("receipt_id")].astype(str)) & set(df[rid].astype(str))) if t.col("receipt_id") else 0)
        hid = h.col("receipt_id")
        if hid:
            keep = [c for c in h.df.columns if c != hid and not str(c).startswith("_") and c not in df.columns]
            df = df.merge(h.df[[hid] + keep].rename(columns={hid: rid}), on=rid, how="left"); hdr = h
    roles = dict(ti.roles); prof = dict(ti.prof)
    if hdr is not None:
        for c in hdr.roles:
            if c in df.columns and c not in roles: roles[c] = hdr.roles[c]; prof[c] = hdr.prof[c]
    tmp = TableInfo.__new__(TableInfo); tmp.name, tmp.df, tmp.descs = ti.name, df, {**(hdr.descs if hdr else {}), **ti.descs}; tmp.roles, tmp.prof = roles, prof
    g = lambda *r, **k: tmp.col(*r, **k)
    dcol = g("receipt_date", dtype=DATE, first=True) or g("receipt_date", "order_date", dtype=DATE)
    f = {"receipt_id": rid, "receipt_date": dcol, "supplier": g("supplier", first=True) or g("supplier"),
         "item_code": g("item_code", first=True), "item_name": g("item_name", first=True), "unit": g("unit", first=True) or g("unit", "base_unit"),
         "qty": g("qty", dtype=NUM, first=True) or g("qty", "qty_in_base", dtype=NUM), "unit_cost": g("unit_cost", dtype=NUM, first=True) or g("unit_cost", "unit_price", "cost", dtype=NUM),
         "amount": g("amount", dtype=NUM, first=True) or g("amount", "line_amount", dtype=NUM), "note": g("note"), "lot_id": g("lot_id"), "expiry": g("expiry", dtype=DATE),
         "qty_in_base": g("qty_in_base", dtype=NUM, first=True)}
    if f["item_code"] is None and f["item_name"] is None: f["item_name"] = g("item_name", "item_code")
    P = dm.frame(df, **{k: v for k, v in f.items() if v})
    if "receipt_date" in P: P["receipt_date"] = pd.to_datetime(P["receipt_date"], errors="coerce").dt.normalize()
    if "amount" not in P and {"qty", "unit_cost"} <= set(P.columns): P["amount"] = P["qty"] * P["unit_cost"]
    st = g("receipt_status", first=True) or g("receipt_status", "status")
    P["cancelled"] = df[st].map(lambda x: dm.canon_status(x, None) == "huỷ") if st else False
    if "item_code" in P: P["item_code"] = P["item_code"].astype(str).str.strip()
    sup = K.get("suppliers", [])
    if sup and "supplier" in P:
        s = sup[0]; sc, sn = s.col("supplier"), next((c for c in s.df.columns if fold_name(c) in ("ten goi", "ten", "ten nha cung cap", "ten ncc")), None)
        if sc and sn: P["supplier_name"] = P["supplier"].astype(str).map(dict(zip(s.df[sc].astype(str), s.df[sn])))
    return P


def build_side(K, C, cv, items, notes):
    """payments, stock, ledger, lots, lot sales, deliveries, credit terms, prices, expenses, returns."""
    for ti in K.get("payments", []):
        g = ti.col
        P = dm.frame(ti.df, order_id=g("order_id"), pay_time=g("pay_time", dtype=DATE) or g("order_time", "order_date", dtype=DATE),
                     amount=g("amount", dtype=NUM, first=True) or g("amount", "line_amount", dtype=NUM), method=g("payment_method", "method"))
        P["order_id"] = P["order_id"].astype(str).str.strip(); P["pay_date"] = pd.to_datetime(P["pay_time"]).dt.normalize()
        C["payments"] = P; break
    st = []
    for ti in K.get("stock", []):
        g = ti.col
        st.append(dm.frame(ti.df, item_code=g("item_code", first=True), item_name=g("item_name", first=True) or g("item_name"),
                           base_unit=g("base_unit", "unit"), qty_base=g("stock_qty", dtype=NUM) or g("qty_base", "qty", dtype=NUM)))
    if not st and items is not None and "stock_qty" in items:
        st.append(items[[c for c in ("item_code", "item_name", "base_unit") if c in items] + ["stock_qty"]].rename(columns={"stock_qty": "qty_base"}))
    if st:
        s = st[0]
        if "item_code" in s: s["item_code"] = s["item_code"].astype(str).str.strip()
        C["stock"] = s
    for ti in K.get("ledger", []):
        g = ti.col
        C["ledger"] = dm.frame(ti.df, item_code=g("item_code"), time=g("event_time", dtype=DATE) or g("order_time", "order_date", dtype=DATE),
                               kind=g("event_kind"), qty_base=g("qty_base", "qty", dtype=NUM), stock_after=g("stock_after", dtype=NUM)); break
    lots = []
    for ti in K.get("lots", []):
        g = ti.col
        x = dm.frame(ti.df, lot_id=g("lot_id"), receipt_id=g("receipt_id"), item_code=g("item_code", first=True) or g("item_code"),
                     item_name=g("item_name", first=True), expiry=g("expiry", dtype=DATE), qty_in_base=g("qty_in_base", "qty_base", "qty", dtype=NUM),
                     unit_cost_base=g("unit_cost", "cost", dtype=NUM), receipt_date=g("receipt_date", dtype=DATE))
        lots.append(x)
    P = C.get("purchases")
    if lots:
        L_ = lots[0]
        if P is not None and "lot_id" in P and "receipt_date" not in L_:
            L_["receipt_date"] = L_["lot_id"].map(dict(zip(P["lot_id"], P["receipt_date"])))
        if "unit_cost_base" not in L_ and P is not None and "lot_id" in P and "unit_cost" in P:
            f = P.get("qty_in_base", None)
            L_["unit_cost_base"] = L_["lot_id"].map(dict(zip(P["lot_id"], P["unit_cost"])))
        if "item_code" in L_: L_["item_code"] = L_["item_code"].astype(str).str.strip()
        C["lots"] = L_
    elif P is not None and "lot_id" in P and "expiry" in P and P["lot_id"].notna().mean() > 0.9:
        lt = P[["lot_id", "receipt_id", "expiry"] + [c for c in ("item_code", "item_name", "receipt_date") if c in P]].copy()
        lt["qty_in_base"] = P["qty_in_base"] if "qty_in_base" in P else P["qty"]
        lt["unit_cost_base"] = P["unit_cost"] if "unit_cost" in P else np.nan
        C["lots"] = lt
    for ti in K.get("lot_sales", []):
        g = ti.col
        C["lot_sales"] = dm.frame(ti.df, order_id=g("order_id"), line_no=g("line_no"), lot_id=g("lot_id"),
                                  qty_base=g("lot_qty_base", first=True) or g("qty_base", "qty", dtype=NUM), returned_base=g("returned_base", dtype=NUM))
        C["lot_sales"]["order_id"] = C["lot_sales"]["order_id"].astype(str).str.strip(); break
    for ti in K.get("deliveries", []):
        g = ti.col
        d = dm.frame(ti.df, order_id=g("order_id"), delivery_date=g("delivery_date", dtype=DATE), trip_no=g("trip_no"), item_name=g("item_name"),
                     item_code=g("item_code"), qty_base=g("qty_base", "qty", dtype=NUM), driver=g("driver"), base_unit=g("base_unit", "unit"))
        pc = g("plate")
        if pc: d["plate"] = ti.df[pc].astype(str)
        d["order_id"] = d["order_id"].astype(str).str.strip(); C["deliveries"] = d; break
    for ti in K.get("terms", []):
        g = ti.col
        t = dm.frame(ti.df, order_id=g("order_id"), credit_days=g("term_days", dtype=NUM), due_date=g("due_date", dtype=DATE))
        t["order_id"] = t["order_id"].astype(str).str.strip(); C["terms"] = t; break
    for ti in K.get("prices", []):
        g = ti.col
        pr = dm.frame(ti.df, date=g("price_date", dtype=DATE), item_code=g("item_code", first=True), item_name=g("item_name", first=True) or g("item_name"),
                      price=g("new_price", dtype=NUM) or g("price", "unit_price", dtype=NUM), old_price=g("old_price", dtype=NUM))
        if "item_code" in pr: pr["item_code"] = pr["item_code"].astype(str).str.strip()
        C["prices"] = pr; break
    for ti in K.get("expenses", []):
        df = ti.df
        dc = next((c for c in df.columns if ti.prof.get(c, {}).get("dtype") == "datetime"), None)
        ac = ti.col("amount", dtype=NUM) or next((c for c in df.columns if ti.prof.get(c, {}).get("dtype") == "number"), None)
        cc = next((c for c in df.columns if fold_name(c) in ("muc", "nhom", "loai", "khoan", "khoan chi", "loai chi", "hang muc", "danh muc")), None)
        nc = ti.col("note")
        if dc and ac:
            C["expenses"] = dm.frame(df, date=dc, amount=ac, category=cc, note=nc); break
    for ti in K.get("returns", []):
        g = ti.col
        r = dm.frame(ti.df, return_id=g("return_id"), order_id=g("order_id"))
        r["order_id"] = r["order_id"].astype(str).str.strip(); C["returns"] = r; break
    return C


def _order_ref(x):
    """the sales order a side-book row refers to: 'STT 26' / 'đơn 26' / 26.0 -> '26' (the sales book's order number as text)."""
    if x is None or (isinstance(x, float) and math.isnan(x)): return None
    if isinstance(x, (int, float, np.integer, np.floating)): return str(int(x))
    m = re.search(r"(\d+)", str(x)); return m.group(1) if m else None


def _norm_ids(s):
    """order codes as text, '26.0' -> '26' (numbers read from a sheet)."""
    return s.map(lambda x: str(int(x)) if isinstance(x, (float, np.floating)) and not math.isnan(x) and float(x).is_integer()
                 else (str(x).strip()[:-2] if isinstance(x, str) and re.fullmatch(r"\d+\.0", str(x).strip()) else (None if x is None or (isinstance(x, float) and math.isnan(x)) else str(x).strip())))


def build_side_books(K, C, notes):
    """books of a shop selling on credit / instalments / with trade-ins / warranty visits (public rulebook 'Cửa hàng điện thoại';
    src/industry.py reads them): instalment book -> C['instalments'] + C['instalment_payments']; debt book -> C['payments']
    + orders.paid_at_purchase; warranty book -> C['warranty']; trade-in book -> C['tradeins']."""
    O = C["orders"]; oid = _norm_ids(O["order_id"])
    O["order_id"] = oid
    if "total_payable" not in O and "paid_total" in O:                # 'Khách trả' of a hand-kept book = what the order costs
        O["total_payable"] = O["paid_total"]
    for ti in K.get("instalments", []):
        df = ti.df.copy(); g = ti.col
        k = g("order_id") or g("line_no"); idc = _norm_ids(df[k]).ffill() if k else None
        dc = next((c for c in ti.cols("order_date", dtype=DATE) if c != g("inst_pay_date")), None)
        head = df[dc].notna() if dc else pd.Series(True, index=df.index)
        I = dm.frame(df[head], order_id=idc[head] if idc is not None else None, buy_date=dc, customer_name=g("customer_name"), phone=g("phone"),
                     item_name=g("item_name") or g("tradein_item"), price=g("unit_price", "price", "line_amount", "total_payable", dtype=NUM),
                     prepaid=g("prepaid", dtype=NUM), n_terms=g("n_terms", dtype=NUM), per_term=g("per_term", dtype=NUM), due_day=g("due_day", dtype=NUM))
        C["instalments"] = I.reset_index(drop=True)
        pc, ac = g("inst_pay_date", dtype=DATE), g("inst_paid", dtype=NUM)
        if pc and ac:
            P = pd.DataFrame({"order_id": idc, "pay_date": pd.to_datetime(df[pc], errors="coerce").dt.normalize(), "amount": _num(df[ac])})
            C["instalment_payments"] = P.dropna(subset=["pay_date", "amount"]).reset_index(drop=True)
        notes.append(f"instalment book {ti.name}: {len(I)} contracts"); break
    for ti in K.get("debt_book", []):
        df = ti.df.copy(); g = ti.col
        nc = g("note"); dc = next((c for c in ti.cols("order_date", dtype=DATE)), None)
        ref = df[nc].map(_order_ref) if nc else None
        if ref is None or ref.notna().mean() < 0.5: continue
        ref = ref.ffill(); owed = _num(df[g("debt_end")]); head = owed.notna() & (df[dc].notna() if dc else True)
        after = pd.Series(owed[head].values, index=ref[head].values); after = after[~after.index.duplicated()]
        pc, ac = g("pay_time", dtype=DATE), g("paid_total", "amount", dtype=NUM)
        pay = pd.DataFrame({"order_id": ref, "pay_date": pd.to_datetime(df[pc], errors="coerce").dt.normalize() if pc else pd.NaT,
                            "amount": _num(df[ac]) if ac else np.nan}).dropna(subset=["pay_date", "amount"])
        tot = pd.to_numeric(O["total_payable"], errors="coerce") if "total_payable" in O else None
        if tot is not None:
            pap = tot.copy()
            m = O["order_id"].isin(set(after.index))
            pap[m] = (tot[m] - O.loc[m, "order_id"].map(after)).clip(lower=0)
            if C.get("instalments") is not None and "order_id" in C["instalments"]:
                mi = O["order_id"].isin(set(C["instalments"]["order_id"].dropna().astype(str)))
                pap[mi] = tot[mi]                                         # instalments are owed through the instalment book
            O["paid_at_purchase"] = pap
        C["payments"] = pay.reset_index(drop=True)
        notes.append(f"debt book {ti.name}: {len(after)} orders owing after purchase, {len(pay)} repayments"); break
    for ti in K.get("warranty_book", []):
        g = ti.col; rc = g("received_date", dtype=DATE)
        bc = next((c for c in ti.cols("order_date", dtype=DATE) if c != rc), None)
        C["warranty"] = dm.frame(ti.df, date=rc, customer_name=g("customer_name"), phone=g("phone"), item_name=g("item_name") or g("tradein_item"),
                                 imei=g("imei"), buy_date=bc, fault=g("fault")).reset_index(drop=True)
        notes.append(f"warranty book {ti.name}: {len(ti.df)} visits"); break
    for ti in K.get("tradein_book", []):
        g = ti.col
        T_ = dm.frame(ti.df, date=g("order_date", dtype=DATE), customer_name=g("customer_name"), phone=g("phone"), item_name=g("tradein_item") or g("item_name"),
                      imei=g("imei"), amount=g("tradein_price", dtype=NUM), order_id=g("tradein_order"))
        if "order_id" in T_: T_["order_id"] = T_["order_id"].map(_order_ref)
        C["tradeins"] = T_.reset_index(drop=True)
        notes.append(f"trade-in book {ti.name}: {len(T_)} phones taken"); break
    C["orders"] = O
    return C


def build(biz):
    """raw tables of one business -> (C, NOTES, report)."""
    rd = parse_readme(biz.readme)
    infos = OrderedDict((t, TableInfo(biz, t, rd)) for t in biz.tables)
    K = defaultdict(list)
    for ti in infos.values():
        ti.kind = classify(ti)
        if ti.kind: K[ti.kind].append(ti)
    notes = []
    items = build_items(K, notes)
    cv = build_conversions(K, rd, items, biz)
    O, L, rep = build_sales(biz, infos, K, rd, items, cv, notes)
    O = build_customers(K, O)
    O = debts_from_notes(O, rd, notes)
    C = {"orders": O, "lines": L}
    mem = build_member_list(K)
    if mem is not None: C["customers"] = mem
    for st in K.get("staff_list", []):
        g = st.col
        C["staff"] = dm.frame(st.df, staff=g("staff_name") or g("staff"), commission_pct=g("commission_pct", dtype=NUM), start_date=g("start_date", dtype=DATE))
        break
    if items is not None: C["items"] = items
    if cv is not None: C["conversions"] = cv
    P = build_purchases(K, cv, items)
    if P is not None: C["purchases"] = P
    C = build_side(K, C, cv, items, notes)
    C = build_side_books(K, C, notes)
    if "lines" in C and "order_id" in C["lines"]: C["lines"]["order_id"] = _norm_ids(C["lines"]["order_id"])
    rep["kinds"] = {t: ti.kind for t, ti in infos.items()}
    used = {t for t, ti in infos.items() if ti.kind is not None and ti.kind not in ("suppliers",)} | set(rep.get("lookups", []))
    if K.get("suppliers") and "purchases" in C: used |= {ti.name for ti in K["suppliers"]}
    rep["used_tables"] = sorted(used); rep["unused"] = [t for t in infos if t not in used]
    rep["fields"] = {k: v for k, v in (rep.get("fields") or {}).items()}
    return C, "; ".join(notes), rep


if __name__ == "__main__":
    import pickle, json
    biz = pickle.load(open(sys.argv[1], "rb"))
    C, notes, rep = build(biz)
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=str)[:3000])
    for k, v in C.items(): print(k, v.shape, list(v.columns)[:25])
    print(notes)
