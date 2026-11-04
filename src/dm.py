#!/usr/bin/env python3
"""DataMate toolkit: rulebook semantics as small, tested functions over canonical tables.

The LLM no longer re-derives the rulebook in every program. It (1) writes a short mapping program per business that
renames raw columns into canonical frames and calls `finalize` (the rulebook's "how files are written" rules live here),
and (2) answers each question with a few calls to the period helpers and metrics below.

Periods  : a Period is (start, end) of whole days, both inclusive.
Canonical: C["orders"] one row per order, C["lines"] one row per order line, optional C["items"], C["purchases"],
           C["payments"], C["stock"], C["lots"], C["prices"], C["expenses"], C["deliveries"], C["terms"]
           (column lists in CANON_DOC).
Nothing here is keyed on a business or question id."""
import re, sys, math, unicodedata
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd

D = pd.Timedelta(days=1)


# ------------------------------------------------------------------ periods (rulebook §2)
@dataclass(frozen=True)
class Period:
    start: pd.Timestamp
    end: pd.Timestamp                       # inclusive (whole day)
    label: str = ""

    def __post_init__(self):
        object.__setattr__(self, "start", pd.Timestamp(self.start).normalize())
        object.__setattr__(self, "end", pd.Timestamp(self.end).normalize())

    @property
    def days(self): return int((self.end - self.start) / D) + 1

    def mask(self, s):
        """bool mask of datetime Series s inside the period (time of day ignored)."""
        d = pd.to_datetime(s).dt.normalize()
        return (d >= self.start) & (d <= self.end)

    def dates(self): return pd.date_range(self.start, self.end, freq="D")

    def shift_days(self, n): return Period(self.start + n * D, self.end + n * D, f"{self.label}{n:+d}d")

    def __repr__(self): return f"Period({self.start.date()}..{self.end.date()}{' ' + self.label if self.label else ''})"


def _ts(x):
    """a day; a Period where a day is meant (stock_at(item, today_p(TODAY))) is its last day (31 failed attempts in v4b)."""
    if isinstance(x, Period): x = x.end
    return pd.Timestamp(x).normalize()


def day(d): d = _ts(d); return Period(d, d, f"day {d.date()}")
def today_p(today): return day(today)
def yesterday(today): return day(_ts(today) - D)
def between(a, b): return Period(_ts(a), _ts(b), "range")


def month(y, m):
    s = pd.Timestamp(int(y), int(m), 1); return Period(s, s + pd.offsets.MonthEnd(0), f"{int(y)}-{int(m):02d}")


def month_noyear(m, today):
    """'tháng M' without a year: this year's month M unless it starts after today, else last year's."""
    t = _ts(today); y = t.year if pd.Timestamp(t.year, int(m), 1) <= t else t.year - 1
    return month(y, m)


def quarter(y, q):
    s = pd.Timestamp(int(y), 3 * (int(q) - 1) + 1, 1); return Period(s, s + pd.offsets.QuarterEnd(0), f"{int(y)}Q{int(q)}")


def quarter_noyear(q, today):
    t = _ts(today); y = t.year if pd.Timestamp(t.year, 3 * (int(q) - 1) + 1, 1) <= t else t.year - 1
    return quarter(y, q)


def year(y): return Period(pd.Timestamp(int(y), 1, 1), pd.Timestamp(int(y), 12, 31), str(y))


def this_month(today):
    """'tháng này': day 1 .. today (inclusive)."""
    t = _ts(today); return Period(t.replace(day=1), t, "this month")


def prev_month_same_days(today):
    """comparison base of 'tháng này': last month from day 1 to the same day number (or its last day)."""
    t = _ts(today); s = (t.replace(day=1) - D).replace(day=1); last = s + pd.offsets.MonthEnd(0)
    e = s.replace(day=min(t.day, last.day)); return Period(s, e, "prev month same days")


def prev_month(p):
    """the whole calendar month before the (month) period p; for 'tháng này' (this_month: day 1 .. today) the rulebook base is the
    same number of days of last month (prev_month_same_days): the programs wrote prev_month(this_month(TODAY)) and compared
    a partial month with a whole one (public dev_005: +15.6 % vs -51.69 %)."""
    if p.label == "this month": return prev_month_same_days(p.end)
    s = (p.start.replace(day=1) - D).replace(day=1); return month(s.year, s.month)


def same_month_last_year(p): return month(p.start.year - 1, p.start.month)


def this_week(today):
    t = _ts(today); return Period(t - t.weekday() * D, t, "this week")


def last_week(today):
    t = _ts(today); mon = t - (t.weekday() + 7) * D; return Period(mon, mon + 6 * D, "last week")


def last_n_days(n, today):
    """'N ngày gần nhất': N consecutive days ending today (today included)."""
    t = _ts(today); return Period(t - (int(n) - 1) * D, t, f"last {n} days")


def ytd(today):
    t = _ts(today); return Period(t.replace(month=1, day=1), t, "year to date")


def until(d, start=None):
    """'đến hết ngày X': from the beginning of the data up to the end of day X."""
    return Period(_ts(start) if start is not None else pd.Timestamp(1900, 1, 1), _ts(d), f"until {_ts(d).date()}")


def months_back(p_or_today, n):
    """the n calendar months ending with the month of p (or of the date), oldest first."""
    s = (p_or_today.start if isinstance(p_or_today, Period) else _ts(p_or_today)).replace(day=1)
    out = []
    for i in range(n - 1, -1, -1):
        m = s - pd.DateOffset(months=i); out.append(month(m.year, m.month))
    return out


def month_seq(y1, m1, y2, m2):
    """months from (y1, m1) to (y2, m2) inclusive, oldest first."""
    out, s, e = [], pd.Timestamp(int(y1), int(m1), 1), pd.Timestamp(int(y2), int(m2), 1)
    while s <= e: out.append(month(s.year, s.month)); s += pd.DateOffset(months=1)
    return out


_CAL = {}


def set_calendar(cal):
    c = cal.copy(); c["date"] = pd.to_datetime(c["date"]); _CAL["cal"] = c


def lunar_to_solar(ly, lm, ld, leap=False):
    cal = _CAL["cal"]
    r = cal[(cal.lunar_year == int(ly)) & (cal.lunar_month == int(lm)) & (cal.lunar_day == int(ld)) &
            (cal.lunar_leap.astype(str).str.lower().eq("true") == bool(leap))]
    return r.date.iloc[0] if len(r) else pd.NaT


def tet(y):
    """'Tết năm Y': solar day of 23 tháng Chạp (lunar year Y-1) .. end of mùng 10 tháng Giêng (lunar year Y)."""
    return Period(lunar_to_solar(int(y) - 1, 12, 23), lunar_to_solar(int(y), 1, 10), f"Tết {y}")


def before_tet(y, n=None):
    """'so với trước Tết': the same number of days (or n days) right before 23 tháng Chạp."""
    t = tet(y); n = n or t.days; return Period(t.start - n * D, t.start - D, f"{n} days before Tết {y}")


def tet_of(today):
    """'Tết năm nay': the Tết whose lunar new year falls in the solar year of today."""
    return tet(_ts(today).year)


# ------------------------------------------------------------------ numbers / rounding / ranking
def round_half_up(x, ndigits=0):
    q = Decimal(1).scaleb(-ndigits)
    v = float(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))
    return int(v) if ndigits == 0 else v


def money(x):
    """money is an integer number of VND (half rounds up)."""
    return round_half_up(x, 0)


def pct(part, whole, nd=2):
    """part / whole in percent, rounded half-up to nd decimals (None when whole is 0)."""
    if whole is None or whole == 0 or (isinstance(whole, float) and math.isnan(whole)): return None
    return round_half_up(100.0 * part / whole, nd)


def growth(new, old, nd=2):
    """percent change from old to new (negative when it fell)."""
    return pct(new - old, old, nd)


def rank(values, k=None, ascending=False, codes=None):
    """order a {name: measure} dict / Series by measure (descending by default), ties by code (or name) ascending.
    Returns the list of names (top k when k is given)."""
    s = pd.Series(values, dtype="float64") if not isinstance(values, pd.Series) else values.astype("float64")
    s = s.dropna()
    key = pd.Series({n: (codes.get(n, n) if isinstance(codes, dict) else n) for n in s.index})
    df = pd.DataFrame({"v": s, "key": key.astype(str)})
    df = df.sort_values(["v", "key"], ascending=[ascending, True], kind="mergesort")
    out = list(df.index)
    return out[:k] if k else out


# ------------------------------------------------------------------ text
def norm_text(s):
    s = unicodedata.normalize("NFC", str(s)).lower()
    s = re.sub(r"[^\w\s]|_", " ", s); return re.sub(r"\s+", " ", s).strip()


def strip_accents(s):
    s = unicodedata.normalize("NFD", str(s)).replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFC", "".join(c for c in s if unicodedata.category(c) != "Mn"))


def key(s): return strip_accents(norm_text(s))


HONORIFIC = re.compile(r"^(?:a|anh|c|chi|e|em|co|chu|bac|thay|tho)\s+(?=\S)")   # not 'ông' / 'bà' (also surnames)


# Regional words for the same thing (North / Central -> one form): questions name items in another region's words
# ('Canh chua cá tràu' = 'Canh chua cá lóc', 'Đậu phụ rán giòn' = 'Đậu hũ chiên giòn', 'Cốc sạc' = 'Củ sạc'). Used only when no
# item matches the name itself; longer phrases first.
REGIONAL = [("pho mát", "phô mai"), ("phó mát", "phô mai"), ("phomat", "phô mai"), ("xây trát", "xây tô"), ("vữa trát", "vữa tô"),
            ("đậu phụ", "đậu hũ"), ("đậu hủ", "đậu hũ"), ("tàu hũ", "đậu hũ"), ("tàu hủ", "đậu hũ"), ("cá tràu", "cá lóc"), ("cá quả", "cá lóc"),
            ("cá chuối", "cá lóc"), ("cốc sạc", "củ sạc"), ("hoa quả", "trái cây"), ("thịt lợn", "thịt heo"), ("găng tay", "bao tay"),
            ("tất chân", "bao chân"), ("mì chính", "bột ngọt"), ("xì dầu", "nước tương"), ("bí ngô", "bí đỏ"), ("mướp đắng", "khổ qua"),
            ("rau mùi", "ngò"), ("giò lụa", "chả lụa"), ("hột vịt", "trứng vịt"), ("hột gà", "trứng gà"), ("xà phòng", "xà bông"),
            ("bánh mỳ", "bánh mì"), ("rán", "chiên"), ("ngao", "nghêu"), ("lợn", "heo"), ("ngô", "bắp"), ("lạc", "đậu phộng"),
            ("vừng", "mè"), ("dứa", "thơm"), ("khóm", "thơm"), ("tất", "vớ"), ("bát", "chén"), ("thìa", "muỗng"), ("sắn", "khoai mì"),
            ("cốc", "ly"), ("kệ", "giá đỡ"), ("quả", "trái")]
_REGIONAL_RE = [(re.compile(r"(?<!\w)" + re.escape(a) + r"(?!\w)"), b) for a, b in REGIONAL]


def regional_key(s):
    """matching key with regional synonyms brought to one form (accents kept while replacing: 'lợn' is not 'lon')."""
    t = re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s)).lower())
    for rx, b in _REGIONAL_RE: t = rx.sub(b, t)
    return _ckey(t)


LINE_REST = re.compile(r"(?:\d+\s*/\s*)?\d+\s*(?:gb|tb)\b")      # rulebook (phones): dòng máy = the name before '…GB'


def person_key(s):
    """key of a person's name without the honorific a book puts in front ('A.Phúc', 'a Phúc', 'anh Phúc' -> 'phuc')."""
    return HONORIFIC.sub("", key(s))


def _ckey(s):
    """key for matching names: a number glued to its unit or not is the same name ('chai 30 ml' = 'chai 30ml')."""
    return re.sub(r"(?<=\d) (?=(?:ml|l|g|kg|mg|cm|mm|m|m2|m3|gr|lit|oz|cc|w|kw|v)\b)", "", key(s))


CANON_STATUS = ("hoàn thành", "huỷ", "trả hàng", "đang giao")


def canon_status(label, default="hoàn thành"):
    """map a raw status / note label to one of the four canonical statuses by keywords (None/blank -> default)."""
    if label is None or (isinstance(label, float) and math.isnan(label)): return default
    k = key(label)
    if not k or k in ("nan", "none"): return default
    if re.search(r"\b(huy|cancel|canceled|cancelled|bo don|khach bom|bom hang|khong giao duoc|giao that bai|giao khong thanh cong|that bai|failed)\b", k): return "huỷ"
    if re.search(r"\b(tra hang|tra lai|tra ve|hoan tra|hoan hang|return|returned|refund|doi tra|doi hang|tra 1 phan|tra mot phan|khach tra|"
                 r"mang tra|dem tra|gui tra|bi tra|tra do|tra sp|tra san pham)\b", k): return "trả hàng"
    if re.search(r"\b(xong|hoan thanh|hoan tat|ok|done|complete|completed|thanh cong|da giao|da nhan|da lay|da thanh toan|success|paid|delivered)\b", k): return "hoàn thành"
    if re.search(r"\b(dang giao|giao hang|cho giao|dang gui|dang ship|shipping|chua giao|dang van chuyen|van chuyen|dang giao dich|dang xu ly|cho xu ly|pending|processing|in transit)\b", k): return "đang giao"
    return None


PAYMENTS = [("cod", r"\b(cod|thu ho|ship cod)\b"), ("ghi nợ", r"\b(debt|ghi no|cong no|no|mua chiu|tra cham|credit|tra sau)\b"),
            ("trả góp", r"\b(tra gop|installment)\b"),
            ("tiền mặt", r"\b(cash|tien mat|tm)\b"), ("chuyển khoản", r"\b(transfer|bank|chuyen khoan|ck|banking|qr)\b"),
            ("ví điện tử", r"\b(ewallet|e wallet|vi dien tu|vi|momo|zalopay|zalo pay|vnpay|shopeepay)\b"), ("thẻ", r"\b(card|the|pos|visa|master)\b")]
CHANNELS = [("sàn TMĐT", r"\b(shopee|lazada|tiki|tiktok|san|tmdt|marketplace|san tmdt)\b"), ("zalo", r"\b(zalo|zalo oa|zalooa)\b"),
            ("điện thoại", r"\b(phone|dien thoai|goi dien|hotline|call)\b"), ("facebook", r"\b(facebook|fb|messenger)\b"),
            ("website", r"\b(web|website|online)\b"), ("grab", r"\b(grab|grabfood|shopeefood|baemin|gojek|be)\b"),
            ("tại quầy", r"\b(store|tai quay|tai cua hang|offline|truc tiep|tai quan|tai tiem|tai shop|tai sap|pos|quay|in store|walk in)\b")]


def _canon_label(x, table):
    if x is None or (isinstance(x, float) and math.isnan(x)): return None
    k = key(x)
    for lab, pat in table:
        if re.search(pat, k): return lab
    return str(x)


def canon_payment(x):
    """payment method label -> tiền mặt | chuyển khoản | ví điện tử | cod | ghi nợ | trả góp | thẻ (else the raw label)."""
    return _canon_label(x, PAYMENTS)


def canon_channel(x):
    """sales channel label -> tại quầy | điện thoại | zalo | sàn TMĐT | facebook | website | grab (else the raw label)."""
    return _canon_label(x, CHANNELS)


def map_status(series, overrides=None):
    """canonical status for every raw label of a column (overrides: {raw label: canonical}). Blank cells stay blank
    (finalize copies the order's first line, then defaults to 'hoàn thành'); unknown labels are kept as they are so
    that check() reports them."""
    ov = {key(k): v for k, v in (overrides or {}).items()}
    def f(x):
        if x is None or (isinstance(x, float) and math.isnan(x)) or not key(x) or key(x) in ("nan", "none"): return None
        if key(x) in ov: return ov[key(x)]
        return canon_status(x, None) or x
    return series.map(f)


# ------------------------------------------------------------------ canonical tables
ORDER_COLS = ["order_id", "order_time", "order_date", "customer_id", "customer_name", "customer_group", "staff", "channel",
              "payment_method", "status", "valid", "revenue", "order_discount", "shipping_fee", "vat", "total_payable",
              "paid_at_purchase", "note", "due_date", "debt_end", "tradein", "points_discount", "plate", "vehicle", "work", "warranty"]   # last 6: industries (src/industry.py)
LINE_COLS = ["order_id", "line_no", "order_time", "order_date", "status", "valid", "item_code", "item_name", "item_group", "unit",
             "qty", "factor", "qty_base", "base_unit", "unit_price", "line_discount", "line_amount", "unit_cost", "returned_qty", "imei"]
WALKIN = re.compile(r"^(khach le|khach vang lai|vang lai|le|khach|walk in|walkin|khach hang le|kh le|kl|k l|khl|kvl|guest|none|nan|0|-|khong ten|an danh)$")


def canon_spelling(series):
    """replace every spelling of a name by the most frequent spelling with the same key (case, spacing, punctuation and
    accents ignored), so that grouping by name follows the rulebook (§3, §8)."""
    v = series.dropna()
    if not len(v): return series
    k = v.map(key); best = pd.DataFrame({"k": k, "v": v}).groupby(["k", "v"]).size().reset_index(name="n")
    best = best.sort_values(["k", "n", "v"], ascending=[True, False, True]).drop_duplicates("k").set_index("k")["v"]
    return series.map(lambda x: best.get(key(x), x) if x is not None and not (isinstance(x, float) and math.isnan(x)) else x)


def is_walkin(name):
    if name is None or (isinstance(name, float) and math.isnan(name)): return True
    return bool(WALKIN.match(key(name))) or not key(name)


def _first_nonnull(g):
    v = g.dropna(); return v.iloc[0] if len(v) else None


_SUMMARY_ID = re.compile(r"^(?:(?:tổng|cộng)(?:\s+(?:cộng|ngày|tháng|tuần|năm|quý|tiền|số|\d+)|\s*$)|(?:tong|cong)(?:\s+(?:cong|ngay|thang|tuan|quy|tien|\d+)|\s*$)"
                         r"|total|sum|subtotal|grand total)\b")             # with accents: 'Tống Nam Hương' is a customer
_SUMMARY_COLS = ("order_id", "customer_name", "customer_id", "note", "staff")


def _summary_rows(df):
    """summary rows of an export: 'tổng 01/05' in the order-code column (dev_010, one per day) or 'Cộng ngày 01/05/2025' in
    the customer column with a blank code (biz_110: 428 rows; the blank code made each one a line of the order above,
    revenue +100 %). A row is a summary when an order-level text cell reads tổng / cộng (+ ngày, tháng, tiền, a number
    ...) / total and the row has no date. Item names are not looked at ('Cộng cà phê' is an item)."""
    if not len(df): return pd.Series(False, index=df.index)
    m = pd.Series(False, index=df.index)
    for c in _SUMMARY_COLS:
        if c in df.columns: m |= df[c].map(lambda x: isinstance(x, str) and bool(_SUMMARY_ID.match(norm_text(x))))
    if "order_date" in df.columns: m &= pd.to_datetime(df["order_date"], errors="coerce").isna()
    elif "order_time" in df.columns: m &= pd.to_datetime(df["order_time"], errors="coerce").isna()
    if m.any():
        c = next(c for c in _SUMMARY_COLS if c in df.columns and df.loc[m, c].map(lambda x: isinstance(x, str)).any())
        print(f"finalize: {int(m.sum())} summary rows dropped ({c} like {df.loc[m, c].dropna().iloc[0]!r}, no date)")
    return m


_ADJ = [("order_discount", re.compile(r"^(?:voucher|ma giam gia|ma giam|giam gia(?: hoa don| don| ca don)?|khuyen mai(?: hoa don| don)?|coupon|"
                                      r"chiet khau(?: hoa don| don| ca don)?)\b")),
        ("points_discount", re.compile(r"^(?:(?:dung|tru|doi|su dung) \d* ?diem|diem (?:thanh vien|tich luy))\b")),
        ("shipping_fee", re.compile(r"^(?:phi ship|phi giao hang|phi giao|phi van chuyen|tien ship|ship hang|ship)\b"))]


def _order_adjustments(L):
    """rows of a sales book that are not items but an adjustment of the whole invoice: 'Voucher' / 'Giảm giá hoá đơn' (order
    discount), 'Dùng 200 điểm' (paid with loyalty points), 'Phí ship' (delivery fee), written with the invoice number, a name and
    an amount but no quantity / unit price (public dev_013). Rulebook §1: an order's revenue is the sum of its ITEM lines, without
    order discounts or delivery fees -> such rows leave the lines and become the order columns order_discount / points_discount /
    shipping_fee (positive). Returns (lines, {column: Series order_id -> amount})."""
    if "item_name" not in L.columns or "line_amount" not in L.columns or "order_id" not in L.columns or not len(L): return L, {}
    noq = L["qty"].isna() if "qty" in L.columns else pd.Series(True, index=L.index)
    if "unit_price" in L.columns: noq = noq | L["unit_price"].isna()
    if "item_code" in L.columns: noq = noq & (L["item_code"].isna() | L["item_code"].astype(str).str.strip().isin(["", "nan", "None"]))
    k = L["item_name"].map(lambda x: key(x) if isinstance(x, str) else "")
    adj, drop = {}, pd.Series(False, index=L.index)
    for col, rx in _ADJ:
        m = noq & k.map(lambda x: bool(rx.match(x)))
        if m.sum() < 5: continue
        amt = L.loc[m, "line_amount"].abs()
        adj[col] = amt.groupby(L.loc[m, "order_id"].astype(str).str.strip()).sum(); drop |= m
    if drop.any():
        import os
        if os.environ.get("ZC_ADJ_ROWS", "keep") == "move":           # rulebook §1 read literally: not in the revenue
            print(f"finalize: {int(drop.sum())} invoice adjustment rows (voucher / points / shipping) moved from the lines to the order columns", flush=True)
            L = L[~drop].reset_index(drop=True)
        else:                                                          # default: the invoice total of the book (v23 public 0.87 vs
            L = L.copy(); L["is_adjustment"] = drop.values             # v24 0.747 with 'move'); never items in rankings / counts
            print(f"finalize: {int(drop.sum())} invoice adjustment rows (voucher / points / shipping) kept in the revenue, marked is_adjustment, "
                  "copied to the order columns", flush=True)
    return L, adj


def finalize(lines, orders=None, conversions=None, items=None, ffill_cols=None, dedupe=True, keep_service_lines=True):
    """Rulebook §1 + §8 normalisation. Returns (orders, lines) canonical frames.

    lines : DataFrame with canonical column names (LINE_COLS; at least order_id, and item_name or item_code, qty,
            line_amount). Order-level columns (ORDER_COLS) may also be on the lines (flat sales books).
    orders: optional DataFrame of order headers with canonical names (one row per order).
    conversions: optional DataFrame [item (name or code), unit, factor] (units -> base unit); items: catalogue with
            item_code / item_name / item_group / base_unit / unit_cost ... (merged into lines when missing).
    Rules applied: blank order-level cells inherit the order's first line; exact duplicate lines of the same order are
    dropped; qty_base = qty * factor; an order with any returned line or returned status is 'trả hàng'; valid = not
    cancelled and not returned; order revenue = sum of line amounts (after line discounts)."""
    L = lines.copy().reset_index(drop=True)
    if "order_id" not in L.columns and (orders is None or not len(orders)):
        # a sales book without order codes (README: 'không ghi mã đơn'): every row is one record, as the reference
        # mappings of such books do (58 failed mapping attempts in v6-v9 stopped here)
        L["order_id"] = "row" + pd.Series(range(len(L)), index=L.index).astype(str)
        print("finalize: the lines have no order_id: every row is one record (a sales book without order codes)")
    L = L[~_summary_rows(L)].reset_index(drop=True)                    # before a blank code continues the order above
    if "order_id" not in L.columns:
        raise ValueError("lines need an order_id column (the code shared by the lines of one order). When the file has one row per "
                         "order (no item lines), build lines from the same rows: frame(t, order_id=<order code column>, "
                         "line_amount=<tiền hàng column>), then finalize(lines, orders)")
    oid = L["order_id"].where(~L["order_id"].astype(str).str.strip().isin(["", "nan", "None", "NaN"]))
    L["order_id"] = oid.ffill().astype(str).str.strip()                # §8: a blank order id continues the order above
    order_level = [c for c in ORDER_COLS if c in L.columns and c not in ("order_id", "revenue", "valid")]
    ff = list(dict.fromkeys(order_level + (ffill_cols or [])))
    if ff:                                                             # §8: blank cells of later lines = the order's first line
        blank = L[ff].apply(lambda s: s.isna() | s.astype(str).str.strip().isin(["", "nan", "None", "NaT"]))
        L[ff] = L[ff].mask(blank)
        g = L.groupby("order_id", sort=False)[ff]; L[ff] = g.ffill(); L[ff] = L.groupby("order_id", sort=False)[ff].bfill()
    if dedupe:                                                         # §8: identical lines of the same order = export error
        cols = [c for c in L.columns if c not in ("line_no", "_source", "_sheet", "_row")]
        L = L[~L.duplicated(subset=cols, keep="first")].reset_index(drop=True)
    if "order_time" in L.columns and "order_date" not in L.columns: L["order_date"] = pd.to_datetime(L["order_time"]).dt.normalize()
    if "order_date" in L.columns: L["order_date"] = pd.to_datetime(L["order_date"]).dt.normalize()
    L = L[~_summary_rows(L)].reset_index(drop=True)
    for c in ("qty", "unit_price", "line_discount", "line_amount", "unit_cost", "factor", "qty_base", "returned_qty"):
        if c in L.columns: L[c] = pd.to_numeric(L[c], errors="coerce")
    L, adj = _order_adjustments(L)                                     # voucher / points / shipping rows are not item lines
    # catalogue info
    if items is not None:                                              # catalogue columns joined by code, else by name
        it = items.copy()                                              # (names compared accent / case / spacing-insensitive)
        for kcol in ("item_code", "item_name"):
            if kcol in L.columns and kcol in it.columns:
                add = [c for c in it.columns if c not in L.columns]
                if add:
                    it["_k"] = it[kcol].map(key); m = it.drop_duplicates("_k").set_index("_k")[add]
                    L = pd.concat([L, m.reindex(L[kcol].map(key).values).set_axis(L.index)], axis=1)
                if kcol == "item_code" and "item_name" in it.columns:     # one spelling per item: the catalogue's
                    nm = dict(zip(it[kcol].map(key), it["item_name"]))
                    if "item_name" in L.columns: L["item_name_raw"] = L["item_name"]
                    L["item_name"] = L[kcol].map(key).map(nm).where(lambda x: x.notna(), L.get("item_name"))
                break
    # units
    if "factor" not in L.columns:
        L["factor"] = 1.0
        if conversions is not None and len(conversions) and "unit" in L.columns:
            cv = conversions.copy(); cv["_u"] = cv["unit"].map(key); cv["_i"] = cv["item"].map(key)
            fmap = {(i, u): float(f) for i, u, f in zip(cv["_i"], cv["_u"], cv["factor"]) if pd.notna(f)}
            ik = L["item_code"].map(key) if "item_code" in L.columns else None
            nk = L["item_name"].map(key) if "item_name" in L.columns else None
            uk = L["unit"].map(key)
            fac = []
            for j in range(len(L)):
                f = None
                for col in (ik, nk):
                    if col is not None and (col.iat[j], uk.iat[j]) in fmap: f = fmap[(col.iat[j], uk.iat[j])]; break
                fac.append(f if f is not None else 1.0)
            L["factor"] = fac
    L["factor"] = L["factor"].fillna(1.0)
    if "qty_base" not in L.columns and "qty" in L.columns: L["qty_base"] = L["qty"] * L["factor"]
    if "line_amount" not in L.columns and {"qty", "unit_price"} <= set(L.columns):
        L["line_amount"] = L["qty"] * L["unit_price"] - (L["line_discount"].fillna(0) if "line_discount" in L.columns else 0)
    # orders
    O = orders.copy() if orders is not None else None
    if O is not None:
        O["order_id"] = O["order_id"].astype(str).str.strip()
        if "order_date" in O.columns: O["order_date"] = pd.to_datetime(O["order_date"]).dt.normalize()
        O = O[~_summary_rows(O)]
        O = O.drop_duplicates("order_id", keep="first")
        for c in [c for c in O.columns if c in L.columns and c != "order_id"]:
            L = L.drop(columns=c)                                      # header table wins for order-level fields
        hdr = [c for c in O.columns if c in ORDER_COLS and c not in ("order_id", "revenue", "valid")]
        L = L.merge(O[["order_id"] + hdr], on="order_id", how="left")
    else:
        hc = [c for c in ORDER_COLS if c in L.columns and c not in ("order_id", "revenue", "valid")]
        O = L.groupby("order_id", sort=False)[hc].first().reset_index() if hc else L[["order_id"]].drop_duplicates()   # first non-null
    if "order_time" in O.columns and "order_date" not in O.columns: O["order_date"] = pd.to_datetime(O["order_time"]).dt.normalize()
    if "order_date" in O.columns: O["order_date"] = pd.to_datetime(O["order_date"]).dt.normalize()
    if "order_date" in O.columns and "order_date" not in L.columns: L = L.merge(O[["order_id", "order_date"]], on="order_id", how="left")
    # status: an order with any returned line (status or negative qty or returned_qty > 0) is 'trả hàng'
    if "status" not in O.columns: O["status"] = "hoàn thành"
    O["status"] = O["status"].where(O["status"].isin(CANON_STATUS) | O["status"].isna(), O["status"].map(lambda x: canon_status(x, None) or x))
    O["status"] = O["status"].fillna("hoàn thành")                    # no status written anywhere on the order
    ret_ids = set()
    if "status" in L.columns:
        ls = L["status"].where(L["status"].isin(CANON_STATUS) | L["status"].isna(), L["status"].map(lambda x: canon_status(x, None)))
        ret_ids |= set(L.loc[ls.eq("trả hàng"), "order_id"]); canc = set(L.loc[ls.eq("huỷ"), "order_id"])
        O.loc[O["order_id"].isin(canc) & O["status"].eq("hoàn thành"), "status"] = "huỷ"
    if "qty" in L.columns: ret_ids |= set(L.loc[L["qty"] < 0, "order_id"])
    if "returned_qty" in L.columns: ret_ids |= set(L.loc[L["returned_qty"].fillna(0) > 0, "order_id"])
    O.loc[O["order_id"].isin(ret_ids) & ~O["status"].eq("huỷ"), "status"] = "trả hàng"
    O["valid"] = ~O["status"].isin(["huỷ", "trả hàng"])
    # revenue of an order = sum of its line amounts (sales lines only: negative return lines are not revenue)
    if "line_amount" in L.columns:
        sale = L["qty"].fillna(1) > 0 if "qty" in L.columns else pd.Series(True, index=L.index)
        rev = L.loc[sale].groupby("order_id")["line_amount"].sum()
        O["revenue"] = O["order_id"].map(rev).fillna(0.0) if "revenue" not in O.columns or O["revenue"].isna().all() else O["revenue"]
    for col, ser in adj.items():                                       # the order columns the toolkit reads (positive amounts)
        if col not in O.columns or pd.to_numeric(O[col], errors="coerce").fillna(0).eq(0).all():
            O[col] = O["order_id"].astype(str).map(ser).fillna(0.0)
    L = L.drop(columns=[c for c in ("status", "valid") if c in L.columns]).merge(O[["order_id", "status", "valid"]], on="order_id", how="left")
    if "order_time" not in L.columns and "order_time" in O.columns: L = L.merge(O[["order_id", "order_time"]], on="order_id", how="left")
    for c in ("customer_id", "customer_name"):
        if c in O.columns: O[c] = O[c].where(~O[c].map(is_walkin), None) if c == "customer_name" else O[c].where(O[c].notna() & O[c].astype(str).str.strip().ne(""), None)
    for c in ("customer_name",):                                       # §8: names that differ only in spelling are one thing
        if c in O.columns: O[c] = canon_spelling(O[c])
    if "item_name" in L.columns: L["item_name"] = canon_spelling(L["item_name"])
    if "payment_method" in O.columns: O["payment_raw"] = O["payment_method"]; O["payment_method"] = O["payment_method"].map(canon_payment)
    if "channel" in O.columns: O["channel_raw"] = O["channel"]; O["channel"] = O["channel"].map(canon_channel)
    O = O[[c for c in ORDER_COLS if c in O.columns] + [c for c in O.columns if c not in ORDER_COLS]]
    L = L[[c for c in LINE_COLS if c in L.columns] + [c for c in L.columns if c not in LINE_COLS]]
    return O.reset_index(drop=True), L.reset_index(drop=True)


def find_table(T, *cols, optional=()):
    """the table of T that has every column in cols (exact names); ties -> the most rows. Lets one mapping program serve
    every export with the same layout whatever the file / sheet names. Raises KeyError when none has them."""
    best = None
    for name, df in T.items():
        if all(c in df.columns for c in cols):
            score = (sum(c in df.columns for c in optional), len(df))
            if best is None or score > best[0]: best = (score, name)
    if best is None: raise KeyError(f"no table has columns {list(cols)}")
    return T[best[1]]


def frame(df, **cols):
    """new DataFrame from a raw table: canonical_name=<raw column name | Series | scalar | function of the table or of a
    row>. Raw names that do not exist are skipped (listed in frame.missing; every skip of a mapping run in
    frame.skipped) so one mapping works for similar exports with fewer columns."""
    out, frame.missing = {}, []
    for k, v in cols.items():
        if isinstance(v, str):
            if v in df.columns: out[k] = df[v]
            else: frame.missing.append(v); frame.skipped.append((k, v))
        elif callable(v) and not isinstance(v, (pd.Series, pd.DataFrame)):
            try: r = v(df)                                               # a function of the table (vectorised) ...
            except Exception: r = None
            if not (isinstance(r, pd.Series) and len(r) == len(df)):
                try: r = df.apply(v, axis=1)                             # ... or of one row
                except Exception as e:
                    raise TypeError(f"frame(): the function given for {k}= fails on the table and on its rows ({type(e).__name__}: {e}). "
                                    f"Give a raw column name, a Series such as t['col'].map(f), or a function of the row (lambda r: ...)") from None
            out[k] = r
        elif v is not None: out[k] = v
    return pd.DataFrame(out, index=df.index)


frame.missing, frame.skipped = [], []


def readme_conversions(text):
    """unit conversions written in a README as '- <item>: 1 <unit> = <n> <base unit>' -> DataFrame item, unit, factor, base_unit."""
    rows = []
    for m in re.finditer(r"^\s*[-*]\s*(.+?):\s*1\s+(.+?)\s*=\s*([\d.,]+)\s+(.+?)\s*$", str(text), re.M):
        f = float(m.group(3).replace(",", ".")) if m.group(3).count(",") == 1 and "." not in m.group(3) else float(m.group(3).replace(",", ""))
        rows.append((m.group(1).strip(), m.group(2).strip(), f, m.group(4).strip()))
    return pd.DataFrame(rows, columns=["item", "unit", "factor", "base_unit"])


def conversions_table(df, item, unit, factor, base_unit=None):
    """conversion table from a raw table: columns item (name or code), unit (larger unit), factor (base units per unit)."""
    out = pd.DataFrame({"item": df[item], "unit": df[unit], "factor": pd.to_numeric(df[factor], errors="coerce")})
    if base_unit: out["base_unit"] = df[base_unit]
    return out.dropna(subset=["factor"])


def check(C):
    """schema checks of canonical tables -> list of problems (empty = OK)."""
    errs = []
    o, l = C.get("orders"), C.get("lines")
    if o is None: return ["C['orders'] is missing"]
    for c in ("order_id", "order_date", "status", "valid", "revenue"):
        if c not in o.columns: errs.append(f"orders lacks {c}")
    if errs: return errs
    if o["status"].isna().any(): errs.append(f"{int(o['status'].isna().sum())} orders have an unmapped status (labels: {sorted(set(map(str, o.loc[o['status'].isna(), 'status'])))[:5]})")
    bad = sorted(set(o["status"].dropna()) - set(CANON_STATUS))
    if bad: errs.append(f"non-canonical statuses {bad[:6]}")
    if o["order_id"].duplicated().any(): errs.append("order_id not unique in orders")
    if o["order_date"].isna().all(): errs.append("every order_date is empty: give frame() the NAME of the raw date column (order_date='Ngày'); "
                                                 "the raw date columns are already parsed, do not call parse_date on a column name")
    elif o["order_date"].isna().mean() > 0.01: errs.append(f"{int(o['order_date'].isna().sum())} orders without a date")
    if l is not None and "order_id" in l.columns:
        miss = ~l["order_id"].isin(set(o["order_id"]))
        if miss.mean() > 0.01: errs.append(f"{int(miss.sum())} lines with an order_id not in orders")
    return errs


CANON_DOC = """C["orders"]: order_id, order_time, order_date, customer_id, customer_name (None = walk-in), customer_group, staff, channel, payment_method, status (hoàn thành|huỷ|trả hàng|đang giao), valid (bool: counts for revenue), revenue (rulebook revenue of the order), order_discount, shipping_fee, vat, total_payable, paid_at_purchase, due_date (ngày phải trả of a credit order), debt_end (money still owed at the end of the data when debts are written only as notes: debt_from_note(note, total_payable))
C["lines"]: order_id, order_time, order_date, status, valid, item_code, item_name, item_group, unit, qty, factor, qty_base, base_unit, unit_price, line_discount, line_amount, unit_cost
C["items"]: item_code, item_name, item_group, base_unit, unit_cost, price, size (the catalogue's size / number column: 'M', 'XL', 'Số 4'; when it has one), ... ; C["purchases"]: receipt_id, receipt_date, supplier, item_code, item_name, unit, qty, factor, qty_base, unit_cost, amount, cancelled
C["payments"]: order_id, pay_time, pay_date, amount, method ; C["stock"]: item_code, item_name, qty_base (snapshot at stock_date) ; C["lots"]: lot_id, item_code, expiry, qty_in_base, unit_cost_base, ...
C["prices"]: item_code / item_name, date, price ; C["expenses"]: date, amount, category
C["lot_sales"]: which lot each sold quantity came from: order_id, lot_id, qty_base, returned_base ; C["ledger"]: stock movements (thẻ kho): item_code, time, kind, qty_base, stock_after (stock after the movement)
C["deliveries"]: one row per delivered line of a trip: order_id, delivery_date, trip_no (number of the trip of that order), item_name, qty_base, driver, plate ; C["conversions"]: item, unit, factor (base units per unit), base_unit"""


# ------------------------------------------------------------------ shop: metrics over canonical tables
def fee_lines(l):
    """lines that are a fee, not an item ('Phí phục vụ', 'Phí giao hàng', 'Phụ phí ...', item group 'phí dịch vụ'): out of
    item counts and rankings when the mapping sets no is_service flag."""
    m = pd.Series(False, index=l.index)
    for c in ("item_name", "item_group"):
        if c in l.columns:
            k = l[c].map(lambda x: key(x) if isinstance(x, str) else "")
            m |= k.str.match(r"(phi|phu phi)\b")
    if "is_adjustment" in l.columns: m |= l["is_adjustment"].fillna(False).astype(bool)   # voucher / points / shipping rows
    return m


class Unanswerable(Exception):
    """the released data cannot answer (rulebook §7): raise it from a program to answer `unanswerable`."""


_MONEY_TXT = re.compile(r"\d[\d.,]*\s*(?:tr(?:iệu)?\s*\d*|triệu|nghìn|ngàn|k(?!\w)|đồng|đ(?!\w)|vnd)?", re.I)


def money_in_text(s):
    """every money amount written in a free-text note, in order ('khách đưa trước 50k, còn thiếu 611 nghìn' -> [50000, 611000];
    '1.200.000đ', '1tr2', '2 triệu'); [] for an empty note."""
    if s is None or (isinstance(s, float) and math.isnan(s)): return []
    import ingest
    out = []
    for m in _MONEY_TXT.finditer(str(s)):
        v = ingest.parse_num(m.group(0).strip())
        if v is not None and not (isinstance(v, float) and math.isnan(v)): out.append(float(v))
    return out


_SETTLED = re.compile(r"\b(da tra du|tra du|da tra het|tra het|tra xong|da thanh toan( du| het)?|tat toan|ck du|chuyen du|da du|xong no|het no|khong con no|da tra roi)\b")
_OWED_KW = re.compile(r"(thieu lai|con thieu|thieu|con no|con lai|ghi so|ghi no|\bno\b)")
_PAID_KW = re.compile(r"(dua truoc|tra truoc|da tra|\btra\b|\bck\b|chuyen khoan|dat coc|coc)")


def debt_from_note(note, total=None):
    """money still owed according to a free-text debt note written as the order's state at the end of the data (public dev_008:
    'ck 913 nghìn, thiếu lại 456.500đ' -> 456500, 'ghi sổ 443 nghìn' -> 443000, 'đưa trước 236k, còn lại ghi sổ' -> total - 236000,
    'nợ cả đơn' / 'chưa trả' -> total, 'nợ 1tr645, khách ck đủ rồi' -> 0). None when the note says nothing about a debt."""
    if note is None or (isinstance(note, float) and math.isnan(note)): return None
    k = key(note)
    if not re.search(r"\b(no|ghi so|thieu|chua tra|con lai|tra truoc|dua truoc)\b", k): return None
    if _SETTLED.search(k): return 0.0
    if re.search(r"\b(ca don|toan bo|het don)\b", k) or re.fullmatch(r"(?:khach )?chua tra(?: tien)?(?: gi)?", k):
        return float(total) if total is not None and not (isinstance(total, float) and math.isnan(total)) else None
    parts = re.split(r"[,;]| va ", str(note)); owed = None; paid = 0.0
    for part in parts:
        pk = key(part); amt = money_in_text(part)
        if _OWED_KW.search(pk) and amt: owed = (owed or 0.0) + amt[-1]
        elif _PAID_KW.search(pk) and amt: paid += amt[-1]
    if owed is not None: return owed
    if re.search(r"\b(con lai|phan con lai|con lai ghi so|con lai ghi no|ghi so phan con lai)\b", k) and total is not None:
        return max(float(total) - paid, 0.0)
    if re.search(r"\b(ghi so|ghi no|no|chua tra)\b", k) and total is not None and not money_in_text(note):
        return float(total)                                            # 'ghi nợ' with no amount: the whole order
    return None


def require(cond, why="the data does not record this"):
    if not cond: raise Unanswerable(why)


def _timed(s):
    """order_time must carry the time of day for shift / hour questions: a column of dates only (every value 00:00) would
    put every order before 06:00 and answer 0 (a mapping error, not 'no orders')."""
    t = pd.to_datetime(s, errors="coerce").dropna()
    if len(t) > 20 and ((t.dt.hour == 0) & (t.dt.minute == 0)).mean() > 0.98:
        raise RuntimeError("order_time holds dates without the time of day: map order_time to the order creation time")


import industry                                                        # public-rulebook industries (phone, repair, mother & baby)


class Shop(industry.IndustryMetrics):
    """Metrics of one business. Every method follows the rulebook defaults; arguments name the variations questions ask
    for. `today` = as-of date (records after it have not happened yet). A period lying completely outside the dates of
    the data raises Unanswerable (§7: no data is not zero); a partly covered period is computed on the covered part."""

    def __init__(self, C, today=None, meta=None):
        self.C = C; self.today = _ts(today) if today is not None else None; self.meta = meta or {}
        self.O = C["orders"]; self.L = C.get("lines")
        d = self.O["order_date"].dropna(); self.first, self.last = (d.min(), d.max()) if len(d) else (None, None)

    def cover(self, p, what="orders"):
        """raise Unanswerable when period p does not overlap the data's date range."""
        if p is None or self.first is None: return
        if p.end < self.first or p.start > max(self.last, self.today or self.last):
            raise Unanswerable(f"no {what} recorded in {p} (data {self.first.date()}..{self.last.date()})")

    # -- selection
    def orders(self, p=None, valid=True, **f):
        """orders of period p (by order date). valid=True: not cancelled/returned (§1); None: every status. A status filter
        for cancelled / returned orders looks at every status (n_orders(p, status="huỷ") was always 0 with valid=True)."""
        if valid is True and f.get("status") in ("huỷ", "trả hàng"): valid = None
        o = self.O; self.cover(p)
        if self.today is not None: o = o[o["order_date"] <= self.today]
        if p is not None: o = o[p.mask(o["order_date"])]
        if valid is True: o = o[o["valid"]]
        elif valid is False: o = o[~o["valid"]]
        return self._filter(o, **f)

    def lines(self, p=None, valid=True, **f):
        if valid is True and f.get("status") in ("huỷ", "trả hàng"): valid = None   # as in orders()
        l = self.L; self.cover(p)
        if self.today is not None: l = l[l["order_date"] <= self.today]
        if p is not None: l = l[p.mask(l["order_date"])]
        if valid is True: l = l[l["valid"]]
        elif valid is False: l = l[~l["valid"]]
        if "qty" in l.columns: l = l[l["qty"].fillna(1) > 0]
        return self._filter(l, **f)

    def _filter(self, df, item=None, group=None, customer=None, customer_group=None, staff=None, channel=None, payment=None,
                shift=None, status=None, where=None, **extra):
        for v, cols, what in ((item, ("item_code", "item_name"), "items"), (group, ("item_group",), "item groups"),
                              (customer, ("customer_id", "customer_name"), "customers"), (customer_group, ("customer_group",), "customer groups"),
                              (staff, ("staff",), "staff"), (channel, ("channel",), "sales channels"), (payment, ("payment_method",), "payment methods"),
                              (shift, ("order_time",), "the time of day")):
            if v is not None: self._need(df, cols, what)
        if item is not None: df = df[self._match(df, ("item_code", "item_name"), self._item_keys(item))]   # every code / spelling
        if group is not None: df = df[self._match(df, ("item_group",), group)]
        if customer is not None: df = df[self._match_person(df, ("customer_id", "customer_name"), customer)]
        if customer_group is not None: df = df[self._match(df, ("customer_group",), customer_group)]
        if staff is not None: df = df[self._match_person(df, ("staff",), staff)]
        if channel is not None: df = df[self._match(df, ("channel",), canon_channel(channel) if isinstance(channel, str) else channel)]
        if payment is not None: df = df[self._match(df, ("payment_method",), canon_payment(payment) if isinstance(payment, str) else payment)]
        if status is not None: df = df[df["status"].eq(status)]
        if shift is not None:
            _timed(df["order_time"]); h = pd.to_datetime(df["order_time"]).dt.hour
            df = df[(h >= 6) & (h <= 13)] if key(shift).startswith("sang") else df[(h >= 14) & (h <= 21)]
        if where is not None: df = df[where(df)]
        for c, v in extra.items():                                      # prescription=True, base_unit="hộp": a column of the frame
            if c not in df.columns: raise TypeError(f"unknown filter {c}= (no column {c!r}; use where=lambda d: ...)")
            if isinstance(v, (bool, np.bool_)): df = df[df[c].fillna(False).astype(bool) == bool(v)]
            elif isinstance(v, str): df = df[self._match(df, (c,), v)]
            else: df = df[df[c] == v]
        return df

    def _match_person(self, df, cols, v):
        """_match for a person; when no value matches exactly, the ONE value whose name parts fit: the nickname a book
        writes for a full name ('Lương Chí Đức' ~ 'Đức', or the reverse), or the name with a phone number after it
        ('Từ Hải Thanh' ~ 'Từ Hải Thanh 0359470831'). Two or more such values: no match (never merges people)."""
        m = self._match(df, cols, v)
        if m.any() or isinstance(v, (list, tuple, set)): return m
        q = person_key(v).split()
        if not q: return m
        hits = set()
        for c in cols:
            if c not in df.columns: continue
            for x in df[c].dropna().unique():
                kx = person_key(x).split()
                if kx and (kx == q[-len(kx):] or q == kx[-len(q):] or (kx[:len(q)] == q and all(t.isdigit() for t in kx[len(q):]))):
                    hits.add(x)
        if len({person_key(x) for x in hits}) == 1:                     # 'Phúc', 'a Phúc', 'A.Phúc': one person, every spelling
            return self._match(df, cols, list(hits))
        return m

    @staticmethod
    def _need(df, cols, what):
        """§7: filtering on information the shop does not record -> Unanswerable (not 0)."""
        if not any(c in df.columns and df[c].notna().any() for c in cols): raise Unanswerable(f"the data does not record {what}")

    @staticmethod
    def _match(df, cols, val, contains=False):
        """rows whose value in any of cols equals val (accent / case / punctuation-insensitive); contains=True also
        accepts values that contain val as whole words (payment 'cod' ~ 'Thu hộ (COD)')."""
        vals = val if isinstance(val, (list, tuple, set)) else [val]
        ks = {_ckey(v) for v in vals}
        def ok(x):
            if x is None or (isinstance(x, float) and math.isnan(x)): return False
            kx = _ckey(x)
            return kx in ks or (contains and any(re.search(r"(^|\s)" + re.escape(k) + r"(\s|$)", kx) for k in ks))
        m = pd.Series(False, index=df.index)
        for c in cols:
            if c in df.columns:
                u = {x: ok(x) for x in df[c].dropna().unique()}
                m |= df[c].map(u).fillna(False).astype(bool)
        return m

    # -- revenue / orders
    @staticmethod
    def _line_level(f): return any(k in f for k in ("item", "group", "where"))

    def revenue(self, p=None, **f):
        """rulebook revenue (§1) of valid orders in p. Item / group / where filters use line amounts (where: a function of
        the lines frame), the others order revenue."""
        if self._line_level(f): return money(self.lines(p, **f)["line_amount"].sum())
        return money(self.orders(p, **f)["revenue"].sum())

    def n_orders(self, p=None, valid=True, **f):
        """number of valid orders in p (item / group / where filters: orders having such a line)."""
        if self._line_level(f): return int(self.lines(p, valid=valid, **f)["order_id"].nunique())
        return int(len(self.orders(p, valid=valid, **f)))

    def cancel_rate(self, p):
        """percent of cancelled orders among every order placed in p (all statuses)."""
        a = self.orders(p, valid=None); return pct((a["status"] == "huỷ").sum(), len(a))

    def aov(self, p=None, **f):
        """average revenue per valid order (money)."""
        o = self.orders(p, **f); return money(o["revenue"].sum() / len(o)) if len(o) else None

    def daily_revenue(self, p, **f):
        o = self.orders(p, **f); s = o.groupby("order_date")["revenue"].sum()
        return s.reindex(p.dates(), fill_value=0.0)

    def weekday_avg(self, p, weekday, by="revenue", **f):
        """average per weekday of p (0 = Monday ... 6 = Sunday): the total over those days divided by the NUMBER of such days
        in p (within the data), days without orders included (public rulebook: 'Trung bình mỗi ngày thứ X của kỳ').
        by: orders | revenue | qty (base units)."""
        self.cover(p)
        days = pd.date_range(p.start, p.end, freq="D"); end = self.data_end() if hasattr(self, "data_end") else None
        if end is not None: days = days[days <= end]
        if self.today is not None: days = days[days <= self.today]
        days = days[days.weekday == int(weekday)]
        require(len(days), "the period has no such weekday")
        if by == "qty":
            l = self.lines(p, **f); tot = float(l.loc[pd.to_datetime(l["order_date"]).dt.weekday == int(weekday), "qty_base"].sum())
        else:
            o = self.orders(p, **f); o = o[pd.to_datetime(o["order_date"]).dt.weekday == int(weekday)]
            tot = float(o["revenue"].sum()) if by == "revenue" else float(len(o))
        return _num(round(tot / len(days), 2)) if by != "revenue" else money(tot / len(days))

    def best_day(self, p, by="revenue", **f):
        """date (YYYY-MM-DD) with the highest revenue (by='revenue') or most orders (by='orders'); ties -> earliest."""
        o = self.orders(p, **f)
        s = o.groupby("order_date")["revenue"].sum() if by == "revenue" else o.groupby("order_date").size()
        if not len(s): return None
        return s.sort_index().idxmax().strftime("%Y-%m-%d")

    def n_days_revenue_at_least(self, p, x, **f):
        s = self.daily_revenue(p, **f); return int((s >= x).sum())

    def n_orders_revenue_at_least(self, p, x, **f):
        o = self.orders(p, **f); return int((o["revenue"] >= x).sum())

    def monthly_revenue(self, months, **f): return [self.revenue(m, **f) for m in months]

    def best_month(self, months, **f):
        v = self.monthly_revenue(months, **f); i = int(np.argmax(v)); return months[i].start.strftime("%Y-%m")

    # -- items
    def item_qty(self, p, item=None, group=None, base_unit=None, **f):
        """quantity sold in base units (bỏ đơn huỷ/trả). base_unit='ly' keeps only items whose base unit is that unit."""
        l = self.lines(p, item=item, group=group, **f)
        if base_unit is not None:
            bu = l["base_unit"] if "base_unit" in l.columns else l.get("unit")
            l = l[bu.map(key) == key(base_unit)]
        return _num(l["qty_base"].sum())

    def item_sales(self, p, by="qty", **f):
        """{item_name: measure} over valid lines of p. by: qty (base units) | revenue | n_orders | profit."""
        l = self.lines(p, **f)
        if "is_service" in l.columns: l = l[~l["is_service"].fillna(False).astype(bool)]   # service fee lines are not items
        else: l = l[~fee_lines(l)]
        if "is_adjustment" in l.columns: l = l[~l["is_adjustment"].fillna(False).astype(bool)]
        name = "item_code" if "item_code" in l.columns and l["item_code"].notna().all() else "item_name"
        if by == "qty": s = l.groupby(name)["qty_base"].sum()
        elif by == "revenue": s = l.groupby(name)["line_amount"].sum()
        elif by == "n_orders":                                           # distinct orders per REPORTED name: codes that share
            nm_ = l[name]                                                 # a catalogue name are one item (a sum would count an
            if name == "item_code" and "item_name" in l.columns:          # order holding both codes twice)
                nm = l.drop_duplicates("item_code").set_index("item_code")["item_name"]; nm_ = l["item_code"].map(lambda c: nm.get(c, c))
            return l.groupby(nm_)["order_id"].nunique()
        elif by == "profit":
            require("unit_cost" in l.columns and l["unit_cost"].notna().any(), "no cost price on the sale lines")
            s = (l["line_amount"] - l["qty"] * l["unit_cost"]).groupby(l[name]).sum()
        else: raise ValueError(by)
        if name == "item_code" and "item_name" in l.columns:                          # report the catalogue name
            nm = l.drop_duplicates("item_code").set_index("item_code")["item_name"]
            s.index = [nm.get(c, c) for c in s.index]; s = s.groupby(level=0).sum()
        return s

    def top_items(self, p, k=1, by="qty", **f):
        s = self.item_sales(p, by=by, **f); r = rank(s, k, codes=self.item_codes())
        return r

    def item_codes(self):
        l = self.L
        if l is None or "item_code" not in l.columns or "item_name" not in l.columns: return None
        return dict(zip(l["item_name"], l["item_code"].astype(str)))

    def item_name(self, x):
        """catalogue name of an item given by code or by any spelling."""
        for df in (self.C.get("items"), self.L):
            if df is not None and "item_name" in df.columns:
                m = self._match(df, ("item_code", "item_name"), x)
                if m.any(): return df.loc[m, "item_name"].iloc[0]
        return x

    # -- customers / staff
    def customer_key(self):
        o = self.O
        if "customer_id" in o.columns and o["customer_id"].notna().any(): return "customer_id"
        require("customer_name" in o.columns and o["customer_name"].notna().any(), "the data does not record customers")
        return "customer_name"

    def customer_sales(self, p, by="revenue", **f):
        o = self.orders(p, valid=(None if by == "n_cancelled" else True), **f); k = self.customer_key(); o = o[o[k].notna()]
        if by == "revenue": return o.groupby(k)["revenue"].sum()
        if by == "n_orders": return o.groupby(k).size()
        if by == "n_days": return o.groupby(k)["order_date"].nunique()
        if by == "aov": g = o.groupby(k)["revenue"]; return (g.sum() / g.size())[g.size() >= 3]
        if by == "n_cancelled": return o[o["status"] == "huỷ"].groupby(k).size()
        raise ValueError(by)

    def top_customers(self, p, k=1, by="revenue", **f):
        s = self.customer_sales(p, by=by, **f); r = rank(s, k)
        if self.customer_key() == "customer_id": r = [self.customer_name(c) for c in r]
        return r

    def customer_name(self, cid):
        o = self.O; m = o.loc[o["customer_id"].astype(str) == str(cid), "customer_name"].dropna()
        return m.iloc[0] if len(m) else cid

    def n_customers(self, p, min_orders=1, **f):
        s = self.customer_sales(p, by="n_orders", **f); return int((s >= min_orders).sum())

    def staff_sales(self, p, by="revenue", **f):
        """{staff: revenue | orders | n_days} over valid orders of p (n_days: working days = days with at least one order of
        that person, public rulebook 'chăm chỉ nhất')."""
        o = self.orders(p, **f); o = o[o["staff"].notna()]
        if by == "n_days": return o.groupby("staff")["order_date"].nunique()
        return o.groupby("staff")["revenue"].sum() if by == "revenue" else o.groupby("staff").size()

    # -- debt (§4)
    def _debt_orders(self, d):
        """orders that can carry debt, placed up to day d: not cancelled, not COD / marketplace."""
        o = self.O[(self.O["order_date"] <= _ts(d)) & (self.O["status"] != "huỷ")]
        if "payment_method" in o.columns:
            o = o[~o["payment_method"].isin(["cod", "sàn TMĐT"])]
        if "channel" in o.columns:
            o = o[~o["channel"].isin(["sàn TMĐT"])]
        return o

    def paid_until(self, d):
        """{order_id: amount paid up to the end of day d} (payments table, else paid_at_purchase). With both, paid_at_purchase is
        what was paid when buying and C['payments'] the later repayments (the debt-book layout, public dev_014: 'Nợ là số còn
        nợ sau khi trả lúc mua'): paid = paid_at_purchase + repayments up to d; a payment row on the order day of the same
        amount is that purchase payment itself (not counted twice). No training reference sets both."""
        P = self.C.get("payments")
        if P is not None and len(P):
            P = P[pd.to_datetime(P["pay_date"]).dt.normalize() <= _ts(d)]
            paid = P.groupby(P["order_id"].astype(str))["amount"].sum()
            o = self.O
            if "paid_at_purchase" in o.columns and pd.to_numeric(o["paid_at_purchase"], errors="coerce").notna().any():
                pap = pd.Series(pd.to_numeric(o["paid_at_purchase"], errors="coerce").fillna(0).values, index=o["order_id"].astype(str).values)
                pap = pap[~pap.index.duplicated()]
                od = pd.Series(pd.to_datetime(o["order_date"]).dt.normalize().values, index=o["order_id"].astype(str).values)
                od = od[~od.index.duplicated()]
                pk = P["order_id"].astype(str)
                same = (pd.to_datetime(P["pay_date"]).dt.normalize().values == pk.map(od).values) & \
                       ((pd.to_numeric(P["amount"], errors="coerce").values - pk.map(pap).fillna(-1e18).values).__abs__() < 1)
                dup = set(pk[same])
                pap = pap[[i not in dup for i in pap.index]]
                paid = paid.add(pap, fill_value=0)
            return paid
        o = self.O; return pd.Series(o["paid_at_purchase"].values, index=o["order_id"]) if "paid_at_purchase" in o.columns else pd.Series(dtype=float)

    def debts(self, d):
        """Series order_id -> amount still owed at the end of day d (> 0 only): the debt book, plus the unpaid part of
        instalment contracts (C['instalments'], public rulebook 'Cửa hàng điện thoại': an instalment order is a credit order)."""
        return self._debt_frame(d).set_index("order_id")["owed"]

    def _debt_frame(self, d):
        """DataFrame [order_id, owed, customer, order_date] of every order with money owed at the end of day d."""
        inst = self._inst_debts(d) if self.C.get("instalments") is not None else None
        try: s = self._book_debts(d)
        except Unanswerable:
            if inst is None or not len(inst): raise
            s = pd.Series(dtype=float)
        o = self.O.set_index("order_id")
        try: k = self.customer_key()                                    # debt totals need no customers; per-customer views do
        except Unanswerable: k = None
        base = pd.DataFrame({"order_id": s.index, "owed": s.values,
                             "customer": o.loc[s.index, k].values if k else None, "order_date": o.loc[s.index, "order_date"].values})
        if inst is None or not len(inst): return base
        base = base[~base["order_id"].astype(str).isin(set(inst["order_id"].astype(str)))]          # the instalment book is
        oc = o[k] if k else None                                                                    # what is owed for those
        if oc is not None:
            oc = oc[~oc.index.duplicated()]
            inst = inst.assign(customer=[oc[i] if i in oc.index and isinstance(oc[i], str) and oc[i].strip() else c
                                         for i, c in zip(inst["order_id"], inst["customer"])])
        return pd.concat([base, inst], ignore_index=True)

    def _book_debts(self, d):
        require("total_payable" in self.O.columns and self.O["total_payable"].notna().any(), "the data does not record order totals (no debts)")
        P = self.C.get("payments")                                     # without any record of what was paid, every order would
        require((P is not None and len(P)) or any(c in self.O.columns and self.O[c].notna().any()     # count as unpaid
                                                  for c in ("paid_at_purchase", "paid_total", "debt_end")), "the data does not record payments (no debts)")
        P = self.C.get("payments")
        if "debt_end" in self.O.columns and self.O["debt_end"].notna().any() and not (P is not None and len(P)) \
                and "paid_at_purchase" not in self.O.columns:
            # the shop records only each order's debt at the END of the data (notes: debt_from_note): known from that day on
            end = self.data_end()
            require(end is None or _ts(d) >= end, f"the data records debts only as their state at the end of the data ({end.date() if end is not None else ''})")
            o = self._debt_orders(d); s = pd.Series(pd.to_numeric(o["debt_end"], errors="coerce").fillna(0).values, index=o["order_id"].values)
            return s[s > 0.5]
        o = self._debt_orders(d); paid = self.paid_until(d)
        owed = o["total_payable"].values - o["order_id"].map(paid).fillna(0).values
        s = pd.Series(owed, index=o["order_id"].values); return s[s > 0.5]

    def debt_total(self, d): return money(self.debts(d).sum())

    def debt_by_customer(self, d):
        f = self._debt_frame(d); f = f[f["customer"].notna()]
        if self.C.get("instalments") is not None and len(f):                # two books spell one customer differently
            pk = f["customer"].map(lambda x: person_key(re.sub(r"[\d\s.\-()+]{8,}$", "", str(x))) if isinstance(x, str) else x)
            lab = f.groupby(pk)["customer"].agg(lambda v: v.value_counts().index[0])
            return f.groupby(pk)["owed"].sum().rename(index=lab)
        return f.groupby("customer")["owed"].sum()

    def n_debtors(self, d): return int((self.debt_by_customer(d) > 0.5).sum())

    def top_debtors(self, d, k=3):
        s = self.debt_by_customer(d); r = rank(s, k)
        names = [self.customer_name(c) if self.customer_key() == "customer_id" else c for c in r]
        return {"columns": ["khach_hang", "so_tien_no"], "rows": [[n, money(s[c])] for n, c in zip(names, r)]}

    def debt_aging(self, d, bins=((0, 7), (8, 15), (16, None))):
        f = self._debt_frame(d); s = f["owed"]; age = (_ts(d) - pd.to_datetime(f["order_date"])).dt.days.values
        out = []
        for lo, hi in bins:
            m = (age >= lo) & ((age <= hi) if hi is not None else True); out.append(money(s.values[m].sum()))
        return out

    def debt_older_than(self, d, n):
        """debt of orders placed more than n days before d ('đơn đặt đã hơn n ngày')."""
        f = self._debt_frame(d); age = (_ts(d) - pd.to_datetime(f["order_date"])).dt.days.values
        return money(f["owed"].values[age > n].sum())

    def debt_flow(self, p, customer=None):
        """[new debt, repaid] of credit orders in p: new = unpaid part on the purchase day of orders placed in p,
        repaid = payments dated in p made after the purchase day (any credit order). Instalment contracts (phone shops): new =
        the instalment total of contracts bought in p, repaid = instalments paid in p."""
        if self.C.get("instalments") is None: return self._book_flow(p, customer)
        try: new, rep = self._book_flow(p, customer)
        except Unanswerable: new, rep = 0, 0
        I = self._inst(p.end)
        if customer is not None: I = self._inst_who(I, customer)
        new += float(I.loc[p.mask(I["buy_date"]), "total"].sum())
        P = self.C.get("instalment_payments")
        if P is not None and len(P):
            P = P[P["order_id"].astype(str).str.strip().isin(set(I["order_id"]))]
            rep += float(pd.to_numeric(P.loc[p.mask(pd.to_datetime(P["pay_date"]).dt.normalize()), "amount"], errors="coerce").sum())
        return [money(new), money(rep)]

    def _book_flow(self, p, customer=None):
        o = self._debt_orders(p.end)
        if "payment_method" in o.columns: o = o[o["payment_method"].isin(["ghi nợ", "trả góp"])]
        if customer is not None: o = o[self._match(o, ("customer_id", "customer_name"), customer)]
        P = self.C.get("payments"); require(P is not None and len(P), "the data does not record payments of credit orders")
        require("total_payable" in o.columns, "the data does not record order totals (no debts)")
        P = P[P["order_id"].astype(str).isin(set(o["order_id"]))].copy()
        P["pay_date"] = pd.to_datetime(P["pay_date"]).dt.normalize()
        od = o.set_index("order_id")["order_date"]; P["od"] = P["order_id"].astype(str).map(od)
        same = P[P["pay_date"] == P["od"]].groupby("order_id")["amount"].sum()
        op = o[p.mask(o["order_date"])]
        new = (op["total_payable"] - op["order_id"].map(same).fillna(0)).clip(lower=0).sum()
        rep = P[(P["pay_date"] > P["od"]) & p.mask(P["pay_date"])]["amount"].sum()
        return [money(new), money(rep)]


    # -- shares / comparisons
    def share(self, p, **f):
        """percent of the shop's revenue in p that comes from the filtered orders / lines (denominator: whole shop, walk-in
        orders included)."""
        return pct(self.revenue(p, **f), self.revenue(p))

    def increasing(self, periods, strict=True, **f):
        """True when revenue rises from each period to the next."""
        v = self.monthly_revenue(periods, **f); return all((b > a) if strict else (b >= a) for a, b in zip(v, v[1:]))

    def peak_hour(self, p, **f):
        """starting hour of the one-hour window with the most orders (order creation time)."""
        self._need(self.O, ("order_time",), "the time of day")
        _timed(self.O["order_time"]); o = self.orders(p, valid=None, **f); h = pd.to_datetime(o["order_time"]).dt.hour.value_counts()
        return int(h.sort_index().idxmax()) if len(h) else None

    def size_qty(self, p, size, **f):
        """quantity of items of a size. When the catalogue records sizes (column size of the lines or of C['items']), by that
        column ('Size XL' = 'XL', 'Số 4' = '4'; an item with an empty size has none, whatever its name says: public rulebook);
        otherwise items whose name ends with '(M)' / '(L)' (case-insensitive). A base unit given as the size
        (size_qty(p, "ly"): a frequent slip of the programs for 'bán được bao nhiêu ly') counts that unit:
        item_qty(p, base_unit=...)."""
        l = self.lines(p, **f)
        sz = lambda v: re.sub(r"^(?:size|so|co)\s+", "", key(v)).strip() if v is not None and not (isinstance(v, float) and math.isnan(v)) else None
        col = l["size"] if "size" in l.columns and l["size"].notna().any() else None
        it = self.C.get("items")
        if col is None and it is not None and {"item_name", "size"} <= set(it.columns) and it["size"].notna().any():
            mp = dict(zip(it["item_name"].map(key), it["size"])); col = l["item_name"].map(lambda x: mp.get(key(x)))
        if col is not None: return _num(l.loc[col.map(sz) == sz(size), "qty_base"].sum())
        m = l["item_name"].astype(str).str.strip().str.lower().str.endswith(f"({str(size).lower()})") | \
            l["item_name"].map(lambda n: bool(re.search(r"(?:^|\s)size " + re.escape(sz(size) or "#") + r"(?:\s|$)", key(n))))   # '… size M 76 miếng'
        if not m.any() and str(size).strip().upper() not in ("S", "M", "L", "XL", "XXL", "NB", "XXXL"):
            bu = l["base_unit"] if "base_unit" in l.columns else l.get("unit")
            if bu is not None and (bu.map(key) == key(size)).any(): return self.item_qty(p, base_unit=size, **f)
        return _num(l.loc[m, "qty_base"].sum())

    def unsold_items(self, p):
        """items sold (valid orders) before p but not in p -> list of names."""
        l = self.lines(None); l = l[~fee_lines(l)]; before = set(l.loc[l["order_date"] < p.start, "item_name"])
        during = set(l.loc[p.mask(l["order_date"]), "item_name"])
        return sorted(before - during)

    # -- purchases (§5)
    def _purchases(self, p=None, item=None):
        P = self.C.get("purchases"); require(P is not None and len(P), "no purchase records")
        P = P.copy(); P["receipt_date"] = pd.to_datetime(P["receipt_date"]).dt.normalize()
        if "cancelled" in P.columns: P = P[~P["cancelled"].fillna(False).astype(bool)]
        if self.today is not None: P = P[P["receipt_date"] <= self.today]
        if p is not None: P = P[p.mask(P["receipt_date"])]
        if item is not None: P = P[self._match(P, ("item_code", "item_name"), self._item_keys(item))]
        return P

    def _item_keys(self, item):
        """an item given by name or code (or a list of them) -> every name / code of that item (catalogue + lines)."""
        ks = set(map(str, item)) if isinstance(item, (list, tuple, set)) else {item}
        found = False
        for df in (self.C.get("items"), self.L):
            if df is not None and {"item_code", "item_name"} <= set(df.columns):
                m = self._match(df, ("item_code", "item_name"), item)
                found |= bool(m.any())
                ks |= set(df.loc[m, "item_code"].astype(str)) | set(df.loc[m, "item_name"].astype(str))
        if not found and isinstance(item, str):                          # a model line: the name before the storage size
            k = _ckey(item)                                              # ('Xiaomi Redmi Note 14' = every '… Note 14 128GB Đen')
            for df in (self.C.get("items"), self.L):
                if df is None or "item_name" not in df.columns: continue
                names = pd.Series(df["item_name"].dropna().astype(str).unique())
                hit = names[names.map(lambda n: _ckey(n).startswith(k + " ") and bool(LINE_REST.match(_ckey(n)[len(k) + 1:])))]
                ks |= set(hit)
                if "item_code" in df.columns and len(hit): ks |= set(df.loc[df["item_name"].astype(str).isin(set(hit)), "item_code"].dropna().astype(str))
                found |= bool(len(hit))
        if not found and isinstance(item, str):                          # regional words ('cá tràu' = 'cá lóc')
            rk = regional_key(item)
            for df in (self.C.get("items"), self.L):
                if df is None or "item_name" not in df.columns: continue
                names = pd.Series(df["item_name"].dropna().astype(str).unique())
                hit = names[names.map(regional_key) == rk]
                ks |= set(hit)
                if "item_code" in df.columns and len(hit): ks |= set(df.loc[df["item_name"].astype(str).isin(set(hit)), "item_code"].dropna().astype(str))
        return list(ks)

    def purchase_total(self, p):
        """money of every receipt dated in p, cancelled receipts excluded."""
        P = self._purchases(p); amt = P["amount"] if "amount" in P.columns else P["qty"] * P["unit_cost"]
        return money(amt.sum())

    def avg_purchase_price(self, item, p, unit=None):
        """quantity-weighted average purchase price of item in p, per purchase unit (or per `unit`)."""
        P = self._purchases(p, item)
        if unit is not None and "unit" in P.columns and (P["unit"].map(key) == key(unit)).any(): P = P[P["unit"].map(key) == key(unit)]
        amt = P["amount"] if "amount" in P.columns else P["qty"] * P["unit_cost"]
        require(P["qty"].sum() > 0, f"no purchase of {item} in {p}")
        return money(amt.sum() / P["qty"].sum())

    def supplier_ranking(self, p, by="amount"):
        """{supplier: purchase money} (by='amount') or {supplier: number of receipts} (by='receipts')."""
        P = self._purchases(p); col = "supplier_name" if "supplier_name" in P.columns and P["supplier_name"].notna().any() else "supplier"
        if by == "amount": return (P["amount"] if "amount" in P.columns else P["qty"] * P["unit_cost"]).groupby(P[col]).sum()
        return P.groupby(col)["receipt_id"].nunique()

    def top_supplier(self, p, by="amount"):
        """supplier with the most purchase money (by='amount') or the most receipts ('nhà cung cấp chính', by='receipts')."""
        return rank(self.supplier_ranking(p, by), 1)[0]

    def purchased_qty_base(self, item, p):
        P = self._purchases(p, item); return float((P["qty"] * P.apply(lambda r: self.factor(item, r.get("unit")), axis=1)).sum()) if len(P) else 0.0

    def factor(self, item, unit):
        """base units in one `unit` of item (1 when unit is the base unit or unknown)."""
        if unit is None or (isinstance(unit, float) and math.isnan(unit)): return 1.0
        cv = self.C.get("conversions")
        if cv is not None and len(cv):
            ks = {key(x) for x in self._item_keys(item)}
            m = cv[cv["item"].map(key).isin(ks) & (cv["unit"].map(key) == key(unit))]
            if len(m): return float(m["factor"].iloc[0])
        l = self.L                                                     # learn it from the sales lines (qty_base / qty)
        if l is not None and {"unit", "qty", "qty_base"} <= set(l.columns):
            m = self._match(l, ("item_code", "item_name"), self._item_keys(item)) & (l["unit"].map(key) == key(unit)) & (l["qty"] > 0)
            if m.any(): return float((l.loc[m, "qty_base"] / l.loc[m, "qty"]).median())
        return 1.0

    def import_minus_sales(self, item, p):
        """purchased minus sold quantity of item in p, both in base units (sales = valid orders)."""
        return _num(self.purchased_qty_base(item, p) - self.item_qty(p, item=item))

    # -- stock (§5)
    def data_end(self):
        """last day of the data ('Dữ liệu có đến hết ...' in the README, else the last order day)."""
        e = self.meta.get("data_end"); return _ts(e) if e is not None else self.last

    def stock_at(self, item, d):
        """stock of item (base units) at the end of day d: ledger if any; else snapshot at the end of the data rolled back
        with purchases and sales after d; else (lots + lot_sales, no snapshot) the remaining quantity of its lots
        (lot_stock: matches the training answers of every lot-based pharmacy, 31 / 32 questions)."""
        d = _ts(d); ks = self._item_keys(item)
        lg = self.C.get("ledger")
        if lg is not None and len(lg):
            g = lg[self._match(lg, ("item_code", "item_name"), ks)].copy(); g["time"] = pd.to_datetime(g["time"])
            g = g[g["time"] < d + D].sort_values("time", kind="mergesort")
            return _num(g["stock_after"].iloc[-1]) if len(g) else 0
        st = self.C.get("stock"); lots = self.C.get("lots"); ls = self.C.get("lot_sales")
        if (st is None or not len(st)) and lots is not None and len(lots) and ls is not None and len(ls):
            require(self._match(lots, ("item_code", "item_name"), ks).any(), f"{item} has no lot")
            L = self.lot_stock(d)
            return _num(float(L.loc[self._match(L, ("item_code", "item_name"), ks), "remaining"].sum()))
        require(st is not None and len(st), "no stock records")
        m = st[self._match(st, ("item_code", "item_name"), ks)]; require(len(m), f"{item} is not in the stock list")
        q = float(m["qty_base"].sum()); end = self.data_end()
        if d >= end: return _num(q)
        after = Period(d + D, end)
        bought = self.purchased_qty_base(item, after) if self.C.get("purchases") is not None else 0.0
        sold = self.stock_out_base(item, after)
        return _num(q - bought + sold)

    def stock_out_base(self, item, p):
        """base units that left the stock in p for item: lines of non-cancelled orders minus returned quantities."""
        l = self.L; l = l[p.mask(l["order_date"]) & (l["status"] != "huỷ")]
        l = l[self._match(l, ("item_code", "item_name"), self._item_keys(item))]
        out = l.loc[l["qty_base"] > 0, "qty_base"].sum() + l.loc[l["qty_base"] < 0, "qty_base"].sum()
        if "returned_qty" in l.columns: out -= l["returned_qty"].fillna(0).sum()
        return float(out)

    def stock_split(self, item, d, big_unit):
        """[whole big units, remaining base units] of the stock at the end of day d."""
        q = self.stock_at(item, d); f = self.factor(item, big_unit); require(f > 1, f"no conversion for {big_unit}")
        return [int(q // f), _num(q - (q // f) * f)]

    def stock_value(self, d):
        """sum over items in stock (> 0) at the end of day d of stock x catalogue cost per base unit."""
        it = self.C.get("items")                                       # CANON_DOC names it unit_cost; older mappings: cost
        cc = next((c for c in ("cost", "unit_cost") if it is not None and c in it.columns and it[c].notna().any()), None)
        require(cc is not None, "no catalogue cost")
        require(pd.to_numeric(it[cc], errors="coerce").fillna(0).gt(0).any(), "the catalogue costs are all 0 (not recorded)")   # no data != 0
        tot = 0.0
        for _, r in it.iterrows():
            k = r["item_code"] if "item_code" in it.columns and pd.notna(r.get("item_code")) else r["item_name"]
            try: q = self.stock_at(k, d)
            except Unanswerable: continue
            if q > 0 and pd.notna(r[cc]): tot += q * float(r[cc])
        return money(tot)

    # -- lots / expiry (pharmacy)
    def lot_stock(self, d):
        """lots with remaining base quantity at the end of day d (received - sold by non-cancelled orders + returned)."""
        lots = self.C.get("lots"); require(lots is not None and len(lots), "no lot records")
        L = lots.copy(); ls = self.C.get("lot_sales")
        if ls is None or not len(ls):                                    # every lot would count as full (v13: 377 lots 'in stock'
            raise RuntimeError("C['lots'] has no C['lot_sales']: the quantity left in each lot is unknown; map the table that "
                               "records which lot each sold quantity came from")   # for 5): an error, not an answer
        if ls is not None and len(ls):
            o = self.O.set_index("order_id"); x = ls.copy(); x["order_id"] = x["order_id"].astype(str)
            x["order_date"] = x["order_id"].map(o["order_date"]); x["status"] = x["order_id"].map(o["status"])
            x = x[(x["order_date"] <= _ts(d)) & (x["status"] != "huỷ")]
            net = x["qty_base"].fillna(0) - (x["returned_base"].fillna(0) if "returned_base" in x.columns else 0)
            used = net.groupby(x["lot_id"]).sum(); L["remaining"] = L["qty_in_base"] - L["lot_id"].map(used).fillna(0)
        else: L["remaining"] = L["qty_in_base"]
        if "receipt_date" in L.columns: L = L[pd.to_datetime(L["receipt_date"]) <= _ts(d)]
        L["expiry"] = pd.to_datetime(L["expiry"]); return L[L["remaining"] > 0]

    def lots_expiring(self, d, until):
        """lots in stock at the end of day d whose expiry is in [d, until]."""
        L = self.lot_stock(d); return L[(L["expiry"] >= _ts(d)) & (L["expiry"] <= _ts(until))]

    def items_expiring(self, d, until):
        """items with a lot in stock at the end of day d expiring in [d, until] -> catalogue names, earliest expiry of the
        item first, then item code (rulebook: liệt kê hàng theo hạn dùng: hạn sớm nhất của mặt hàng trước, cùng hạn theo mã)."""
        L = self.lots_expiring(d, until)
        if not len(L): return []
        k = "item_code" if "item_code" in L.columns else "item_name"
        g = L.groupby(k)["expiry"].min().reset_index(); g["_c"] = g[k].astype(str)
        return [self.item_name(c) for c in g.sort_values(["expiry", "_c"], kind="mergesort")[k]]

    def out_of_stock(self, d):
        """items bought to be sold (in the stock list, the lots or the purchases: not dishes made on site) whose stock
        is 0 at the end of day d -> catalogue names by item code (rulebook: mặt hàng hết hàng cuối ngày)."""
        keys = []
        for t in ("stock", "lots", "purchases", "ledger"):
            df = self.C.get(t)
            if df is not None and len(df):
                k = "item_code" if "item_code" in df.columns else "item_name"
                keys = list(dict.fromkeys(df[k].dropna().astype(str))); break
        require(keys, "no stock records")
        out = []
        for c in sorted(keys):
            try:
                if float(self.stock_at(c, d)) <= 0: out.append(self.item_name(c))
            except Unanswerable: pass
        return out

    def nearest_expiry(self, item, d):
        L = self.lot_stock(d); L = L[self._match(L, ("item_code", "item_name"), self._item_keys(item)) & (L["expiry"] >= _ts(d))]
        require(len(L), f"no lot of {item} in stock"); return L["expiry"].min().strftime("%Y-%m-%d")

    # -- prices
    def price_on(self, item, d):
        """unit price (per base unit, before line discount) written on the lines of item on day d, any order status."""
        l = self.L; l = l[(l["order_date"] == _ts(d)) & self._match(l, ("item_code", "item_name"), self._item_keys(item))]
        require(len(l), f"no sale of {item} on {_ts(d).date()}")
        per = l["unit_price"] / l["factor"].replace(0, 1) if "factor" in l.columns else l["unit_price"]
        return money(per.mode().iloc[0])

    def price_change_date(self, item, new_price):
        """first day item's price rose to new_price (price list, else the sales lines)."""
        pr = self.C.get("prices")
        if pr is not None and len(pr):
            x = pr[self._match(pr, ("item_code", "item_name"), self._item_keys(item))].copy()
            x["date"] = pd.to_datetime(x["date"]); x = x.sort_values("date", kind="mergesort")
            prev = x["price"].shift(1); hit = x[(x["price"].round() == round(new_price)) & ((prev < x["price"]) | prev.isna())]
            if len(hit): return hit["date"].iloc[0].normalize()
        l = self.L[self._match(self.L, ("item_code", "item_name"), self._item_keys(item))].sort_values("order_date", kind="mergesort")
        per = (l["unit_price"] / l["factor"].replace(0, 1)).round()
        hit = l[per == round(new_price)]
        require(len(hit), f"no price {new_price} for {item}"); return hit["order_date"].iloc[0].normalize()

    def price_effect(self, item, n, date=None, new_price=None):
        """percent change of base quantity sold in the n days from the price rise (day included) vs the n days before."""
        D0 = _ts(date) if date is not None else self.price_change_date(item, new_price)
        after = self.item_qty(Period(D0, D0 + (n - 1) * D), item=item); before = self.item_qty(Period(D0 - n * D, D0 - D), item=item)
        return growth(after, before)

    # -- profit
    def gross_profit(self, p, **f):
        """revenue minus cost of goods sold, cost = qty x cost per sold unit written on the sale line (§5)."""
        l = self.lines(p, **f); require("unit_cost" in l.columns and self.L["unit_cost"].notna().any(), "no cost price on the sale lines")
        return money((l["line_amount"] - l["qty"] * l["unit_cost"]).sum())

    def net_profit(self, p):
        """gross profit of p minus every expense dated in p."""
        E = self.C.get("expenses"); require(E is not None and len(E), "no expense book")
        e = E[p.mask(pd.to_datetime(E["date"]))]["amount"].sum(); return money(self.gross_profit(p) - e)

    # -- deliveries (building materials)
    def n_delivery_trips(self, p):
        """delivery trips with a delivery date in p (every trip of every order counts, returned orders included)."""
        g = self.C.get("deliveries"); require(g is not None and len(g), "no delivery records")
        g = g.copy(); g["delivery_date"] = pd.to_datetime(g["delivery_date"]).dt.normalize()
        if self.today is not None: g = g[g["delivery_date"] <= self.today]
        g = g[p.mask(g["delivery_date"])]; return int(g[["order_id", "trip_no"]].drop_duplicates().shape[0])

    def _delivered(self, ordered, d=None, item=None):
        """delivery rows up to the end of day d (default today) of the orders PLACED in period `ordered` (not cancelled)."""
        g = self.C.get("deliveries"); require(g is not None and len(g), "no delivery records")
        g = g.copy(); g["delivery_date"] = pd.to_datetime(g["delivery_date"]).dt.normalize(); g["order_id"] = g["order_id"].astype(str)
        d = _ts(d) if d is not None else self.today
        if d is not None: g = g[g["delivery_date"] <= d]
        o = self.orders(ordered, valid=None); o = o[o["status"] != "huỷ"]
        g = g[g["order_id"].isin(set(o["order_id"].astype(str)))]
        if item is not None:
            names = self._item_keys(item) if hasattr(self, "_item_keys") else item
            g = g[self._match(g, ("item_code", "item_name"), names)]
        return g

    def delivered_qty(self, ordered, d=None, item=None):
        """quantity (base units) delivered up to the end of day d of the orders placed in `ordered` ('các đơn đặt tháng 3, tới hôm
        nay đã chở đi bao nhiêu thùng X'); every trip counts, also of orders returned later (rulebook 'Vật liệu xây dựng')."""
        return _num(pd.to_numeric(self._delivered(ordered, d, item)["qty_base"], errors="coerce").sum())

    def n_orders_by_trips(self, ordered, d=None, min_trips=2):
        """orders placed in `ordered` delivered in at least min_trips trips up to the end of day d ('đơn đã giao từ 2 đợt trở lên')."""
        g = self._delivered(ordered, d); col = "trip_no" if "trip_no" in g.columns else "delivery_date"
        return int((g.groupby("order_id")[col].nunique() >= int(min_trips)).sum())

    def n_due_orders(self, p=None, d=None, after=False):
        """credit orders (not cancelled, placed up to day d = today by default) whose due date (orders.due_date, 'ngày phải trả')
        falls in period p; after=True: due strictly after day d (a due date ON d is not later)."""
        require("due_date" in self.O.columns and self.O["due_date"].notna().any(), "the data does not record due dates of credit orders")
        d = _ts(d) if d is not None else self.today
        o = self.O[self.O["status"] != "huỷ"]
        if d is not None: o = o[o["order_date"] <= d]
        due = pd.to_datetime(o["due_date"], errors="coerce").dt.normalize(); o = o[due.notna()]; due = due[due.notna()]
        if after: return int((due > d).sum())
        return int(p.mask(due).sum()) if p is not None else int(len(o))

    def pct_customers(self, p, min_orders, **f):
        """percent of the customers who bought in p that have at least min_orders valid orders."""
        s = self.customer_sales(p, by="n_orders", **f); return pct((s >= min_orders).sum(), len(s))


def _num(x):
    x = float(x); return int(round(x)) if abs(x - round(x)) < 1e-9 else round(x, 2)


def shop_api(shop):
    """the metric functions bound to one shop, as plain callables for the program namespace."""
    names = ["orders", "lines", "revenue", "n_orders", "cancel_rate", "aov", "daily_revenue", "best_day", "n_days_revenue_at_least",
             "n_orders_revenue_at_least", "monthly_revenue", "best_month", "item_qty", "item_sales", "top_items", "customer_sales",
             "top_customers", "n_customers", "staff_sales", "debts", "debt_total", "debt_by_customer", "n_debtors", "top_debtors",
             "debt_aging", "debt_older_than", "debt_flow", "share", "increasing", "weekday_avg", "peak_hour", "size_qty", "unsold_items",
             "purchase_total", "avg_purchase_price", "supplier_ranking", "top_supplier", "import_minus_sales", "stock_at", "stock_split",
             "stock_value", "lot_stock", "lots_expiring", "items_expiring", "out_of_stock", "nearest_expiry", "price_on", "price_change_date", "price_effect",
             "gross_profit", "net_profit", "n_delivery_trips", "pct_customers", "factor", "delivered_qty", "n_orders_by_trips", "n_due_orders"] + industry.INDUSTRY_NAMES
    return {n: getattr(shop, n) for n in names}


PERIOD_API = {f.__name__: f for f in (day, today_p, yesterday, between, month, month_noyear, quarter, quarter_noyear, year, this_month,
                                      prev_month_same_days, prev_month, same_month_last_year, this_week, last_week, last_n_days,
                                      ytd, until, months_back, month_seq, tet, before_tet, tet_of, lunar_to_solar)}
NUM_API = {f.__name__: f for f in (round_half_up, money, pct, growth, rank, canon_status, map_status, canon_payment, canon_channel, key, require)}


def api(shop, today):
    """names available to answer programs: periods, number helpers and the shop's metrics."""
    g = dict(PERIOD_API); g.update(NUM_API); g.update(industry.HELPERS); g.update(shop_api(shop))
    g.update({"TODAY": _ts(today), "D": D, "Period": Period, "Unanswerable": Unanswerable, "C": shop.C, "pd": pd, "np": np, "math": math, "re": re})
    return g


_TOOLKIT_NS = {}                                                       # the toolkit of the program being executed


class _ToolkitModule(__import__("types").ModuleType):
    def __getattr__(self, name):
        if name in _TOOLKIT_NS: return _TOOLKIT_NS[name]
        raise AttributeError(f"the toolkit has no {name!r} (it is already imported: call the functions directly)")


class _ToolkitFinder:
    """`import datamate` / `from datamate_toolkit import revenue` (the generated programs tried this in 12.9 % of the
    train attempts and 21.5 % of the public ones, all ModuleNotFoundError): any module named like data(_)mate* or
    toolkit* is the toolkit namespace of the current program. Appended to sys.meta_path, so real modules come first."""
    import re as _re
    RX = _re.compile(r"(?i)^(data[_ ]?mate|datamate|toolkit|(data_)?builder)\w*$")   # builder*: mapping programs
    def find_spec(self, fullname, path=None, target=None):
        if self.RX.match(fullname.split(".")[0]):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(fullname, self)
        return None
    def create_module(self, spec): return _ToolkitModule(spec.name)
    def exec_module(self, module): pass


if not any(isinstance(f, _ToolkitFinder) for f in sys.meta_path): sys.meta_path.append(_ToolkitFinder())


def run_program(code, shop, today, extra=None):
    """execute an answer program -> ANSWER dict; Unanswerable -> {"type": "unanswerable"}."""
    g = api(shop, today)
    if extra: g.update(extra)
    _TOOLKIT_NS.clear(); _TOOLKIT_NS.update(g)
    try: exec(compile(code, "<program>", "exec"), g)
    except Unanswerable: return {"type": "unanswerable", "answer": None}
    return g.get("ANSWER")


API_DOC = """Toolkit (already imported; every metric follows the RULEBOOK defaults: valid orders = not cancelled and not returned,
revenue = line amounts after line discounts without VAT / delivery / order vouchers, records after TODAY ignored,
rankings descending with ties by code, money rounded half-up to whole VND, percentages to 2 decimals):
- Periods (whole days, both ends included): day("YYYY-MM-DD"), today_p(TODAY), yesterday(TODAY), between(a, b), month(y, m),
  month_noyear(m, TODAY) ('tháng M' with no year), quarter(y, q), quarter_noyear(q, TODAY), year(y), this_month(TODAY) (day 1 ..
  today), prev_month_same_days(TODAY) (the base 'cùng kỳ tháng trước' of this_month), prev_month(p), same_month_last_year(p),
  this_week(TODAY), last_week(TODAY) (full Monday-Sunday), last_n_days(n, TODAY), ytd(TODAY), until(d) (from the start of the
  data), month_seq(y1, m1, y2, m2), months_back(p, n), tet(y), tet_of(TODAY), before_tet(y, n=None). p.start, p.end, p.days, p.mask(series).
- Filters (keyword arguments of most metrics): item=, group=, customer=, customer_group=, staff=, channel=, payment=,
  shift="sáng"|"chiều", status=, where=lambda d: <boolean Series over the lines (or orders) frame>.
  Payments are canonical: "tiền mặt", "chuyển khoản", "ví điện tử", "cod", "ghi nợ", "thẻ"; channels: "tại quầy", "điện thoại", "zalo", "sàn TMĐT".
- Orders / revenue: revenue(p, **f), n_orders(p, **f), aov(p, **f), cancel_rate(p), orders(p, valid=True|None|False, **f) and
  lines(p, **f) -> DataFrames, daily_revenue(p), best_day(p, by="revenue"|"orders"), n_days_revenue_at_least(p, x),
  n_orders_revenue_at_least(p, x), monthly_revenue(periods), best_month(periods), increasing(periods), share(p, **f) (percent of
  the whole shop's revenue), weekday_avg(p, weekday 0=Monday), peak_hour(p), size_qty(p, "M"|"L").
- Items: item_qty(p, item=|group=, base_unit=None) (base units), item_sales(p, by="qty"|"revenue"|"n_orders"|"profit") -> Series,
  top_items(p, k, by="qty", **f) -> list, unsold_items(p), price_on(item, d), price_change_date(item, new_price),
  price_effect(item, n, date=None, new_price=None) (% change of quantity, n days from the rise vs n days before).
- Customers / staff: customer_sales(p, by="revenue"|"n_orders"|"n_days"|"aov"|"n_cancelled") -> Series, top_customers(p, k, by=...),
  n_customers(p, min_orders=1), pct_customers(p, min_orders), staff_sales(p, by="revenue"|"orders") -> Series.
- Debt (at the end of day d): debt_total(d), n_debtors(d), top_debtors(d, k) -> table, debt_aging(d, bins=((0,7),(8,15),(16,None))),
  debt_older_than(d, n), debt_flow(p, customer=None) -> [new credit, repaid], debts(d), debt_by_customer(d).
- Purchases / stock: purchase_total(p), avg_purchase_price(item, p, unit=None), top_supplier(p, by="amount"|"receipts"),
  supplier_ranking(p, by), import_minus_sales(item, p), stock_at(item, d), stock_split(item, d, big_unit) -> [big, base],
  stock_value(d), lot_stock(d), lots_expiring(d, until) -> DataFrame (remaining, unit_cost_base, expiry), nearest_expiry(item, d),
  items_expiring(d, until) -> names of items with a lot in stock expiring in [d, until] (earliest expiry first, then code),
  weekday_avg(p, weekday 0=Mon..6=Sun, by="revenue"|"orders"|"qty") -> per-weekday average, divided by the number of such days in p,
  out_of_stock(d) -> names of items bought to be sold whose stock is 0 at the end of d (by code),
  factor(item, unit).
- Profit: gross_profit(p, **f), net_profit(p); deliveries: n_delivery_trips(p), delivered_qty(ordered_period, d=TODAY, item=None),
  n_orders_by_trips(ordered_period, d=TODAY, min_trips=2); credit due dates: n_due_orders(p, d=TODAY) / n_due_orders(d=TODAY, after=True).
- Helpers: growth(new, old) (% change, negative when it fell), pct(part, whole), money(x), round_half_up(x, nd), rank(values, k),
  key(s) (accent-free comparison key), require(cond, why). raise Unanswerable(why) when the data cannot answer (rulebook §7);
  a metric raises it itself when the period has no data at all or the needed information is not recorded.
- C: the canonical tables (pandas DataFrames) for anything the metrics do not cover; T: the cleaned raw tables."""

MAP_DOC = """Builder API for the mapping program (already imported):
- tbl(*columns): the table of T that has all these columns (use it instead of T["name"] so that other exports of the same
  layout map too). frame(df, canonical_name=<raw column name | Series | value>, ...) -> DataFrame (missing raw names are skipped).
- map_status(series, {raw label: canonical}) -> canonical statuses ("hoàn thành", "huỷ", "trả hàng", "đang giao"); keyword
  rules cover the usual labels, give overrides only for labels they would miss. Blank cells stay blank (finalize fills them).
- finalize(lines, orders=None, conversions=None, items=None, dedupe=True) -> (orders, lines): applies the rulebook "how files
  are written" rules (blank order cells / ids inherit the order's first line, exact duplicate lines of one order dropped,
  qty_base = qty x factor, an order with any returned line is 'trả hàng', valid, order revenue = sum of line amounts).
  Use dedupe=False for books that are not written by order (every row is one record: give each row its own order_id).
- conversions_table(df, item_col, unit_col, factor_col, base_unit_col=None); readme_conversions(README) for '1 thùng = 24 lon' lines.
- canon_status, canon_payment, canon_channel, key, parse_num, parse_date, pd, np, re, README (text), T, RAW (original strings);
  money_in_text(note) -> amounts in a note; debt_from_note(note, total) -> money still owed by a free-text debt note."""
