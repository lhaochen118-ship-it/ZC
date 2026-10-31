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
import re, math, unicodedata
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


def _ts(x): return pd.Timestamp(x).normalize()


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
    """the whole calendar month before the (month) period p."""
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
    s = re.sub(r"[^\w\s]", " ", s); return re.sub(r"\s+", " ", s).strip()


def strip_accents(s):
    s = unicodedata.normalize("NFD", str(s)).replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFC", "".join(c for c in s if unicodedata.category(c) != "Mn"))


def key(s): return strip_accents(norm_text(s))


CANON_STATUS = ("hoàn thành", "huỷ", "trả hàng", "đang giao")


def canon_status(label, default="hoàn thành"):
    """map a raw status / note label to one of the four canonical statuses by keywords (None/blank -> default)."""
    if label is None or (isinstance(label, float) and math.isnan(label)): return default
    k = key(label)
    if not k or k in ("nan", "none"): return default
    if re.search(r"\b(huy|cancel|canceled|cancelled|bo don|khach bom|bom hang|khong giao duoc|giao that bai|giao khong thanh cong|that bai|failed)\b", k): return "huỷ"
    if re.search(r"\b(tra hang|tra lai|tra ve|hoan tra|hoan hang|return|returned|refund|doi tra|doi hang|tra 1 phan|tra mot phan|khach tra)\b", k): return "trả hàng"
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
            ("tại quầy", r"\b(store|tai quay|tai cua hang|offline|truc tiep|tai quan|pos|quay)\b")]


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
              "paid_at_purchase", "note"]
LINE_COLS = ["order_id", "line_no", "order_time", "order_date", "status", "valid", "item_code", "item_name", "item_group", "unit",
             "qty", "factor", "qty_base", "base_unit", "unit_price", "line_discount", "line_amount", "unit_cost", "returned_qty"]
WALKIN = re.compile(r"^(khach le|khach vang lai|vang lai|le|khach|walk in|walkin|khach hang le|kh le|guest|none|nan|0|-)$")


def is_walkin(name):
    if name is None or (isinstance(name, float) and math.isnan(name)): return True
    return bool(WALKIN.match(key(name))) or not key(name)


def _first_nonnull(g):
    v = g.dropna(); return v.iloc[0] if len(v) else None


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
    if "order_id" not in L.columns: raise ValueError("lines need an order_id column")
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
    for c in ("qty", "unit_price", "line_discount", "line_amount", "unit_cost", "factor", "qty_base", "returned_qty"):
        if c in L.columns: L[c] = pd.to_numeric(L[c], errors="coerce")
    # catalogue info
    if items is not None:                                              # catalogue columns joined by code, else by name
        it = items.copy()                                              # (names compared accent / case / spacing-insensitive)
        for kcol in ("item_code", "item_name"):
            if kcol in L.columns and kcol in it.columns:
                add = [c for c in it.columns if c not in L.columns]
                if add:
                    it["_k"] = it[kcol].map(key); m = it.drop_duplicates("_k").set_index("_k")[add]
                    L = L.join(m, on=L[kcol].map(key).rename("_k")) if False else pd.concat([L, m.reindex(L[kcol].map(key).values).set_axis(L.index)], axis=1)
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
    L = L.drop(columns=[c for c in ("status", "valid") if c in L.columns]).merge(O[["order_id", "status", "valid"]], on="order_id", how="left")
    if "order_time" not in L.columns and "order_time" in O.columns: L = L.merge(O[["order_id", "order_time"]], on="order_id", how="left")
    for c in ("customer_id", "customer_name"):
        if c in O.columns: O[c] = O[c].where(~O[c].map(is_walkin), None) if c == "customer_name" else O[c].where(O[c].notna() & O[c].astype(str).str.strip().ne(""), None)
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
    """new DataFrame from a raw table: canonical_name=<raw column name | Series | scalar>. Raw names that do not exist
    are skipped (listed in frame.missing) so one mapping works for similar exports with fewer columns."""
    out, frame.missing = {}, []
    for k, v in cols.items():
        if isinstance(v, str):
            if v in df.columns: out[k] = df[v]
            else: frame.missing.append(v)
        elif v is not None: out[k] = v
    return pd.DataFrame(out, index=df.index)


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
    if o["order_date"].isna().mean() > 0.01: errs.append(f"{int(o['order_date'].isna().sum())} orders without a date")
    if l is not None and "order_id" in l.columns:
        miss = ~l["order_id"].isin(set(o["order_id"]))
        if miss.mean() > 0.01: errs.append(f"{int(miss.sum())} lines with an order_id not in orders")
    return errs


CANON_DOC = """C["orders"]: order_id, order_time, order_date, customer_id, customer_name (None = walk-in), customer_group, staff, channel, payment_method, status (hoàn thành|huỷ|trả hàng|đang giao), valid (bool: counts for revenue), revenue (rulebook revenue of the order), order_discount, shipping_fee, vat, total_payable, paid_at_purchase
C["lines"]: order_id, order_time, order_date, status, valid, item_code, item_name, item_group, unit, qty, factor, qty_base, base_unit, unit_price, line_discount, line_amount, unit_cost
C["items"]: item_code, item_name, item_group, base_unit, unit_cost, price, ... ; C["purchases"]: receipt_id, receipt_date, supplier, item_code, item_name, unit, qty, factor, qty_base, unit_cost, amount, cancelled
C["payments"]: order_id, pay_time, pay_date, amount, method ; C["stock"]: item_code, item_name, qty_base (snapshot at stock_date) ; C["lots"]: lot_id, item_code, expiry, qty_in_base, unit_cost_base, ...
C["prices"]: item_code / item_name, date, price ; C["expenses"]: date, amount, category"""


# ------------------------------------------------------------------ shop: metrics over canonical tables
class Unanswerable(Exception):
    """the released data cannot answer (rulebook §7): raise it from a program to answer `unanswerable`."""


def require(cond, why="the data does not record this"):
    if not cond: raise Unanswerable(why)


class Shop:
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
        """orders of period p (by order date). valid=True: not cancelled/returned (§1); None: every status."""
        o = self.O; self.cover(p)
        if self.today is not None: o = o[o["order_date"] <= self.today]
        if p is not None: o = o[p.mask(o["order_date"])]
        if valid is True: o = o[o["valid"]]
        elif valid is False: o = o[~o["valid"]]
        return self._filter(o, **f)

    def lines(self, p=None, valid=True, **f):
        l = self.L; self.cover(p)
        if self.today is not None: l = l[l["order_date"] <= self.today]
        if p is not None: l = l[p.mask(l["order_date"])]
        if valid is True: l = l[l["valid"]]
        elif valid is False: l = l[~l["valid"]]
        if "qty" in l.columns: l = l[l["qty"].fillna(1) > 0]
        return self._filter(l, **f)

    def _filter(self, df, item=None, group=None, customer=None, customer_group=None, staff=None, channel=None, payment=None,
                shift=None, status=None, where=None):
        for v, cols, what in ((item, ("item_code", "item_name"), "items"), (group, ("item_group",), "item groups"),
                              (customer, ("customer_id", "customer_name"), "customers"), (customer_group, ("customer_group",), "customer groups"),
                              (staff, ("staff",), "staff"), (channel, ("channel",), "sales channels"), (payment, ("payment_method",), "payment methods"),
                              (shift, ("order_time",), "the time of day")):
            if v is not None: self._need(df, cols, what)
        if item is not None: df = df[self._match(df, ("item_code", "item_name"), item)]
        if group is not None: df = df[self._match(df, ("item_group",), group)]
        if customer is not None: df = df[self._match(df, ("customer_id", "customer_name"), customer)]
        if customer_group is not None: df = df[self._match(df, ("customer_group",), customer_group)]
        if staff is not None: df = df[self._match(df, ("staff",), staff)]
        if channel is not None: df = df[self._match(df, ("channel",), canon_channel(channel) if isinstance(channel, str) else channel)]
        if payment is not None: df = df[self._match(df, ("payment_method",), canon_payment(payment) if isinstance(payment, str) else payment)]
        if status is not None: df = df[df["status"].eq(status)]
        if shift is not None:
            h = pd.to_datetime(df["order_time"]).dt.hour
            df = df[(h >= 6) & (h <= 13)] if key(shift).startswith("sang") else df[(h >= 14) & (h <= 21)]
        if where is not None: df = df[where(df)]
        return df

    @staticmethod
    def _need(df, cols, what):
        """§7: filtering on information the shop does not record -> Unanswerable (not 0)."""
        if not any(c in df.columns and df[c].notna().any() for c in cols): raise Unanswerable(f"the data does not record {what}")

    @staticmethod
    def _match(df, cols, val, contains=False):
        """rows whose value in any of cols equals val (accent / case / punctuation-insensitive); contains=True also
        accepts values that contain val as whole words (payment 'cod' ~ 'Thu hộ (COD)')."""
        vals = val if isinstance(val, (list, tuple, set)) else [val]
        ks = {key(v) for v in vals}
        def ok(x):
            if x is None or (isinstance(x, float) and math.isnan(x)): return False
            kx = key(x)
            return kx in ks or (contains and any(re.search(r"(^|\s)" + re.escape(k) + r"(\s|$)", kx) for k in ks))
        m = pd.Series(False, index=df.index)
        for c in cols:
            if c in df.columns:
                u = {x: ok(x) for x in df[c].dropna().unique()}
                m |= df[c].map(u).fillna(False).astype(bool)
        return m

    # -- revenue / orders
    def revenue(self, p=None, **f):
        """rulebook revenue (§1) of valid orders in p. Item / group filters use line amounts, others order revenue."""
        if any(k in f for k in ("item", "group")) or f.get("where_lines"):
            f.pop("where_lines", None); return money(self.lines(p, **f)["line_amount"].sum())
        return money(self.orders(p, **f)["revenue"].sum())

    def n_orders(self, p=None, valid=True, **f):
        if any(k in f for k in ("item", "group")): return int(self.lines(p, valid=valid, **f)["order_id"].nunique())
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
        l = self.lines(p, **f); name = "item_name" if "item_name" in l.columns else "item_code"
        if by == "qty": s = l.groupby(name)["qty_base"].sum()
        elif by == "revenue": s = l.groupby(name)["line_amount"].sum()
        elif by == "n_orders": s = l.groupby(name)["order_id"].nunique()
        elif by == "profit": s = (l["line_amount"] - l["qty"] * l["unit_cost"]).groupby(l[name]).sum()
        else: raise ValueError(by)
        return s

    def top_items(self, p, k=1, by="qty", **f):
        s = self.item_sales(p, by=by, **f); r = rank(s, k, codes=self.item_codes())
        return r

    def item_codes(self):
        l = self.L
        if l is None or "item_code" not in l.columns or "item_name" not in l.columns: return None
        return dict(zip(l["item_name"], l["item_code"].astype(str)))

    # -- customers / staff
    def customer_key(self):
        o = self.O
        return "customer_id" if "customer_id" in o.columns and o["customer_id"].notna().any() else "customer_name"

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
        o = self.orders(p, **f); o = o[o["staff"].notna()]
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
        """{order_id: amount paid up to the end of day d} (payments table, else paid_at_purchase)."""
        P = self.C.get("payments")
        if P is not None and len(P):
            P = P[pd.to_datetime(P["pay_date"]).dt.normalize() <= _ts(d)]
            return P.groupby(P["order_id"].astype(str))["amount"].sum()
        o = self.O; return pd.Series(o["paid_at_purchase"].values, index=o["order_id"]) if "paid_at_purchase" in o.columns else pd.Series(dtype=float)

    def debts(self, d):
        """Series order_id -> amount still owed at the end of day d (> 0 only)."""
        o = self._debt_orders(d); paid = self.paid_until(d)
        owed = o["total_payable"].values - o["order_id"].map(paid).fillna(0).values
        s = pd.Series(owed, index=o["order_id"].values); return s[s > 0.5]

    def debt_total(self, d): return money(self.debts(d).sum())

    def debt_by_customer(self, d):
        s = self.debts(d); o = self.O.set_index("order_id"); k = self.customer_key()
        c = o.loc[s.index, k]; s = s[c.notna().values]; return s.groupby(c[c.notna()].values).sum()

    def n_debtors(self, d): return int((self.debt_by_customer(d) > 0.5).sum())

    def top_debtors(self, d, k=3):
        s = self.debt_by_customer(d); r = rank(s, k)
        names = [self.customer_name(c) if self.customer_key() == "customer_id" else c for c in r]
        return {"columns": ["khach_hang", "so_tien_no"], "rows": [[n, money(s[c])] for n, c in zip(names, r)]}

    def debt_aging(self, d, bins=((0, 7), (8, 15), (16, None))):
        s = self.debts(d); age = (_ts(d) - self.O.set_index("order_id").loc[s.index, "order_date"]).dt.days.values
        out = []
        for lo, hi in bins:
            m = (age >= lo) & ((age <= hi) if hi is not None else True); out.append(money(s.values[m].sum()))
        return out

    def debt_older_than(self, d, n):
        """debt of orders placed more than n days before d ('đơn đặt đã hơn n ngày')."""
        s = self.debts(d); age = (_ts(d) - self.O.set_index("order_id").loc[s.index, "order_date"]).dt.days.values
        return money(s.values[age > n].sum())

    def debt_flow(self, p, customer=None):
        """[new debt, repaid] of credit orders in p: new = unpaid part on the purchase day of orders placed in p,
        repaid = payments dated in p made after the purchase day (any credit order)."""
        o = self._debt_orders(p.end)
        if "payment_method" in o.columns: o = o[o["payment_method"].isin(["ghi nợ", "trả góp"])]
        if customer is not None: o = o[self._match(o, ("customer_id", "customer_name"), customer)]
        P = self.C.get("payments"); P = P[P["order_id"].astype(str).isin(set(o["order_id"]))].copy()
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

    def weekday_avg(self, p, weekday, **f):
        """average revenue per weekday (0 = Monday) in p, dividing by every such weekday of the period (days without
        orders included)."""
        days = [d for d in p.dates() if d.weekday() == weekday]
        s = self.daily_revenue(p, **f); return money(s.loc[days].sum() / len(days)) if days else None

    def peak_hour(self, p, **f):
        """starting hour of the one-hour window with the most orders (order creation time)."""
        self._need(self.O, ("order_time",), "the time of day")
        o = self.orders(p, valid=None, **f); h = pd.to_datetime(o["order_time"]).dt.hour.value_counts()
        return int(h.sort_index().idxmax()) if len(h) else None

    def size_qty(self, p, size, **f):
        """quantity of items whose name ends with '(M)' / '(L)' (case-insensitive)."""
        l = self.lines(p, **f); m = l["item_name"].astype(str).str.strip().str.lower().str.endswith(f"({size.lower()})")
        return _num(l.loc[m, "qty_base"].sum())

    def unsold_items(self, p):
        """items sold (valid orders) before p but not in p -> list of names."""
        l = self.lines(None); before = set(l.loc[l["order_date"] < p.start, "item_name"])
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
        """an item given by name or code -> every name / code of that item (catalogue + lines)."""
        ks = {item}
        for df in (self.C.get("items"), self.L):
            if df is not None and {"item_code", "item_name"} <= set(df.columns):
                m = self._match(df, ("item_code", "item_name"), item)
                ks |= set(df.loc[m, "item_code"].astype(str)) | set(df.loc[m, "item_name"].astype(str))
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
        """stock of item (base units) at the end of day d: ledger if any, else snapshot at the end of the data rolled back
        with purchases and sales after d."""
        d = _ts(d); ks = self._item_keys(item)
        lg = self.C.get("ledger")
        if lg is not None and len(lg):
            g = lg[self._match(lg, ("item_code", "item_name"), ks)].copy(); g["time"] = pd.to_datetime(g["time"])
            g = g[g["time"] < d + D].sort_values("time", kind="mergesort")
            return _num(g["stock_after"].iloc[-1]) if len(g) else 0
        st = self.C.get("stock"); require(st is not None and len(st), "no stock records")
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
        it = self.C.get("items"); require(it is not None and "cost" in it.columns and it["cost"].notna().any(), "no catalogue cost")
        tot = 0.0
        for _, r in it.iterrows():
            k = r["item_code"] if "item_code" in it.columns and pd.notna(r.get("item_code")) else r["item_name"]
            try: q = self.stock_at(k, d)
            except Unanswerable: continue
            if q > 0 and pd.notna(r["cost"]): tot += q * float(r["cost"])
        return money(tot)

    # -- lots / expiry (pharmacy)
    def lot_stock(self, d):
        """lots with remaining base quantity at the end of day d (received - sold by non-cancelled orders + returned)."""
        lots = self.C.get("lots"); require(lots is not None and len(lots), "no lot records")
        L = lots.copy(); ls = self.C.get("lot_sales")
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
             "stock_value", "lot_stock", "lots_expiring", "nearest_expiry", "price_on", "price_change_date", "price_effect",
             "gross_profit", "net_profit", "n_delivery_trips", "pct_customers", "factor"]
    return {n: getattr(shop, n) for n in names}


PERIOD_API = {f.__name__: f for f in (day, today_p, yesterday, between, month, month_noyear, quarter, quarter_noyear, year, this_month,
                                      prev_month_same_days, prev_month, same_month_last_year, this_week, last_week, last_n_days,
                                      ytd, until, months_back, month_seq, tet, before_tet, tet_of, lunar_to_solar)}
NUM_API = {f.__name__: f for f in (round_half_up, money, pct, growth, rank, canon_status, map_status, canon_payment, canon_channel, key, require)}
