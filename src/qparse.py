#!/usr/bin/env python3
"""Rule-based reading of Vietnamese period expressions and of the common question templates.

parse_period(text, today) -> dm.Period | None     ("tháng 6/2025", "hqua", "tuần rồi", "quý 4/2025", "7 ngày gần nhất", ...)
parse(question, today)   -> (template, kwargs) | None

Used (1) by tests/eval_lib.py to check the canonical tables + metrics against the train answers without an LLM, and
(2) at run time as one more voter / a hint for the LLM. Only generic language patterns, nothing keyed on ids."""
import re
import pandas as pd
import dm

NUMW = {"mot": 1, "hai": 2, "ba": 3, "bon": 4, "tu": 4, "nam": 5, "sau": 6, "bay": 7, "tam": 8, "chin": 9, "muoi": 10}
MONTHW = {"gieng": 1, "mot": 1, "hai": 2, "ba": 3, "tu": 4, "bon": 4, "nam": 5, "sau": 6, "bay": 7, "tam": 8, "chin": 9, "muoi": 10,
          "muoi mot": 11, "muoi hai": 12, "chap": 12}


def K(s):
    """accent-free, lowercase, punctuation-light key that keeps / - : . , between digits."""
    s = dm.strip_accents(str(s)).lower()
    s = re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", s)                       # 15.000.000 -> 15000000
    s = re.sub(r"[^\w\s/:\-%.,]", " ", s)
    s = re.sub(r"(?<!\d)[.,]|[.,](?!\d)", " ", s)
    return re.sub(r"\s+", " ", s).strip()


Y = r"(20\d\d|\d\d)"
MON = r"(?:thang|thg|th|t)\s*"


def _y(s): y = int(s); return y + 2000 if y < 100 else y


def parse_period(text, today):
    """first period expression of the text (None when there is none)."""
    t = K(text); today = pd.Timestamp(today).normalize()
    m = re.search(r"tu (?:ngay )?(\d{1,2})/(\d{1,2})/" + Y + r" (?:den|toi|-) (?:het )?(?:ngay )?(\d{1,2})/(\d{1,2})/" + Y, t)
    if m:
        a = pd.Timestamp(_y(m.group(3)), int(m.group(2)), int(m.group(1))); b = pd.Timestamp(_y(m.group(6)), int(m.group(5)), int(m.group(4)))
        return dm.between(a, b)
    if re.search(r"\btet\b", t):
        m = re.search(r"tet (?:nam |am lich )?(20\d\d)", t)
        if m: return dm.tet(int(m.group(1)))
        if re.search(r"tet nam (truoc|ngoai|roi)", t): return dm.tet(today.year - 1)
        return dm.tet_of(today)
    if re.search(r"(dau thang nay|tu dau thang|tu mung 1|thang nay|thang ni|th nay)", t): return dm.this_month(today)
    if re.search(r"(tu dau nam|dau nam (toi|den)|nam nay tinh tu)", t): return dm.ytd(today)
    m = re.search(r"(\d+) ngay (gan nhat|qua|vua qua|gan day|lien tiep gan nhat)", t)
    if m: return dm.last_n_days(int(m.group(1)), today)
    if re.search(r"tuan (roi|truoc|vua roi|qua)", t): return dm.last_week(today)
    if re.search(r"tuan nay", t): return dm.this_week(today)
    m = re.search(r"quy (\d)\s*(?:/|-|nam )\s*" + Y, t)
    if m: return dm.quarter(_y(m.group(2)), int(m.group(1)))
    m = re.search(r"quy (\d)\b", t)
    if m: return dm.quarter_noyear(int(m.group(1)), today)
    m = re.search(r"(?:ngay )?(\d{1,2})/(\d{1,2})/(20\d\d)", t)
    if m: return dm.day(pd.Timestamp(int(m.group(3)), int(m.group(2)), int(m.group(1))))
    m = re.search(r"\b" + MON + r"(\d{1,2})\s*(?:/|-|nam )\s*" + Y + r"\b", t) or re.search(r"(?<![\d/])(\d{1,2})[/-](20\d\d|\d\d)\b(?![/\d])", t)
    if m: return dm.month(_y(m.group(2)), int(m.group(1)))
    m = re.search(r"\b(?:thang|thg|th)\s*(\d{1,2})\b(?!/)", t)
    if m: return dm.month_noyear(int(m.group(1)), today)
    if re.search(r"(hom qua|hqua|bua qua|ngay qua\b|hom truoc)", t): return dm.yesterday(today)
    if re.search(r"(hom nay|hnay|bua nay|ngay nay|trong ngay)", t): return dm.today_p(today)
    return None


def all_months(text, today):
    """every 'tháng M/Y' in the text, in order (for 3-month trend / list questions)."""
    t = K(text); out = []
    for m in re.finditer(r"(?:" + MON + r")?(\d{1,2})\s*(?:/|-)\s*(20\d\d)", t): out.append(dm.month(int(m.group(2)), int(m.group(1))))
    return out


def number_after(t, pat):
    m = re.search(pat, t); return float(m.group(1)) if m else None


class Vocab:
    """entity names of one shop (keys -> display value) for linking question words to data values."""
    def __init__(self, shop):
        O, L = shop.O, shop.L
        def vals(df, c): return [] if df is None or c not in df.columns else [x for x in df[c].dropna().unique() if str(x).strip()]
        self.items = {dm.key(x): x for c in ("item_name", "item_code") for x in vals(L, c)}
        it = shop.C.get("items")
        if it is not None:
            for c in ("item_name", "item_code"): self.items.update({dm.key(x): x for x in vals(it, c)})
        self.groups = {dm.key(x): x for x in vals(L, "item_group")}
        self.cgroups = {dm.key(x): x for x in vals(O, "customer_group")}
        self.staff = {dm.key(x): x for x in vals(O, "staff")}
        self.customers = {dm.key(x): x for c in ("customer_name", "customer_id") for x in vals(O, c)}

    @staticmethod
    def find(t, table, min_len=3):
        """longest table key occurring as whole words in t (accent-free key of the question)."""
        best = None
        for k, v in table.items():
            if len(k) >= min_len and re.search(r"(^|\s)" + re.escape(k) + r"(\s|$)", t) and (best is None or len(k) > len(best[0])): best = (k, v)
        return best[1] if best else None


PAYMENTS = [("cod", "cod"), ("thu ho", "cod"), ("tien mat", "tiền mặt"), ("chuyen khoan", "chuyển khoản"), ("vi dien tu", "ví điện tử"),
            ("ghi no", "ghi nợ"), ("the", "thẻ")]


def parse(q, today, shop=None):
    """(template, kwargs) for the common templates; None when unsure."""
    t = K(q); tk = dm.key(q); p = parse_period(q, today)
    def has(*pats): return any(re.search(x, t) for x in pats)
    f = {}
    if shop is not None:
        V = Vocab(shop)
        it = V.find(tk, V.items)
        if it is not None: f["item"] = it
        m = re.search(r"nhom (?:hang )?(.+?)(?: ban| chiem| trong| thang| tuan| co | dc| duoc|$)", tk)
        if m and "item" not in f:
            g = V.find(m.group(1), V.groups) or V.find(tk, V.groups)
            cg = V.find(m.group(1), V.cgroups)
            if has(r"khach") and cg is not None: f["customer_group"] = cg
            elif g is not None: f["group"] = g
        m = re.search(r"nhan vien (?:ma )?(\w+)", tk)
        if m and V.find(m.group(1), V.staff): f["staff"] = V.find(m.group(1), V.staff)
    for kw_, lab in PAYMENTS:
        if re.search(r"\b" + kw_ + r"\b", t) and has(r"tra bang|thanh toan|tra qua|bang|qua"): f["payment"] = lab; break
    if has(r"ca sang"): f["shift"] = "sáng"
    elif has(r"ca chieu"): f["shift"] = "chiều"
    rev = has(r"doanh thu|dthu|revenue|ban duoc (bao nhieu|bnhiu|bn|nhieu|nhiu|may|total) ?tien|tong cong bao nhieu tien|ban dc .*tien|ban duoc .*tien|thu ve|tien hang|tien ban|bao nhieu tien|bnhiu tien|bn tien")
    if has(r"\bhuy\b") and has(r"ti le|ty le|%|phan tram") and p: return "cancel_rate", {"p": p}
    if has(r"trung binh|binh quan") and has(r"don") and rev and p and not has(r"nhap|ngay thu|moi ngay"):
        return "aov", {"p": p, **f}
    if has(r"ngay nao") and p and has(r"doanh thu|dthu|nhieu tien|nhiu tien|revenue|cao nhat"): return "best_day", {"p": p}
    if has(r"co may ngay|bao nhieu ngay|bnhiu ngay|bn ngay|may bua|co may hom") and p:
        x = number_after(t, r"(\d{4,})\s*(?:dong|d\b|vnd)?\s*tro len")
        if x: return "n_days_rev_ge", {"p": p, "x": x}
    if has(r"don") and has(r"tro len") and p and has(r"doanh thu|dthu|tien hang|gia tri"):
        x = number_after(t, r"(\d{4,})\s*(?:dong|d\b|vnd|đ)?\s*tro len")
        if x: return "n_orders_rev_ge", {"p": p, "x": x}
    if has(r"tang hay giam|thay doi|tang truong|len hay xuong|cao hay thap|hon hay kem") and rev and not f:
        if has(r"cung ky thang (truoc|roi|trc|lien truoc)") or (has(r"dau thang|thang nay") and has(r"cung ky")): return "growth_mtd", {}
        ms = all_months(q, today)
        if len(ms) == 2: return "growth", {"a": ms[0], "b": ms[1]}
        if p is not None and has(r"thang (truoc|lien truoc|lien trc|trc|ngay truoc)"): return "growth_prev", {"p": p}
        if p is not None and has(r"cung (ky|thang) nam (ngoai|truoc|roi)"): return "growth_yoy", {"p": p}
        return None
    if rev and p and not has(r"khach|hoat chat|ke don|nhap|lai|loi|chi phi|tra no|no\b|kenh|tai quay|online|giao|size|ngay nao|thang nao|tung thang|moi thang|lan luot|\bva\b|so voi|tang|giam|top|nhat|trung binh|binh quan|%|phan tram|ti trong|ty trong|chiem"):
        if not f and has(r"nhom|mat hang|san pham|hang nao|\bsp\b|thanh toan|tra bang"): return None
        return "revenue", {"p": p, **f}
    if has(r"so don|may don|bao nhieu don|bnhiu don|bn don|so bill|bao nhieu bill|may bill|bao nhieu don hang|nhieu don|tong don|ddown|don hang") and p \
            and not has(r"huy|tra|khach|mat hang|san pham|nhom|tro len|nhan vien|giao|%|ti le|phan tram|trung binh|co mua|co ban|kenh|online|tai quay|thanh toan"):
        return "n_orders", {"p": p, **f}
    return None


def run_template(shop, tpl, kw, today):
    """answer dict of a parsed template on a dm.Shop."""
    f = {k: v for k, v in kw.items() if k not in ("p", "x", "a", "b")}
    if tpl == "revenue": return {"type": "number", "answer": shop.revenue(kw["p"], **f)}
    if tpl == "n_orders": return {"type": "number", "answer": shop.n_orders(kw["p"], **f)}
    if tpl == "cancel_rate": return {"type": "percent", "answer": shop.cancel_rate(kw["p"])}
    if tpl == "aov": return {"type": "number", "answer": shop.aov(kw["p"], **f)}
    if tpl == "best_day": return {"type": "date", "answer": shop.best_day(kw["p"])}
    if tpl == "n_days_rev_ge": return {"type": "number", "answer": shop.n_days_revenue_at_least(kw["p"], kw["x"])}
    if tpl == "n_orders_rev_ge": return {"type": "number", "answer": shop.n_orders_revenue_at_least(kw["p"], kw["x"])}
    if tpl == "growth_mtd": return {"type": "percent", "answer": dm.growth(shop.revenue(dm.this_month(today)), shop.revenue(dm.prev_month_same_days(today)))}
    if tpl == "growth": return {"type": "percent", "answer": dm.growth(shop.revenue(kw["a"]), shop.revenue(kw["b"]))}
    if tpl == "growth_prev": return {"type": "percent", "answer": dm.growth(shop.revenue(kw["p"]), shop.revenue(dm.prev_month(kw["p"])))}
    if tpl == "growth_yoy": return {"type": "percent", "answer": dm.growth(shop.revenue(kw["p"]), shop.revenue(dm.same_month_last_year(kw["p"])))}
    raise KeyError(tpl)
