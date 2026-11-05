"""Industry definitions of the public rulebook (§6) that the training rulebook does not have: phone shops (trade-ins,
instalments, warranty visits, model lines), motorbike repair shops (tickets, jobs written in one cell, vehicles by plate,
mechanics' commission) and mother & baby shops (vouchers / points / shipping, sizes and numbers from the catalogue,
new members). Mixed into dm.Shop; the canonical tables they read are documented in INDUSTRY_CANON / INDUSTRY_API (shown
in the prompts of shops of that industry only). Nothing here is keyed on a business or question id.

Imported by dm.py after its text / number helpers; uses them through the module `dm` at call time."""
import re, math, calendar, functools
import numpy as np
import pandas as pd


def _dm():
    import dm
    return dm


def _isnull(x): return x is None or (isinstance(x, float) and math.isnan(x)) or (isinstance(x, str) and not x.strip())


# ------------------------------------------------------------------ phone shops
_GB = re.compile(r"^(.*?)\s*\d+(?:[.,]\d+)?\s*(?:GB|TB)\b", re.I)


def model_line(name):
    """rulebook 'Cửa hàng điện thoại': dòng máy = the part of the item name before the storage size '…GB' ('iPhone 16 Pro Max
    256GB Đen' -> 'iPhone 16 Pro Max'); a name without '…GB' is a line by itself ('Nokia 105 4G': 4G is the network)."""
    if _isnull(name): return None
    s = re.sub(r"\s+", " ", str(name)).strip(); m = _GB.match(s)
    return m.group(1).strip() if m and m.group(1).strip() else s


def brand_of(name):
    """hãng = the first word of the item name; an iPhone is Apple."""
    if _isnull(name): return None
    w = str(name).strip().split()
    if not w: return None
    k = _dm().key(w[0])
    return "Apple" if k.startswith("iphone") or k in ("apple", "ipad", "macbook") else w[0]


def device_kind(label):
    """'máy mới' | 'máy cũ' | 'phụ kiện' from an item type label of the shop ('Máy mới', 'máy cũ (đã qua sử dụng)', 'PK',
    'Phụ kiện'); None when the label says nothing about it. Also reads a question's words ('mới', 'cũ', 'phụ kiện')."""
    if _isnull(label): return None
    k = _dm().key(label)
    if re.search(r"\b(phu kien|pk|accessor)", k): return "phụ kiện"
    if re.search(r"\b(cu|qua su dung|da su dung|like new|used|2nd|second hand)\b", k): return "máy cũ"
    if re.search(r"\b(moi|new|may|dien thoai|phone)\b", k): return "máy mới"
    return None


def _months_after(d, i, day):
    y, m = d.year + (d.month - 1 + i) // 12, (d.month - 1 + i) % 12 + 1
    return pd.Timestamp(y, m, min(int(day), calendar.monthrange(y, m)[1]))


# ------------------------------------------------------------------ repair shops
PLATE = re.compile(r"\b(\d{2}\s?[A-ZĐ]{1,2}\d?\s?[-.]?\s?\d{3}(?:[.,]?\d{2})?)\s*$", re.I)


def split_vehicle(text):
    """'wave alpha 50K2-994.07' -> ('wave alpha', '50K2-994.07'): a cell holding the vehicle model and its plate (the plate is
    the trailing Vietnamese number plate). No plate found -> (text, None)."""
    if _isnull(text): return None, None
    s = re.sub(r"\s+", " ", str(text)).strip(); m = PLATE.search(s)
    if not m: return s, None
    return s[:m.start()].strip(" -,;:") or None, re.sub(r"\s+", "", m.group(1)).upper()


def plate_key(p):
    return None if _isnull(p) else re.sub(r"[^0-9a-z]", "", _dm().key(p))


def _vtokens(name):
    k = _dm().key(str(name).replace("α", " alpha ").replace("Α", " alpha "))
    return re.sub(r"\s+", " ", k).strip().split()


def _vsame(a, b):
    """two spellings of one vehicle line (rulebook: only case or an abbreviation differs): same letters without spaces
    ('Air Blade' / 'airblade', 'SH Mode' / 'shmode'), one name contained in the other as whole words ('Alpha' / 'Wave Alpha',
    'Vision' / 'Vision 110'), initials ('AB' / 'Air Blade') or a prefix of a one-word name ('Ex' / 'Exciter')."""
    a, b = split_vehicle(a)[0] or a, split_vehicle(b)[0] or b
    ta, tb = _vtokens(a), _vtokens(b)
    if not ta or not tb: return False
    if len(ta) > len(tb): ta, tb = tb, ta
    ja, jb = "".join(ta), "".join(tb)
    if ja == jb: return True
    if any(tb[i:i + len(ta)] == ta for i in range(len(tb) - len(ta) + 1)) and not all(t.isdigit() for t in ta): return True
    letters = [t for t in tb if not t.isdigit()]
    if len(ta) == 1 and len(letters) >= 2 and ja == "".join(t[0] for t in letters): return True
    if len(ta) == 1 and len(ja) >= 2 and len(letters) == 1 and letters[0].startswith(ja) and not ja.isdigit(): return True
    return False


def vehicle_lines(names):
    """{spelling: line label} grouping the spellings of each vehicle line (union of _vsame pairs; the label is the most
    frequent spelling of the group). names: Series / list of spellings (repeats give the frequencies)."""
    raw = pd.Series(list(names)).dropna().astype(str).str.strip(); raw = raw[raw.ne("")]
    model = {x: (split_vehicle(x)[0] or x) for x in raw.unique()}      # a cell that still holds the plate: its model part
    s = raw.map(model); cnt = s.value_counts(); u = list(cnt.index)
    if len(u) > 400:                                                  # not a list of vehicle lines: letters only
        return {x: model[x] for x in model}
    parent = {x: x for x in u}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    joined = {x: "".join(_vtokens(x)) for x in u}
    for i, a in enumerate(u):                                         # 1. same letters ('Air Blade' / 'airblade', 'Wave α')
        for b in u[i + 1:]:
            if joined[a] == joined[b]: parent[find(a)] = find(b)
    for a in sorted(u, key=lambda x: len(joined[x])):                # 2. a shorter spelling joins the ONE line it abbreviates
        roots = {find(b) for b in u if len(joined[b]) > len(joined[a]) and _vsame(a, b)}     # ('Alpha' would not join both
        if len(roots) == 1: parent[find(a)] = roots.pop()                                    # 'Wave Alpha' and 'Alpha X')
    groups = {}
    for x in u: groups.setdefault(find(x), []).append(x)
    lab = {}
    for g in groups.values():
        best = max(g, key=lambda x: (cnt[x], -len(x)))
        for x in g: lab[x] = best
    return {x: lab[model[x]] for x in model}


# canonical jobs of a motorbike repair ticket: (job, accent-free patterns of a task that does it, patterns that rule it out).
# Rulebook: 'Nhớt' written plain is engine oil, not gear oil ('nhớt láp'); 'ruột' written plain is a new inner tube, not a
# patch; 'sửa điện' does not include bulbs and batteries; a warranty ticket is not a ticket doing that job.
JOBS = [
    ("thay nhớt láp", r"\blap\b|nhot hop so|dau hop so|nhot cau|dau cau|nhot so\b", r""),
    ("thay nhớt", r"\bnhot\b|\bdau may\b|\bdau nhot\b|castrol|motul|\bshell\b|\bels\b|\bidemitsu\b|\bliqui|\bmobil\b|\bad nhot\b|\bthay dau\b",
     r"\blap\b|hop so|\bcau\b|nhot so\b|loc nhot"),
    ("vá ruột", r"\bva\b|\bvaruot\b", r""),
    ("thay ruột", r"\bruot\b", r"\bva\b"),
    ("thay vỏ", r"\bvo\b(?! xe khong)|\blop\b", r"\bva\b"),
    ("sửa điện", r"\bdien\b|\bie\b|\bmach\b|\bic\b", r"bong den|\bden\b|\bac quy\b|\bacquy\b|\bbinh\b|\bdien thoai\b"),
    ("thay bóng đèn", r"bong den|\bden\b|den pha|den xi nhan|den hau", r""),
    ("thay ắc quy", r"\bac quy\b|\bacquy\b|\bbinh\b|\bbinh dien\b", r"binh xang|binh nuoc"),
    ("bảo dưỡng", r"\bbd\b|\bbdtq\b|bao duong|bao tri|\btq\b", r""),
    ("rửa xe", r"\brua\b(?! kim)|\brua xe\b|\brx\b", r"kim phun|rua noi"),
    ("vệ sinh nồi", r"\bnoi\b", r""),
    ("vệ sinh kim phun", r"kim phun|\bkp\b", r""),
    ("thay bố thắng", r"\bbo\b|bo thang|ma phanh|\bphanh\b|\bthang\b", r"bo nhong|bo xich"),
    ("chỉnh sên", r"\bsen\b(?! dia)|\bxich\b(?! dia)", r"nhong|\bnsd\b|bo nhong"),
    ("thay nhông sên dĩa", r"\bnsd\b|nhong|sen dia|bo nhong xich", r""),
    ("thay bugi", r"\bbu ?gi\b|\bbuji\b", r""),
    ("thay dây curoa", r"\bcu ?roa\b|\bcuroa\b|\bday cam\b", r""),
    ("thay nước làm mát", r"nuoc mat|nuoc lam mat|nuoc giai nhiet", r""),
    ("thay lọc gió", r"loc gio|\blg\b", r""),
]
_ABBR = [(r"\bvs\b", "ve sinh"), (r"\bbh\b", "bao hanh"), (r"\bt\b", "thay"), (r"\bko\b", "khong")]


@functools.lru_cache(maxsize=65536)
def task_key(t):
    k = _dm().key(t)
    for a, b in _ABBR: k = re.sub(a, b, k)
    return re.sub(r"\s+", " ", k).strip()


def split_jobs(text):
    """the jobs of one ticket's 'what was done' cell ('bdtq + vs nồi', 'thay nhớt, vá ruột và rửa xe') -> list of tasks."""
    if _isnull(text): return []
    return [x.strip() for x in re.split(r"\s*(?:\+|,|;|/|&|\n|\s-\s|\bvà\b|\bva\b(?=\s+(?:thay|rua|ve|vs|lam|sua|bao|chinh|can|tang)\b))\s*",
                                         str(text)) if x and x.strip()]


@functools.lru_cache(maxsize=65536)
def job_of(task):
    """canonical job of one written task (JOBS), or None."""
    k = task_key(task)
    if k.startswith("bao hanh"): k = k[8:].strip()                    # 'bh vá': the job of a warranty visit
    for job, pat, excl in JOBS:
        if re.search(pat, k) and not (excl and re.search(excl, k)): return job
    return None


def job_query(q):
    """the canonical job a question names ('thay dầu láp' -> 'thay nhớt láp', 'vá ruột xe' -> 'vá ruột'); None if unknown."""
    k = task_key(q)
    for job, pat, excl in JOBS:
        if _dm().key(job) == k: return job
    return job_of(q)


def _person_groups(written, full_names):
    """{written staff name: person label}: each spelling in a book that writes given names / nicknames ('Đạt', 'a Đạt',
    'A.Đạt') goes to the ONE full name of the staff list whose last name parts it matches ('Hà Loan Đạt'); without a list,
    spellings with the same name key are one person (label = most frequent spelling)."""
    dm = _dm()
    w = pd.Series(list(written)).dropna().astype(str)
    cnt = w.value_counts(); out = {}
    fulls = [f for f in dict.fromkeys(full_names or []) if not _isnull(f)]
    fk = {f: dm.person_key(f).split() for f in fulls}
    for x in cnt.index:
        kx = dm.person_key(x).split()
        hit = [f for f, kf in fk.items() if kx and (kf == kx or kf[-len(kx):] == kx or kx[-len(kf):] == kf)]
        if len(hit) == 1: out[x] = hit[0]
    rest = [x for x in cnt.index if x not in out]
    by = {}
    for x in rest: by.setdefault(" ".join(dm.person_key(x).split()), []).append(x)
    for g in by.values():
        lab = max(g, key=lambda x: cnt[x])
        for x in g: out[x] = lab
    return out


# ------------------------------------------------------------------ mother & baby
def _is_diaper(group, name=None):
    k = _dm().key(group) if not _isnull(group) else ""
    if re.search(r"\b(bim|ta|ta dan|ta quan|bim ta)\b", k): return True
    if not k and not _isnull(name): return bool(re.match(r"(ta|bim)\b", _dm().key(name)))
    return False


class IndustryMetrics:
    """metrics of the public rulebook's industries; mixed into dm.Shop (self.C, self.O, self.L, orders(), lines(), ...)."""

    # -- phone shops: devices, model lines, brands
    def _kinds(self, l):
        """per line: 'máy mới' / 'máy cũ' / 'phụ kiện' / 'máy' (a device of unknown kind)."""
        g = l["item_group"].map(device_kind) if "item_group" in l.columns else pd.Series(None, index=l.index, dtype=object)
        if "kind" in l.columns: g = g.where(g.notna(), l["kind"].map(device_kind))
        has_imei = "imei" in l.columns and self.L is not None and "imei" in self.L.columns and self.L["imei"].notna().any()
        _dm().require(g.notna().any() or has_imei, "the data does not tell devices from accessories")
        if has_imei and g.isna().any():                                   # README: devices carry an IMEI, accessories do not
            dev = l["imei"].notna() & l["imei"].astype(str).str.strip().ne("") & l["imei"].astype(str).str.lower().ne("nan")
            used = l["item_name"].map(lambda n: bool(re.search(r"\bcu\b", _dm().key(n))) if isinstance(n, str) else False)
            g = g.where(g.notna(), pd.Series(np.where(dev, np.where(used, "máy cũ", "máy mới"), "phụ kiện"), index=l.index))
        return g

    def _devices(self, p, kind=None, brand=None, line=None, **f):
        dm = _dm(); l = self.lines(p, **f); g = self._kinds(l)
        keep = g.notna() & g.ne("phụ kiện")
        if kind is not None:
            kk = device_kind(kind)
            if kk == "phụ kiện": keep = g.eq("phụ kiện")
            elif kk in ("máy mới", "máy cũ"):
                dm.require(g.isin(["máy mới", "máy cũ"]).any(), "the data does not tell new from used devices")
                keep = g.eq(kk)
        l = l[keep]
        if brand is not None: l = l[l["item_name"].map(lambda n: dm.key(brand_of(n) or "")) == dm.key(brand if dm.key(brand) not in ("iphone", "ip") else "Apple")]
        if line is not None:
            lk = re.sub(r"\s+", "", dm.key(model_line(line) or line))
            l = l[l["item_name"].map(lambda n: re.sub(r"\s+", "", dm.key(model_line(n) or ""))) == lk]
        return l

    def devices_sold(self, p, kind=None, brand=None, line=None, **f):
        """number of devices (phones, tablets) sold in p, accessories left out (valid orders). kind='máy mới'|'máy cũ',
        brand='Samsung' (first word of the name; iPhone = Apple), line='iPhone 13' (dòng máy: the name before '…GB')."""
        return _dm()._num(self._devices(p, kind, brand, line, **f)["qty_base"].sum())

    def device_sales(self, p, by="qty", level="line", kind=None, **f):
        """Series {model line | item | brand: measure} over devices sold in p. by='qty'|'revenue'; level='line'|'item'|'brand'."""
        l = self._devices(p, kind, **f)
        lab = l["item_name"].map(model_line) if level == "line" else l["item_name"].map(brand_of) if level == "brand" else l["item_name"]
        return l.groupby(lab)["qty_base" if by == "qty" else "line_amount"].sum()

    def top_lines(self, p, k=1, by="qty", kind=None, level="line", **f):
        """the k model lines (level='item': items, 'brand': brands) with the most devices sold (by='revenue': revenue)."""
        return _dm().rank(self.device_sales(p, by, level, kind, **f), k)

    def device_share(self, p, kind=None, brand=None, line=None, by="revenue", **f):
        """percent of the devices sold in p (revenue, or by='qty' the number of devices) that are of that kind / brand / line;
        accessories are in neither part."""
        col = "qty_base" if by == "qty" else "line_amount"
        return _dm().pct(self._devices(p, kind, brand, line, **f)[col].sum(), self._devices(p, **f)[col].sum())

    def avg_device_price(self, p, kind=None, **f):
        """average price of one device sold in p (device revenue / number of devices; accessories left out)."""
        l = self._devices(p, kind, **f); n = l["qty_base"].sum()
        return _dm().money(l["line_amount"].sum() / n) if n else None

    # -- phone shops: trade-ins, what the customer pays
    def _tradein(self):
        """Series order_id -> trade-in value deducted from the order (orders.tradein, else C['tradeins'] by order_id)."""
        dm = _dm(); o = self.O
        if "tradein" in o.columns and pd.to_numeric(o["tradein"], errors="coerce").fillna(0).gt(0).any():
            return pd.Series(pd.to_numeric(o["tradein"], errors="coerce").fillna(0).values, index=o["order_id"].astype(str).values)
        t = self.C.get("tradeins"); dm.require(t is not None and len(t), "the data does not record trade-ins")
        dm.require("order_id" in t.columns and t["order_id"].astype(str).isin(set(o["order_id"].astype(str))).mean() > 0.5,
                   "trade-ins are not linked to the sales orders")
        return t.groupby(t["order_id"].astype(str))["amount"].sum()

    def tradein_total(self, p, **f):
        """trade-in value (tiền thu máy cũ) deducted from the valid orders of p (rulebook: cancelled / returned orders left out)."""
        o = self.orders(p, **f); return _dm().money(o["order_id"].astype(str).map(self._tradein()).fillna(0).sum())

    def n_tradein_orders(self, p, **f):
        o = self.orders(p, **f); return int((o["order_id"].astype(str).map(self._tradein()).fillna(0) > 0).sum())

    def tradein_share(self, p, **f):
        """percent of the valid orders of p that took a phone in trade."""
        o = self.orders(p, **f); return _dm().pct(self.n_tradein_orders(p, **f), len(o))

    def payable_total(self, p, **f):
        """khách phải trả of the valid orders of p: after the order discount and the trade-in, debt included (orders.total_payable)."""
        dm = _dm(); o = self.orders(p, **f)
        dm.require("total_payable" in o.columns and self.O["total_payable"].notna().any(), "the data does not record what customers pay")
        return dm.money(pd.to_numeric(o["total_payable"], errors="coerce").fillna(0).sum())

    def avg_payable(self, p, **f):
        """average khách phải trả per valid order of p."""
        o = self.orders(p, **f); return _dm().money(self.payable_total(p, **f) / len(o)) if len(o) else None

    # -- phone shops: instalments (rulebook: an instalment order is a credit order; unpaid instalments are debt)
    def _inst(self, d):
        """instalment contracts bought up to day d with: total (n_terms x per_term, else price - prepaid), paid (instalment
        payments up to d), remaining, due_dates."""
        dm = _dm(); I = self.C.get("instalments"); dm.require(I is not None and len(I), "the data does not record instalment contracts")
        I = I.copy(); I["buy_date"] = pd.to_datetime(I["buy_date"], errors="coerce").dt.normalize()
        d = dm._ts(d); I = I[I["buy_date"].notna() & (I["buy_date"] <= d)]
        I["order_id"] = I["order_id"].astype(str).str.strip() if "order_id" in I.columns else I.index.astype(str)
        num = lambda c: pd.to_numeric(I[c], errors="coerce") if c in I.columns else pd.Series(np.nan, index=I.index)
        n, per, price, pre = num("n_terms"), num("per_term"), num("price"), num("prepaid").fillna(0)
        dm.require(n.notna().any() and (per.notna().any() or price.notna().any()),
                   "the instalment contracts lack their number of terms / amount per term")   # (v24 dev_014: raw headers -> 906M overdue)
        I["total"] = (n * per).where(n.notna() & per.notna(), price - pre)
        if per.isna().all(): I["per_term"] = I["total"] / n
        P = self.C.get("instalment_payments"); paid = pd.Series(dtype=float)
        if P is not None and len(P):
            P = P[pd.to_datetime(P["pay_date"], errors="coerce").dt.normalize() <= d]
            paid = pd.to_numeric(P["amount"], errors="coerce").groupby(P["order_id"].astype(str).str.strip()).sum()
        I["paid"] = I["order_id"].map(paid).fillna(0.0)
        I["remaining"] = (I["total"] - I["paid"]).clip(lower=0)
        dd = num("due_day")
        I["due_dates"] = [[_months_after(b, i, (int(x) if not math.isnan(x) else b.day)) for i in range(1, int(k) + 1)] if not math.isnan(k) else []
                          for b, k, x in zip(I["buy_date"], n.fillna(np.nan), dd.fillna(np.nan))]
        return I

    def _inst_who(self, I, customer):
        if customer is None: return I
        return I[self._match_person(I, ("customer_id", "customer_name"), customer)]

    def instalment_remaining(self, d, customer=None):
        """instalment money still unpaid at the end of day d (every term, due or not yet due)."""
        I = self._inst_who(self._inst(d), customer); return _dm().money(I["remaining"].sum())

    def instalment_overdue(self, d, customer=None):
        """unpaid money of the terms due BEFORE day d (rulebook: a term due on the day asked is not overdue yet)."""
        I = self._inst_who(self._inst(d), customer); return _dm().money(self._overdue(I, d).sum())

    def _overdue(self, I, d):
        d = _dm()._ts(d)
        due = pd.Series([sum(1 for x in dl if x < d) for dl in I["due_dates"]], index=I.index)
        return (due * pd.to_numeric(I["per_term"], errors="coerce") - I["paid"]).clip(lower=0).clip(upper=I["remaining"])

    def n_overdue_customers(self, d):
        """customers with at least one overdue, not fully paid instalment term at the end of day d."""
        I = self._inst(d); o = self._overdue(I, d); I = I[o > 0.5]
        k = "customer_id" if "customer_id" in I.columns and I["customer_id"].notna().all() else "customer_name"
        return int(I[k].map(lambda x: _dm().person_key(x) if isinstance(x, str) else x).nunique())

    def terms_due(self, d, customer=None):
        """number of instalment terms due up to day d, the term due ON d included."""
        d = _dm()._ts(d); I = self._inst_who(self._inst(d), customer)
        return int(sum(sum(1 for x in dl if x <= d) for dl in I["due_dates"]))

    def _inst_debts(self, d):
        """[(order_id, owed, customer, buy_date)] of instalment contracts with money unpaid at the end of d."""
        try: I = self._inst(d)
        except _dm().Unanswerable: return pd.DataFrame(columns=["order_id", "owed", "customer", "order_date"])
        I = I[I["remaining"] > 0.5]
        cust = I["customer_id"] if "customer_id" in I.columns and I["customer_id"].notna().any() else I.get("customer_name")
        return pd.DataFrame({"order_id": I["order_id"].values, "owed": I["remaining"].values,
                             "customer": cust.values if cust is not None else None, "order_date": I["buy_date"].values})

    # -- warranty (phone: visits book; repair: warranty tickets)
    def _warranty_book(self, p=None):
        W = self.C.get("warranty")
        if W is None or not len(W): return None
        W = W.copy(); W["date"] = pd.to_datetime(W["date"], errors="coerce").dt.normalize()
        if self.today is not None: W = W[W["date"] <= self.today]
        return W[p.mask(W["date"])] if p is not None else W

    def _is_warranty_ticket(self, o):
        if "warranty" in o.columns and self.O["warranty"].fillna(False).astype(bool).any():   # an all-False column (a mapping
            return o["warranty"].fillna(False).astype(bool)                                  # that never set it) is no column
        note = o["note"].map(lambda x: _dm().key(x) if isinstance(x, str) else "") if "note" in o.columns else pd.Series("", index=o.index)
        return note.str.contains(r"\bbao hanh\b|\bbh\b") & (pd.to_numeric(o["revenue"], errors="coerce").fillna(0) == 0)

    def warranty_visits(self, p, line=None, **f):
        """warranty visits received in p (phone shops: rows of the warranty book; line= a model line) or warranty tickets
        dated in p (repair shops)."""
        dm = _dm(); W = self._warranty_book(p)
        if W is not None:
            if line is not None:
                lk = re.sub(r"\s+", "", dm.key(model_line(line) or line))
                W = W[W["item_name"].map(lambda n: re.sub(r"\s+", "", dm.key(model_line(n) or ""))) == lk]
            return int(len(W))
        o = self.orders(p, **f); w = self._is_warranty_ticket(o)
        dm.require("warranty" in o.columns or w.any(), "the data does not record warranty visits")
        return int(w.sum())

    def warranty_rate(self, p, **f):
        """phone shops: warranty visits received in p / devices sold in p (valid orders); repair shops: warranty tickets dated
        in p / tickets with money in p. Visits for devices sold or repaired before p count (rulebook)."""
        if self._warranty_book() is not None: return _dm().pct(self.warranty_visits(p), self.devices_sold(p))
        return _dm().pct(self.warranty_visits(p, **f), self.n_tickets(p, paid=True, **f))

    def warranty_sales(self, p, by="visits"):
        """Series {model line: warranty visits received in p} (phone shops)."""
        W = self._warranty_book(p); _dm().require(W is not None, "the data does not record warranty visits")
        return W.groupby(W["item_name"].map(model_line)).size()

    def top_warranty_lines(self, p, k=1):
        return _dm().rank(self.warranty_sales(p), k)

    def devices_with_warranty(self, sold, d):
        """devices sold in period `sold` (valid orders, by IMEI) brought back for warranty at least once up to the end of day d."""
        dm = _dm(); W = self._warranty_book(None)
        dm.require(W is not None and "imei" in W.columns, "the data does not record warranty visits by IMEI")
        l = self._devices(sold); dm.require("imei" in l.columns, "the sales book does not record IMEIs")
        ik = lambda s: s.astype(str).str.replace(r"\D", "", regex=True)
        W = W[W["date"] <= dm._ts(d)]
        sold = l["imei"].dropna().astype(str).str.split(r"\s*[/;,|]\s*").explode()     # a line of 2 devices writes 2 IMEIs
        sold = ik(sold[sold.str.strip().ne("")])
        return int(sold.isin(set(ik(W["imei"].dropna()))).sum())

    # -- repair shops: one order = one ticket (phiếu, lượt xe)
    def n_tickets(self, p, paid=None, warranty=None, job=None, vehicle=None, **f):
        """tickets (lượt xe) dated in p, warranty tickets included (rulebook). paid=True: only tickets with money; warranty=True:
        only warranty tickets; job='vá ruột': tickets with money doing that job; vehicle='Air Blade': tickets of that line."""
        o = self._tickets(p, vehicle, **f); w = self._is_warranty_ticket(o)
        if job is not None: return int(self._job_mask(o[~w], job).sum())
        if paid: o = o[~w]
        elif warranty: o = o[w]
        return int(len(o))

    def _tickets(self, p, vehicle=None, **f):
        o = self.orders(p, **f)
        if vehicle is not None:
            _dm().require("vehicle" in self.O.columns, "the data does not record vehicle models")
            lines = vehicle_lines(self.O["vehicle"]); lab = {x: lines.get(x) for x in self.O["vehicle"].dropna().astype(str).str.strip().unique()}
            target = {lab[x] for x in lab if _vsame(x, vehicle)}
            o = o[o["vehicle"].astype(str).str.strip().map(lab).isin(target)]
        return o

    def _job_mask(self, o, job):
        dm = _dm(); dm.require("work" in o.columns and self.O["work"].notna().any(), "the data does not record the jobs of a ticket")
        j = job_query(job)
        if j is not None: return o["work"].map(lambda w: any(job_of(t) == j for t in split_jobs(w)))
        jk = task_key(job)                                               # a job the taxonomy does not know: by its words
        return o["work"].map(lambda w: any(all(x in task_key(t).split() for x in jk.split()) for t in split_jobs(w)))

    def job_tickets(self, p, job, **f):
        """tickets with money in p that did this job ('thay nhớt', 'vá ruột', 'sửa điện', 'thay dầu láp' ...; rulebook
        definitions of plain 'nhớt' / 'ruột' / 'sửa điện'). Warranty tickets do not count."""
        return self.n_tickets(p, job=job, **f)

    def avg_ticket(self, p, **f):
        """revenue of p / tickets with money in p (warranty tickets left out of the count)."""
        n = self.n_tickets(p, paid=True, **f); return _dm().money(self.revenue(p, **f) / n) if n else None

    def _plates(self, o):
        _dm().require("plate" in o.columns and self.O["plate"].notna().any(), "the data does not record number plates")
        return o["plate"].map(plate_key)

    def n_vehicles(self, p, vehicle=None, min_paid=None, **f):
        """distinct vehicles (number plates) with a ticket in p (warranty tickets included); min_paid=2: vehicles with at least
        2 tickets with money in p (xe khách quen, xe tới từ 2 lần)."""
        o = self._tickets(p, vehicle, **f)
        if min_paid:
            o = o[~self._is_warranty_ticket(o)]; c = self._plates(o).value_counts(); return int((c >= min_paid).sum())
        return int(self._plates(o).dropna().nunique())

    def top_plates(self, p, k=1, **f):
        """number plates with the most tickets with money in p (questions 'xe nào' are answered by the plate)."""
        o = self._tickets(p, **f); o = o[~self._is_warranty_ticket(o)]
        s = o.groupby(o["plate"].astype(str).str.strip()).size(); return _dm().rank(s, k)

    def vehicle_sales(self, p, by="revenue", **f):
        """Series {vehicle line: revenue | tickets} over the tickets of p (spellings of one line grouped)."""
        o = self._tickets(p, **f); _dm().require("vehicle" in o.columns, "the data does not record vehicle models")
        lab = vehicle_lines(self.O["vehicle"]); g = o["vehicle"].astype(str).str.strip().map(lab)
        return o.groupby(g)["revenue"].sum() if by == "revenue" else o.groupby(g).size()

    def top_vehicles(self, p, k=1, by="revenue", **f):
        return _dm().rank(self.vehicle_sales(p, by, **f), k)

    def _staff_label(self):
        """{written staff name: person label}; full names from C['staff'] when the shop keeps a staff list."""
        S = self.C.get("staff"); full = list(S["staff"]) if S is not None and "staff" in S.columns else []
        return _person_groups(self.O["staff"], full)

    def staff_stats(self, p, by="revenue", **f):
        """Series {person: measure} over p, every spelling of one person grouped (full name when a staff list exists).
        by='revenue' | 'tickets' (orders / tickets, warranty included) | 'paid_tickets' | 'n_days' (working days: days with at
        least one order or ticket of that person, rulebook 'chăm chỉ nhất')."""
        o = self.orders(p, **f); o = o[o["staff"].notna()]; lab = o["staff"].astype(str).map(self._staff_label()).fillna(o["staff"])
        if by == "revenue": return o.groupby(lab)["revenue"].sum()
        if by in ("tickets", "orders"): return o.groupby(lab).size()
        if by == "paid_tickets": w = self._is_warranty_ticket(o); return o[~w].groupby(lab[~w]).size()
        if by == "n_days": return o.groupby(lab)["order_date"].nunique()
        raise ValueError(by)

    def top_staff(self, p, k=1, by="revenue", **f):
        return _dm().rank(self.staff_stats(p, by, **f), k)

    def commission(self, p, staff=None, **f):
        """each mechanic's commission over p = commission % of the staff list x total money of the tickets they did in p.
        staff=None -> table [tho, hoa_hong] by amount descending; staff='Đạt' -> that person's amount."""
        dm = _dm(); S = self.C.get("staff")
        dm.require(S is not None and len(S) and "commission_pct" in S.columns, "the data does not record commission rates")
        rev = self.staff_stats(p, "revenue", **f)
        pctm = {str(a): float(b) for a, b in zip(S["staff"], pd.to_numeric(S["commission_pct"], errors="coerce")) if pd.notna(b)}
        rate = lambda name: next((v for k_, v in pctm.items() if dm.person_key(k_) == dm.person_key(name)), None)
        amt = pd.Series({n: dm.money(v * rate(n) / 100.0) for n, v in rev.items() if rate(n) is not None}, dtype=float)
        if staff is not None:                                             # a mechanic of the list with no ticket in p earns 0
            lab = _person_groups([staff], list(pctm)).get(staff, staff)
            dm.require(rate(lab) is not None, f"no mechanic named {staff} in the staff list")
            m = [n for n in amt.index if dm.person_key(n) == dm.person_key(lab)]
            return dm.money(amt[m[0]]) if m else 0
        order = dm.rank(amt)
        return {"columns": ["tho", "hoa_hong"], "rows": [[n, dm.money(amt[n])] for n in order]}

    # -- mother & baby: money off / shipping, members, diaper packs
    def _order_money(self, p, col, what, **f):
        dm = _dm(); dm.require(col in self.O.columns and self.O[col].notna().any(), f"the data does not record {what}")
        o = self.orders(p, **f); return dm.money(pd.to_numeric(o[col], errors="coerce").fillna(0).abs().sum())

    def voucher_total(self, p, **f):
        """money off by vouchers on the valid invoices of p (positive; by invoice date)."""
        return self._order_money(p, "order_discount", "vouchers", **f)

    def points_total(self, p, **f):
        """money off paid with loyalty points on the valid invoices of p (positive)."""
        return self._order_money(p, "points_discount", "loyalty points", **f)

    def shipping_total(self, p, **f):
        """shipping fees of the valid invoices of p (positive)."""
        return self._order_money(p, "shipping_fee", "shipping fees", **f)

    def new_members(self, p):
        """members who registered in p (customers with a registration date in p, also those who never bought)."""
        dm = _dm(); K = self.C.get("customers")
        dm.require(K is not None and "registered" in K.columns and K["registered"].notna().any(), "the data does not record registration dates")
        r = pd.to_datetime(K["registered"], errors="coerce").dt.normalize()
        if self.today is not None: r = r[r <= self.today]
        return int(p.mask(r).sum())

    def _brand_col(self, l):
        dm = _dm(); it = self.C.get("items")
        if "brand" in l.columns and l["brand"].notna().any(): return l["brand"]
        if it is not None and {"item_name", "brand"} <= set(it.columns) and it["brand"].notna().any():
            mp = dict(zip(it["item_name"].map(dm.key), it["brand"])); return l["item_name"].map(lambda x: mp.get(dm.key(x)))
        return None

    def _size_col(self, l):
        dm = _dm(); it = self.C.get("items")
        if "size" in l.columns and l["size"].notna().any(): return l["size"]
        if it is not None and {"item_name", "size"} <= set(it.columns) and it["size"].notna().any():
            mp = dict(zip(it["item_name"].map(dm.key), it["size"])); return l["item_name"].map(lambda x: mp.get(dm.key(x)))
        return None

    def _pack_lines(self, p, size=None, brand=None, diapers=None, **f):
        dm = _dm(); l = self.lines(p, **f)
        if diapers:
            d = [_is_diaper(g, n) for g, n in zip(l["item_group"] if "item_group" in l.columns else [None] * len(l), l["item_name"])]
            l = l[np.array(d, dtype=bool)]
        if size is not None:
            col = self._size_col(l)
            sz = lambda v: re.sub(r"^(?:size|so|co)\s+", "", dm.key(v)).strip() if not _isnull(v) else None
            if col is not None: l = l[col.map(sz) == sz(size)]
            else: l = l[l["item_name"].map(lambda n: bool(re.search(r"(?:^|\s)(?:size|so)\s+" + re.escape(sz(size) or "#") + r"(?:\s|$)", dm.key(n))))]
        if brand is not None:
            col = self._brand_col(l); bk = dm.key(brand)
            if col is not None: l = l[col.map(lambda x: dm.key(x) if not _isnull(x) else "") == bk]
            else: l = l[l["item_name"].map(lambda n: bool(re.search(r"(?:^|\s)" + re.escape(bk) + r"(?:\s|$)", dm.key(n))))]
        return l

    def diaper_qty(self, p, size=None, brand=None, **f):
        """diaper packs (bịch / gói, base units) sold in p, of that size and / or brand (catalogue columns; diapers only)."""
        return _dm()._num(self._pack_lines(p, size, brand, diapers=True, **f)["qty_base"].sum())

    def pack_share(self, p, size=None, brand=None, **f):
        """percent of the diaper packs (bịch / gói, base units) sold in p that are of that size and / or brand; the whole is every
        diaper item of the period (rulebook 'Mẹ và bé'; size and brand from the catalogue columns)."""
        whole = self._pack_lines(p, diapers=True, **f)["qty_base"].sum()
        return _dm().pct(self._pack_lines(p, size, brand, diapers=True, **f)["qty_base"].sum(), whole)

    def brand_sales(self, p, by="qty", size=None, **f):
        """Series {brand: quantity (base units) | revenue} of the lines of p (group= / size= filters; brand from the catalogue)."""
        l = self._pack_lines(p, size=size, **f); col = self._brand_col(l)
        _dm().require(col is not None, "the catalogue does not record brands")
        return l.groupby(col)["qty_base" if by == "qty" else "line_amount"].sum()

    def top_brands(self, p, k=1, by="qty", size=None, **f):
        return _dm().rank(self.brand_sales(p, by, size, **f), k)


INDUSTRY_NAMES = ["devices_sold", "device_sales", "top_lines", "device_share", "avg_device_price", "tradein_total", "n_tradein_orders",
                  "tradein_share", "payable_total", "avg_payable", "instalment_remaining", "instalment_overdue", "n_overdue_customers",
                  "terms_due", "warranty_visits", "warranty_rate", "warranty_sales", "top_warranty_lines", "devices_with_warranty",
                  "n_tickets", "job_tickets", "avg_ticket", "n_vehicles", "top_plates", "vehicle_sales", "top_vehicles", "staff_stats",
                  "top_staff", "commission", "voucher_total", "points_total", "shipping_total", "new_members", "pack_share", "diaper_qty",
                  "brand_sales", "top_brands"]
HELPERS = {"model_line": model_line, "brand_of": brand_of, "split_vehicle": split_vehicle, "split_jobs": split_jobs, "job_of": job_of}

# shown in the mapping prompt (CANONICAL TABLES) / answer prompt (TOOLKIT) of a shop whose README names that rulebook industry
INDUSTRY_CANON = {
    "cửa hàng điện thoại": """Tables of a phone shop (map every one the files have):
- C["items"].item_group = the shop's item type ('máy mới' | 'máy cũ' | 'phụ kiện'); lines.item_name = the FULL item name (a lookup
  sheet of short names -> full names: map every short name); lines.imei.
- orders.total_payable = khách phải trả of the order (after the order discount and the trade-in, debt included); orders.order_discount
  (bớt); orders.tradein = trade-in value deducted from the order (or C["tradeins"]: order_id, date, amount, item_name, imei).
- C["instalments"] (sổ góp, one row per contract): order_id (the sales order), buy_date, customer_name, customer_id, item_name,
  price, prepaid, n_terms, per_term, due_day (day of month of each term; the first term is due the month after the purchase);
  C["instalment_payments"]: order_id, pay_date, amount (every instalment paid).
- C["warranty"] (one row per warranty visit): date (received), item_name (model line), imei, buy_date, customer_name.
- Debt book: C["payments"] (order_id, pay_date, amount) so that the toolkit's debts work; an instalment order is NOT also a debt of the
  debt book (the toolkit adds unpaid instalments to the debts itself).""",
    "sửa xe": """Tables of a repair shop (one order = one ticket / lượt xe; every row of the repair book is one ticket: dedupe=False):
- one line per ticket: lines with order_id (one per row), order_date, staff (the mechanic as written), line_amount = the ticket's
  money, item_name = "phiếu sửa xe", qty = 1, plate and vehicle (split_vehicle(cell) -> (model, plate) when one cell holds both),
  work = the jobs cell as written, warranty = True for a warranty ticket (note / money 0 as the README says), note; then
  orders, lines = finalize(lines, dedupe=False) (the order-level columns go to orders).
- C["staff"] (staff list): staff (full name), commission_pct, start_date.""",
    "mẹ và bé": """Tables of a mother & baby shop:
- orders.order_discount = voucher money off (positive), orders.points_discount = money off paid with points (positive),
  orders.shipping_fee (positive); a voucher / points / shipping row written as a line of the invoice goes to these order columns,
  not to the item lines.
- C["items"]: item_name, item_group, brand (hãng), size (the catalogue's Số / Size column: 'Số 4', 'XL'; empty when blank).
- C["customers"] (member list): customer_id, customer_name, registered (registration date), points.""",
}

INDUSTRY_API = {
    "cửa hàng điện thoại": """Phone-shop toolkit (rulebook 'Cửa hàng điện thoại'; accessories are never devices):
- devices_sold(p, kind=None|"máy mới"|"máy cũ", brand=None, line=None), device_sales(p, by="qty"|"revenue", level="line"|"item"|"brand",
  kind=None) -> Series, top_lines(p, k, by="qty", kind=None, level="line"), device_share(p, kind=, brand=, line=, by="revenue"|"qty")
  (% of devices only), avg_device_price(p, kind=None); model_line(name), brand_of(name).
- tradein_total(p), n_tradein_orders(p), tradein_share(p) (% of valid orders), payable_total(p), avg_payable(p) (khách phải trả).
- Instalments at the end of day d: instalment_remaining(d, customer=None) (every unpaid term), instalment_overdue(d, customer=None)
  (terms due before d), n_overdue_customers(d), terms_due(d, customer) (due up to d, d included). debt_total / n_debtors /
  top_debtors / debt_aging / debt_flow include unpaid instalments (rulebook).
- Warranty: warranty_visits(p, line=None), warranty_rate(p) (visits in p / devices sold in p), top_warranty_lines(p, k),
  devices_with_warranty(sold_period, d).""",
    "sửa xe": """Repair-shop toolkit (rulebook 'Sửa xe'; orders are tickets):
- n_tickets(p, paid=None, warranty=None, job=None, vehicle=None, staff=None) (every ticket, warranty included, unless paid=True /
  warranty=True), job_tickets(p, "thay nhớt"|"vá ruột"|"sửa điện"|..., staff=None) (tickets with money doing that job),
  avg_ticket(p) (revenue / tickets with money), warranty_visits(p), warranty_rate(p) (warranty tickets / tickets with money).
- n_vehicles(p, vehicle=None, min_paid=None) (distinct plates; min_paid=2: vehicles with >= 2 tickets with money), top_plates(p, k),
  vehicle_sales(p, by="revenue"|"tickets") -> Series, top_vehicles(p, k, by="revenue").
- staff_stats(p, by="revenue"|"tickets"|"paid_tickets"|"n_days") -> Series by person (every spelling of a mechanic grouped),
  top_staff(p, k, by=...), commission(p, staff=None) (table [tho, hoa_hong], or one mechanic's amount).""",
    "mẹ và bé": """Mother & baby toolkit (rulebook 'Mẹ và bé'):
- voucher_total(p), points_total(p), shipping_total(p) (positive, valid invoices by invoice date), new_members(p) (registered in p).
- size_qty(p, "XL"|"Số 4", group=...) (catalogue Số / Size column), diaper_qty(p, size=None, brand=None) (diaper packs sold),
  pack_share(p, size=None, brand=None) (% of every diaper pack sold),
  brand_sales(p, by="qty"|"revenue", size=None, group=...) -> Series, top_brands(p, k, by="qty", size=None, group=...).""",
}


def industry_key(title):
    """the INDUSTRY_* key of a rulebook industry title ('Cửa hàng điện thoại', 'Sửa xe', 'Mẹ và bé'), or None."""
    if not title: return None
    k = _dm().key(title)
    for name in INDUSTRY_CANON:
        if _dm().key(name) == k or _dm().key(name) in k: return name
    return None


# ------------------------------------------------------------------ industry from the data (when the README does not name it)
def _texty(s): return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)

_MB = re.compile(r"\b(ta (?:dan|quan)|bim|sua bot|binh sua|an dam|so sinh|nui ti|khan uot|tre em|cho be)\b")


def detect(biz):
    """the rulebook industry title ('Cửa hàng điện thoại' | 'Sửa xe' | 'Mẹ và bé') a shop's DATA shows, for READMEs that do
    not name it (public dev_010: 'Dữ liệu bán hàng của tiệm'); None otherwise. Conservative: no training shop is detected.
    - phone shop: a column IMEI and device names with a storage size ('128GB');
    - repair shop: a column of jobs ('Làm gì', 'Công việc', 'Nội dung sửa') and number plates in most rows of some column;
    - mother & baby: at least a third of the distinct item names are diapers, formula, baby bottles, weaning food."""
    dm = _dm()
    cols = {t: [dm.key(c) for c in df.columns] for t, df in biz.tables.items()}
    for t, df in biz.tables.items():
        if any(c == "imei" or c.startswith("imei ") for c in cols[t]):
            txt = [c for c in df.columns if _texty(df[c])]
            if not txt: continue
            names = pd.Series(df[txt].astype(str).values.ravel()[:20000]).astype(str)
            if names.str.contains(r"\d+\s*GB\b", case=False, regex=True).mean() > 0.01: return "Cửa hàng điện thoại"
    for t, df in biz.tables.items():
        if any(re.search(r"\b(lam gi|cong viec|noi dung sua|hang muc sua|viec lam)\b", c) for c in cols[t]):
            for c in df.columns:
                v = df[c].dropna().astype(str).head(3000)
                if len(v) > 20 and v.map(lambda x: bool(PLATE.search(x))).mean() > 0.6: return "Sửa xe"
    names = set()
    for t, df in biz.tables.items():
        for c in df.columns:
            if re.search(r"\b(ten hang|ten san pham|san pham|hang hoa|mat hang|ten_hang|item)\b", dm.key(c).replace("_", " ")) and _texty(df[c]):
                names |= set(df[c].dropna().astype(str).str.strip().unique()[:3000])
    if len(names) >= 10 and sum(1 for n in names if _MB.search(dm.key(n))) / len(names) >= 1 / 3: return "Mẹ và bé"
    return None
