#!/usr/bin/env python3
"""Rule-based reading of Vietnamese period expressions and of the common question templates -> a short program over
the dm toolkit (the same API the LLM-written programs use).

parse_period(text, today) -> (dm.Period, code) | (None, None)   ("tháng 6/2025", "hqua", "tuần rồi", "quý 4/2025", ...)
parse(question, today, shop) -> (template, code) | None          code assigns ANSWER, e.g.
    ANSWER = {"type": "number", "answer": revenue(month(2025, 6), item="Coca Cola 330ml")}

Used (1) by tests/eval_lib.py to check the canonical tables + metrics against the train answers without an LLM,
(2) to build few-shot (question -> program) examples from train, and (3) at run time as one more voter.
Only generic language patterns; nothing is keyed on business or question ids."""
import re
import pandas as pd
import dm


def K(s):
    """accent-free, lowercase key that keeps / - : . , % between digits; '15.000.000' -> '15000000'."""
    s = dm.strip_accents(str(s)).lower().replace("≥", " tro len ").replace(">=", " tro len ")
    s = re.sub(r"(?<=\d)[.,](?=\d{3}(?!\d))", "", s); s = re.sub(r"(?<=\d)[.,](?=\d{3}(?!\d))", "", s)
    s = re.sub(r"[^\w\s/:\-%.,]", " ", s)
    s = re.sub(r"(?<!\d)[.,]|[.,](?!\d)", " ", s)
    return re.sub(r"\s+", " ", s).strip()


NEG = {"k": "khong", "ko": "khong", "hok": "khong", "hong": "khong", "kh": "khong", "kg": "khong", "khg": "khong", "hem": "khong", "hk": "khong"}


def untelex(t):
    """undo Telex typing left in accent-free text ('bawngf' -> 'bang', 'toonf' -> 'ton', 'dduwowcj' -> 'duoc') and spell
    colloquial negations as 'khong'. Vietnamese syllables never end in f / s / r / x / j, so those are tone keys."""
    out = []
    for w in t.split():
        if re.fullmatch(r"[a-z]+", w) and len(w) > 1:
            v = w.replace("dd", "d").replace("aa", "a").replace("ee", "e").replace("oo", "o")
            v = re.sub(r"(?<=[aou])w", "", v).replace("uow", "uo")
            if len(v) > 2 and v[-1] in "fsrxj" and re.search(r"[aeiouy]", v[:-1]): v = v[:-1]
            w = v
        out.append(NEG.get(w, w))
    return " ".join(out)


Y = r"(20\d\d|\d\d)"
MON = r"(?:thang|thg|th|t)\s*"


def _y(s): y = int(s); return y + 2000 if y < 100 else y


def _date(d, m, y): return pd.Timestamp(_y(y), int(m), int(d))


def _dc(ts): return f'"{pd.Timestamp(ts).strftime("%Y-%m-%d")}"'


def parse_period(text, today):
    """first period expression of the text -> (Period, code) or (None, None)."""
    t = K(text); today = pd.Timestamp(today).normalize()
    m = re.search(r"tu (?:ngay )?(\d{1,2})/(\d{1,2})/" + Y + r" (?:den|toi|-|cho den) (?:het )?(?:ngay )?(\d{1,2})/(\d{1,2})/" + Y, t)
    if m:
        a, b = _date(*m.group(1, 2, 3)), _date(*m.group(4, 5, 6)); return dm.between(a, b), f"between({_dc(a)}, {_dc(b)})"
    m = re.search(r"(?:tu |giai doan )?(\d{1,2})/(\d{1,2})/" + Y + r" (?:den|toi|-) (?:\w+ )?(?:ngay )?(\d{1,2})/(\d{1,2})/" + Y, t)
    if m:
        a, b = _date(*m.group(1, 2, 3)), _date(*m.group(4, 5, 6)); return dm.between(a, b), f"between({_dc(a)}, {_dc(b)})"
    if re.search(r"\btet\b", t):
        m = re.search(r"tet (?:nam |am lich |nguyen dan )?(20\d\d)", t)
        if m: return dm.tet(int(m.group(1))), f"tet({int(m.group(1))})"
        if re.search(r"tet nam (truoc|ngoai|roi)", t): return dm.tet(today.year - 1), f"tet({today.year - 1})"
        return dm.tet_of(today), "tet_of(TODAY)"
    if re.search(r"(dau thang nay|tu dau thang|tu mung 1|thang nay|thang ni|th nay|thang hien tai)", t): return dm.this_month(today), "this_month(TODAY)"
    if re.search(r"(tu dau nam|dau nam (toi|den)|nam nay tinh tu|tu ngay 1/1|tu 1/1|tu mung 1 tet duong)", t): return dm.ytd(today), "ytd(TODAY)"
    m = re.search(r"(\d+) ngay (gan nhat|qua|vua qua|gan day|lien tiep gan nhat|cuoi|gan nhut|vua roi)", t)
    if m: return dm.last_n_days(int(m.group(1)), today), f"last_n_days({int(m.group(1))}, TODAY)"
    if re.search(r"tuan (roi|truoc|vua roi|vua qua|qua|trc)", t): return dm.last_week(today), "last_week(TODAY)"
    if re.search(r"tuan nay", t): return dm.this_week(today), "this_week(TODAY)"
    m = re.search(r"quy (\d)\s*(?:/|-|nam |cua nam )\s*" + Y, t)
    if m: return dm.quarter(_y(m.group(2)), int(m.group(1))), f"quarter({_y(m.group(2))}, {int(m.group(1))})"
    m = re.search(r"quy (\d)\b", t)
    if m: return dm.quarter_noyear(int(m.group(1)), today), f"quarter_noyear({int(m.group(1))}, TODAY)"
    m = re.search(r"(?:ngay )?(\d{1,2})/(\d{1,2})/(20\d\d)", t)
    if m: d = _date(*m.group(1, 2, 3)); return dm.day(d), f"day({_dc(d)})"
    m = re.search(r"\b" + MON + r"(\d{1,2})\s*(?:/|-|nam |cua nam )\s*" + Y + r"\b", t) or re.search(r"(?<![\d/])(\d{1,2})[/-](20\d\d|\d\d)\b(?![/\d])", t)
    if m: return dm.month(_y(m.group(2)), int(m.group(1))), f"month({_y(m.group(2))}, {int(m.group(1))})"
    m = re.search(r"\b(?:thang|thg|th)\s*(\d{1,2})\b(?!/)", t)
    if m: return dm.month_noyear(int(m.group(1)), today), f"month_noyear({int(m.group(1))}, TODAY)"
    if re.search(r"(hom qua|hqua|bua qua|ngay qua\b|hom truoc|ngay hom qua)", t): return dm.yesterday(today), "yesterday(TODAY)"
    if re.search(r"(hom nay|hnay|bua nay|ngay nay|trong ngay|cuoi ngay)", t): return dm.today_p(today), "today_p(TODAY)"
    return None, None


def all_months(text):
    """every explicit month 'M/YYYY' of the text in order -> list of code strings."""
    t = K(text); out = []
    for m in re.finditer(r"(?<![\d/])(\d{1,2})\s*[/-]\s*(20\d\d)\b", t): out.append(f"month({int(m.group(2))}, {int(m.group(1))})")
    return out


def until_day(text, today):
    """'đến hết ngày X' / 'cuối ngày hôm nay' -> code of the day."""
    t = K(text); m = re.search(r"(\d{1,2})/(\d{1,2})/(20\d\d)", t)
    return _dc(_date(*m.group(1, 2, 3))) if m else "TODAY"


def money_in(t, after=r""):
    """first money amount of the key text (15000000, '15 trieu', '55k')."""
    m = re.search(after + r"(\d+(?:[.,]\d+)?)\s*(trieu|tr|k|nghin|ngan)?\s*(?:dong|d\b|vnd)?", t)
    if not m: return None
    v = float(m.group(1).replace(",", ".")); u = m.group(2) or ""
    return v * (1e6 if u.startswith("tr") else 1e3 if u in ("k", "nghin", "ngan") else 1)


class Vocab:
    """entity names of one shop (key -> display value) for linking question words to data values."""
    def __init__(self, shop):
        O, L = shop.O, shop.L
        def vals(df, c): return [] if df is None or c not in df.columns else [x for x in df[c].dropna().unique() if str(x).strip()]
        self.items = {dm.key(x): x for c in ("item_name",) for x in vals(L, c)}
        it = shop.C.get("items")
        if it is not None: self.items.update({dm.key(x): x for x in vals(it, "item_name") if dm.key(x) not in self.items})
        self.codes = {dm.key(x): x for x in vals(L, "item_code")}
        self.groups = {dm.key(x): x for x in vals(L, "item_group")}
        self.cgroups = {dm.key(x): x for x in vals(O, "customer_group")}
        self.staff = {dm.key(x): x for x in vals(O, "staff")}
        self.customers = {dm.key(x): x for x in vals(O, "customer_name")}
        self.ingredients = {dm.key(x): x for x in vals(L, "active_ingredient")}

    @staticmethod
    def find(t, table, min_len=3):
        """longest table key occurring as whole words in t (accent-free key of the question)."""
        best = None
        for k, v in table.items():
            if len(k) >= min_len and (best is None or len(k) > len(best[0])) and re.search(r"(^|\s)" + re.escape(k) + r"(\s|$)", t): best = (k, v)
        return best[1] if best else None


PAY_WORDS = [("cod", "cod"), ("thu ho", "cod"), ("tien mat", "tiền mặt"), ("chuyen khoan", "chuyển khoản"), ("vi dien tu", "ví điện tử"),
             ("ghi no", "ghi nợ"), ("the ngan hang", "thẻ"), ("quet the", "thẻ")]
CHAN_WORDS = [("zalo", "zalo"), ("dien thoai", "điện thoại"), ("tai quay", "tại quầy"), ("tai cua hang", "tại quầy"), ("san", "sàn TMĐT"),
              ("shopee", "sàn TMĐT"), ("facebook", "facebook"), ("website", "website")]
UNITS = r"(kg|chai|goi|hop|thung|bao|vien|lon|ly|cai|tuyp|vi|phan|dia|cuc|m3|m2|bo|binh|cay|cuon|lo|ong|mieng|to|noi|con|lan|tam|met|m|ket|qua|bich|lit|lit)"


def J(x): return repr(x) if not isinstance(x, str) else '"' + x.replace('"', '\\"') + '"'


def fargs(f):
    return "".join(f", {k}={J(v)}" for k, v in f.items())


def ans(t, expr): return f'ANSWER = {{"type": "{t}", "answer": {expr}}}'


def parse(q, today, shop=None):
    """(template, code) for the common templates; None when unsure."""
    t0 = K(q); t = untelex(t0); tk = dm.key(q); p, P = parse_period(q, today)
    if P is None: p, P = parse_period(t, today)
    def has(*pats): return any(re.search(x, t) or re.search(x, t0) for x in pats)
    V = Vocab(shop) if shop is not None else None
    f = {}; item = None
    if V is not None:
        item = V.find(tk, V.items) or V.find(tk, V.codes)
        if item is not None: f["item"] = item
        m = re.search(r"nhom (?:hang |mon |san pham )?(.+)", tk)
        if m and item is None:
            cg = V.find(m.group(1), V.cgroups)
            if has(r"khach") and cg is not None: f["customer_group"] = cg
            else:
                g = V.find(m.group(1), V.groups)
                if g is not None: f["group"] = g
        m = re.search(r"(?:nhan vien|nguoi ban|nv)(?: ma)? (nv\s*\d+)", tk)
        if m: f["staff"] = V.find(m.group(1).replace(" ", ""), V.staff) or m.group(1).upper().replace(" ", "")
        m = re.search(r"hoat chat (.+)", tk)
        if m and V.ingredients:
            ing = V.find(m.group(1), V.ingredients)
            if ing is not None: f["where"] = f"lambda d: d['active_ingredient'].map(key) == key({J(ing)})"
    pays = [lab for w, lab in PAY_WORDS if re.search(r"\b" + w + r"\b", t)]
    chans = [lab for w, lab in CHAN_WORDS if re.search(r"(qua|ban|tai|kenh|dat)\s+(?:\w+\s+)?" + w + r"\b", t) or (w in ("zalo", "shopee") and w in t)]
    if has(r"ca sang") and not has(r"ca chieu"): f["shift"] = "sáng"
    elif has(r"ca chieu") and not has(r"ca sang"): f["shift"] = "chiều"
    if has(r"ke don|can don bac si|theo don"):
        neg = bool(re.search(r"khong ke don|khong can don|tru thuoc ke don|tru .*ke don|khong theo don|khong phai thuoc ke don", t))
        f["where"] = "lambda d: ~d['prescription'].fillna(False).astype(bool)" if neg else "lambda d: d['prescription'].fillna(False).astype(bool)"
    fa = lambda d=None: fargs({k: v for k, v in (d if d is not None else f).items() if k != "where"}) + (f", where={f['where']}" if "where" in (d if d is not None else f) else "")
    rev = has(r"doanh thu|dthu|revenue|ban duoc .*tien|ban dc .*tien|thu ve|tien hang|tien ban|bao nhieu tien|bnhiu tien|bn tien|nhiu tien|nhieu tien|mang ve|thu duoc")
    debt = has(r"\bno\b|cong no|tra no|mua chiu|chua tra|thu no|ghi no")
    # ---------------- debt (§4)
    if debt and not has(r"nhap"):
        d = until_day(q, today)
        if has(r"tuoi no"): return "debt_aging", ans("list", f"debt_aging({d})")
        m = re.search(r"(?:hon|qua|tren) (\d+) ngay", t)
        if m and has(r"don dat|dat (da )?(hon|qua)"): return "debt_older", ans("number", f"debt_older_than({d}, {int(m.group(1))})")
        m = re.search(r"(\d+) khach", t) or re.search(r"top (\d+)", t)
        if m and has(r"nhieu nhat|nhiu nhat|lon nhat|cao nhat"): return "top_debtors", ans("table", f"top_debtors({d}, {int(m.group(1))})")
        if has(r"no them|ghi no them|mua chiu them") and has(r"tra"):
            cust = V.find(tk, V.customers) if V else None
            return "debt_flow", ans("list", f"debt_flow({P}" + (f", customer={J(cust)}" if cust else "") + ")") if P else None
        if has(r"tra bot no|tra no|thu no|thu duoc .*no|thu hoi|khach tra") and P and not has(r"con no|chua tra"): return "debt_repaid", ans("number", f"debt_flow({P})[1]")
        if has(r"bao nhieu khach|so khach|may khach|bnhiu khach|bn khach") and has(r"con no|chua tra|cong no"): return "n_debtors", ans("number", f"n_debtors({d})")
        if has(r"con no|chua tra|tong no|cong no"): return "debt_total", ans("number", f"debt_total({d})")
        return None
    # ---------------- price effect
    if has(r"tang gia|len gia|doi gia|tang len") and item is not None:
        m = re.search(r"(\d+) ngay (?:dau|ke tu|tu)", t) or re.search(r"(\d+) ngay", t)
        n = int(m.group(1)) if m else None
        dd = re.search(r"(\d{1,2})/(\d{1,2})/(20\d\d)", t)
        pr = re.findall(r"(\d{3,})\s*(?:d|dong)\b", t)
        if n is None: return None
        if dd: return "price_effect", ans("percent", f"price_effect({J(item)}, {n}, date={_dc(_date(*dd.group(1, 2, 3)))})")
        if pr: return "price_effect", ans("percent", f"price_effect({J(item)}, {n}, new_price={float(pr[-1])})")
        return None
    # ---------------- purchases
    if has(r"nhap") and not has(r"nhap tru ban|tru so ban|chenh lech nhap"):
        if has(r"binh quan|trung binh") and item is not None and P:
            u = re.search(r"(?:mot|1|moi) " + UNITS + r"\b", t)
            return "avg_purchase", ans("number", f"avg_purchase_price({J(item)}, {P}" + (f", unit={J(u.group(1))}" if u else "") + ")")
        if has(r"nha cung cap|ncc") and P:
            by = "receipts" if has(r"chinh|nhieu phieu|nhieu lan") else "amount"
            return "top_supplier", ans("string", f"top_supplier({P}, by={J(by)})")
        if P and has(r"tong|bao nhieu tien|het bao nhieu|gia tri|tien hang|bnhiu|bn\b|nhieu tien|tien nhap"): return "purchase_total", ans("number", f"purchase_total({P})")
        return None
    if has(r"nhap tru ban|tru so ban|chenh lech nhap") and item is not None and P: return "import_minus_sales", ans("number", f"import_minus_sales({J(item)}, {P})")
    # ---------------- lots / expiry
    if has(r"han dung|het han|\blo\b"):
        d = "TODAY"
        m = re.search(r"den het (?:ngay )?(\d{1,2})/(\d{1,2})/(20\d\d)", t) or re.search(r"(?:toi|den) (?:het )?(?:ngay )?(\d{1,2})/(\d{1,2})/(20\d\d)", t)
        until = _dc(_date(*m.group(1, 2, 3))) if m else None
        n = re.search(r"(\d+) ngay toi", t)
        if until is None and n: until = f"TODAY + {int(n.group(1))} * D"
        if has(r"gan het han nhat|het han som nhat|gan het han nhut") and item is not None: return "nearest_expiry", ans("date", f"nearest_expiry({J(item)}, {d})")
        if until is not None:
            if has(r"tien von|gia tri|bao nhieu tien|bnhiu tien"): return "expiring_value", ans("number", f"money((lots_expiring({d}, {until})['remaining'] * lots_expiring({d}, {until})['unit_cost_base']).sum())")
            if has(r"may lo|bao nhieu lo|so lo|bnhiu lo"): return "expiring_lots", ans("number", f"len(lots_expiring({d}, {until}))")
        return None
    # ---------------- stock
    if has(r"gia von|tien von") and has(r"kho|\bton\b|hang ton"): return "stock_value", ans("number", f"stock_value({until_day(q, today)})")
    if has(r"ton kho|con ton|trong kho|con lai trong kho|ton cuoi|kho con|con bao nhieu|con may") and not rev:
        d = until_day(q, today)
        if has(r"gia von|tien von|gia tri") : return "stock_value", ans("number", f"stock_value({d})")
        if item is not None:
            m = re.search(r"(?:may|bao nhieu|bnhiu|bn) " + UNITS + r" (?:chan|nguyen)", t)
            if m: return "stock_split", ans("list", f"stock_split({J(item)}, {d}, {J(m.group(1))})")
            return "stock", ans("number", f"stock_at({J(item)}, {d})")
        return None
    # ---------------- deliveries
    if has(r"dot (giao|hang)|chuyen hang|luot giao|dot cho") and P: return "deliveries", ans("number", f"n_delivery_trips({P})")
    # ---------------- profit
    if has(r"lai rong|loi nhuan rong|tru het chi phi|tru chi phi|loi that|lai thuc") and P: return "net_profit", ans("number", f"net_profit({P})")
    if has(r"lai gop|loi gop|lai nhat|loi nhat|lai nhieu nhat|loi nhieu nhat|lai cao nhat|lai gop cao nhat|cho loi|mang lai lai") and P:
        if has(r"hang nao|mat hang|san pham nao|mon nao|\bsp\b|nao") : return "top_profit_item", ans("string", f"top_items({P}, 1, by='profit')[0]")
        return "gross_profit", ans("number", f"gross_profit({P}{fa()})")
    # ---------------- unsold items
    m = re.search(r"(?:it nhat|tu|tren|o|trong) (\d+) don (?:hang )?khac nhau|co mat (?:trong|o) (?:it nhat )?(\d+) don|o (\d+) don (?:hang )?khac nhau", t)
    if m and P and has(r"mat hang|mon|san pham"):
        n = int(next(g for g in m.groups() if g)); return "items_in_n_orders", ans("number", f"int((item_sales({P}, by='n_orders') >= {n}).sum())")
    if has(r"\bbi e\b|ban khong duoc|ban ko duoc|ban ko dc|khong ai mua|e han|\bees\b|mat hang e|mon e|hang e|\be\b.*khong ban|\be\b.*ko ban|\be\b.*k ban") and P and has(r"mat hang|mon|san pham"):
        return "unsold", ans("number", f"len(unsold_items({P}))")
    # ---------------- price on a day
    if has(r"don gia|gia ban|gia bao nhieu|gia bnhiu|ban ra voi|voi don gia|gia mot|gia 1") and item is not None and p is not None and p.days == 1:
        return "price_on", ans("number", f"price_on({J(item)}, {P.replace('day(', '').rstrip(')') if P.startswith('day(') else 'TODAY'})")
    # ---------------- cancel
    if has(r"\bhuy\b"):
        if has(r"ti le|ty le|%|phan tram") and P: return "cancel_rate", ans("percent", f"cancel_rate({P})")
        if has(r"bao nhieu don|may don|so don|bnhiu don|bn don|dem") and P and not has(r"khach"): return "n_cancelled", ans("number", f"len(orders({P}, valid=None, status='huỷ'))")
        if has(r"khach") and has(r"nhieu|nhat") and P: return "top_canceller", ans("string", f"top_customers({P}, 1, by='n_cancelled')[0]")
        return None
    # ---------------- monthly lists / trends / best month
    ms = all_months(q)
    if has(r"tang lien tuc|luon cao hon|thang sau .*cao hon|tang dan|co phai .*cao hon") and rev:
        if len(ms) >= 2 and has(r"\b(tu|giua) "):
            if len(ms) == 2: return "increasing", ans("boolean", f"increasing(month_seq({ms[0][6:-1]}, {ms[1][6:-1]}))")
        if len(ms) >= 2: return "increasing", ans("boolean", f"increasing([{', '.join(dict.fromkeys(ms))}])")
        return None
    if has(r"lan luot|moi thang|tung thang|theo thu tu|tu thang cu|3 thang lien") and rev and not pays and not chans:
        if len(ms) == 2 and has(r"(tu|ket thuc|den) "):
            if has(r"ket thuc"): return "months_list", ans("list", f"monthly_revenue(months_back({ms[-1]}, 3))")
            return "months_list", ans("list", f"monthly_revenue(month_seq({ms[0][6:-1]}, {ms[1][6:-1]}))")
        if len(ms) >= 2: return "months_list", ans("list", f"monthly_revenue([{', '.join(ms)}])")
        m = re.search(r"(\d) thang (?:lien nhau |lien tiep )?ket thuc", t)
        if m and ms: return "months_list", ans("list", f"monthly_revenue(months_back({ms[0]}, {int(m.group(1))}))")
        return None
    if has(r"thang nao") and rev and len(ms) >= 2:
        seq = f"month_seq({ms[0][6:-1]}, {ms[-1][6:-1]})" if len(ms) == 2 and has(r"\b(tu|den|toi)\b") else f"[{', '.join(ms)}]"
        return "best_month", ans("date", f"best_month({seq})")
    # ---------------- growth
    if has(r"tang hay giam|thay doi|tang truong|len hay xuong|cao hay thap|hon hay kem|tang giam|lech bao nhieu %|bao nhieu phan tram so|% so") and (rev or has(r"don")):
        meas = "revenue" if rev else "n_orders"
        if has(r"ca chieu") and has(r"ca sang"): return "growth_shift", ans("percent", f"growth({meas}({P}, shift='chiều'), {meas}({P}, shift='sáng'))") if P else None
        if has(r"truoc tet") and P: return "growth_tet", ans("percent", f"growth({meas}({P}), {meas}(before_tet({P[4:-1] if P.startswith('tet(') else 'TODAY.year'})))") if P.startswith("tet") else None
        if has(r"tet nam (truoc|ngoai|roi)") and has(r"tet nam nay|tet nay") : return "growth_tet_yoy", ans("percent", f"growth({meas}(tet_of(TODAY)), {meas}(tet(TODAY.year - 1)))")
        if has(r"cung ky thang (truoc|roi|trc|lien truoc)") or (has(r"dau thang|thang nay") and has(r"cung ky")):
            return "growth_mtd", ans("percent", f"growth({meas}(this_month(TODAY){fa()}), {meas}(prev_month_same_days(TODAY){fa()}))")
        if len(ms) == 2: return "growth2", ans("percent", f"growth({meas}({ms[0]}{fa()}), {meas}({ms[1]}{fa()}))")
        if P and P.startswith("month") and has(r"thang (truoc|lien truoc|lien trc|trc|ngay truoc|truoc do|lien ke)"): return "growth_prev", ans("percent", f"growth({meas}({P}{fa()}), {meas}(prev_month({P}){fa()}))")
        if P and P.startswith("month") and has(r"cung (ky|thang) nam (ngoai|truoc|roi)|nam ngoai|nam truoc"): return "growth_yoy", ans("percent", f"growth({meas}({P}{fa()}), {meas}(same_month_last_year({P}){fa()}))")
        return None
    # ---------------- shares
    if has(r"chiem|ti trong|ty trong|phan tram|%|dong gop|ti le") and P and rev:
        g = {k: v for k, v in f.items() if k != "shift"}
        if len(chans) == 1: g["channel"] = chans[0]
        if len(pays) == 1: g["payment"] = pays[0]
        if f.get("shift"): g["shift"] = f["shift"]
        if g: return "share", ans("percent", f"share({P}{fa(g)})")
        return None
    # ---------------- customers
    if has(r"khach") and P:
        m = re.search(r"(?:tu|it nhat) (\d+) don", t)
        if m:
            if has(r"phan tram|%|ti le"): return "pct_customers", ans("percent", f"pct_customers({P}, {int(m.group(1))})")
            return "n_customers_min", ans("number", f"n_customers({P}, min_orders={int(m.group(1))})")
        by = "n_orders" if has(r"nhieu (lan|don)|nhiu (lan|don)|so don|nhieu don") else "n_days" if has(r"trung thanh|nhieu ngay") else "aov" if has(r"\bsop\b") else "revenue" if has(r"nhieu tien|nhiu tien|chi tieu|tien nhieu|mua nhieu nhat|cao nhat|lon nhat|nhieu nhat|nhiu nhat|tong tien mua") else None
        if by:
            m = re.search(r"(?<![\d/])([1-9]|1\d|20) khach", t)
            if m: return "top_customers", ans("list", f"top_customers({P}, {int(m.group(1))}, by={J(by)})")
            if has(r"\bai\b|khach nao|la ai|nguoi nao"): return "top_customer", ans("string", f"top_customers({P}, 1, by={J(by)})[0]")
        if "customer_group" in f and rev and has(r"trung binh|binh quan") and has(r"don"): return "aov", ans("number", f"aov({P}{fa()})")
        if "customer_group" in f and rev: return "revenue", ans("number", f"revenue({P}{fa()})")
        return None
    # ---------------- staff
    if has(r"nhan vien|nguoi ban|\bnv\b") and P:
        if "staff" in f and rev: return "revenue", ans("number", f"revenue({P}{fa()})")
        if has(r"nhat|gioi nhat|cao nhat"): return "top_staff", ans("string", "rank(staff_sales(" + P + "), 1)[0]")
        return None
    # ---------------- items: top / qty / size
    m = re.search(r"(?:top |liet ke |^|cho minh |cho toi )(\d+) (?:mat hang|hang|san pham|mon|sp|best)", t) or re.search(r"top (\d+)", t) or re.search(r"(\d+) (?:mat hang|mon|san pham|hang) (?:ban )?(?:chay|nhieu)", t)
    best = has(r"ban chay|chay nhat|ban nhieu nhat|ban dc nhieu nhat|ban duoc nhieu nhat|hot nhat|best seller|ban nhiu nhat|ban dc nhiu nhat|nhieu nhat|nhiu nhat")
    if best and P and (has(r"mat hang|hang nao|san pham|mon nao|\bsp\b|mon |hang gi|la gi|la chi|hang chi|best seller|top|nhung") or m):
        by = "revenue" if has(r"nhieu tien|nhiu tien|doanh thu") else "n_orders" if has(r"nhieu don|ua chuong") else "qty"
        g = {k: v for k, v in f.items() if k in ("group",)}
        if m: return "top_items", ans("list", f"top_items({P}, {int(m.group(1))}, by={J(by)}{fa(g)})")
        return "top_item", ans("string", f"top_items({P}, 1, by={J(by)}{fa(g)})[0]")
    if has(r"size (m|l)\b") and P and not rev:
        sz = re.search(r"size (m|l)\b", t).group(1).upper(); return "size_qty", ans("number", f"size_qty({P}, {J(sz)})")
    if P and not rev and (item is not None or "group" in f) and (re.search(r"(bao nhieu|may|bnhiu|bn|so|tong cong|total) " + UNITS + r"\b", t) or has(r"ban ra bao nhieu|ban duoc bao nhieu|ban dc bao nhieu|so luong")):
        return "item_qty", ans("number", f"item_qty({P}{fa()})")
    # ---------------- revenue variants
    if has(r"co may ngay|bao nhieu ngay|bnhiu ngay|bn ngay|may bua|co may hom|so ngay|dem so ngay|bao nhieu bua") and P:
        x = money_in(t, r"(?:tu|dat|tren|it nhat)\s*")
        if x and has(r"tro len"): return "n_days_rev_ge", ans("number", f"n_days_revenue_at_least({P}, {x:.0f})")
    if has(r"don") and has(r"tro len") and P and (rev or has(r"gia tri")):
        x = money_in(t, r"(?:tu|it nhat|tri gia|gia tri)\s*")
        if x: return "n_orders_rev_ge", ans("number", f"n_orders_revenue_at_least({P}, {x:.0f}{fa()})")
    if has(r"trung binh|binh quan") and has(r"don|bill|hoa don") and rev and P and not has(r"nhap|moi ngay"):
        return "aov", ans("number", f"aov({P}{fa()})")
    if has(r"ngay nao|ngay may|hom nao") and P and (rev or has(r"cao nhat|nhieu nhat")):
        by = "orders" if has(r"nhieu don|dat khach|don nhat") else "revenue"
        return "best_day", ans("date", f"best_day({P}, by={J(by)})")
    if has(r"gio cao diem|gio nao") and P: return "peak_hour", ans("number", f"peak_hour({P})")
    if rev and P and (len(pays) == 2 or len(chans) == 2) and has(r"\bva\b|voi|lan luot|moi loai"):
        key_ = "payment" if len(pays) == 2 else "channel"; vals = pays if len(pays) == 2 else chans
        words = PAY_WORDS if key_ == "payment" else CHAN_WORDS
        pos = {lab: min((t.find(w) for w, l2 in words if l2 == lab and t.find(w) >= 0), default=999) for lab in vals}
        order = sorted(vals, key=lambda v: pos[v])
        return "rev_pair", ans("list", f"[revenue({P}, {key_}={J(order[0])}), revenue({P}, {key_}={J(order[1])})]")
    if rev and P and not has(r"tang|giam|so voi|so vs|top|nhat|%|phan tram|ti trong|chiem|lan luot|\bva\b|moi loai|tung"):
        g = dict(f)
        if len(pays) == 1 and has(r"tra bang|thanh toan|tra qua|bang"): g["payment"] = pays[0]
        if len(chans) == 1: g["channel"] = chans[0]
        if not g and has(r"mat hang|san pham|\bsp\b|nhom|kenh|tra bang|thanh toan"): return None
        return "revenue", ans("number", f"revenue({P}{fa(g)})")
    if has(r"so don|may don|bao nhieu don|bnhiu don|bn don|so bill|bao nhieu bill|may bill|bao nhieu hoa don|nhieu don|tong don|ddown|don hang|\bdh\b|o do|dem so don|bao nhieu don hang|chot duoc") and P \
            and not has(r"tra hang|khach|tro len|giao|%|ti le|phan tram|trung binh|co mua|co ban"):
        g = dict(f)
        if len(chans) == 1: g["channel"] = chans[0]
        if len(pays) == 1 and has(r"tra bang|thanh toan"): g["payment"] = pays[0]
        return "n_orders", ans("number", f"n_orders({P}{fa(g)})")
    return None
