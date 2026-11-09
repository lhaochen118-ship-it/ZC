#!/usr/bin/env python3
"""Hint bundle of one question (docs/08_MASTER_PLAN.md §5.2, module M5): what general rules read from the question and the
shop's data, given to the model as context. Logic only (allowed by the user's boundary of 08/11): no toolkit metric name, no
template name, no program; the model still writes every program.

    build_hints(question, as_of, shop, normalizer=None, rule_terms=()) ->
      {"clean": cleaned question (when it differs), "period": {"text", "code", "from", "to"} | None, "periods": [...],
       "entities": [{"text", "kind", "data_value", "how", "candidates"}], "answer_type": ... | None,
       "rule_terms": [{"term", "definition"}], "missing_data": [...], "measure_words": [...]}
'code' of a period is a period expression of the toolkit (month(2025, 7), last_week(TODAY), ...): the plan allows it."""
import re, difflib, unicodedata
import dm, qparse, industry

KINDS = ("item", "group", "customer", "staff", "supplier", "customer_group", "ingredient", "channel", "payment", "vehicle")
MEASURES = [("doanh thu", r"\bdoanh thu\b|\bdthu\b|\bdoanh so\b|\btien ban\b|\bthu ve\b|\bban duoc bao nhieu tien\b"),
            ("số đơn", r"\bso don\b|\bbao nhieu don\b|\bmay don\b|\bbill\b|\bhoa don\b"),
            ("số lượng bán", r"\bso luong\b|\bban (?:ra |duoc )?(?:bao nhieu|may)\b|\bban chay\b|\bban it\b|\bban nhieu\b"),
            ("giá trị đơn trung bình", r"\btrung binh (?:moi|1|mot) don\b|\bgia tri trung binh\b|\bbinh quan (?:moi|1|mot) don\b"),
            ("tỉ lệ huỷ", r"\bti le huy\b|\bty le huy\b|\bhuy bao nhieu phan tram\b"),
            ("lợi nhuận", r"\bloi nhuan\b|\blai gop\b|\blai rong\b|\btien lai\b|\bloi\b"), ("giá vốn", r"\bgia von\b"),
            ("công nợ", r"\bno\b|\bcong no\b|\bghi so\b|\bchua tra\b"), ("tồn kho", r"\bton kho\b|\bcon (?:lai )?bao nhieu\b|\bcon ton\b|\bton\b"),
            ("nhập hàng", r"\bnhap hang\b|\bnhap\b|\bphieu nhap\b"), ("giá bán", r"\bgia ban\b|\bgia bao nhieu\b|\bgia\b"),
            ("tăng trưởng", r"\btang truong\b|\btang bao nhieu\b|\bgiam bao nhieu\b|\bso voi\b"),
            ("khách hàng", r"\bkhach\b"), ("nhân viên", r"\bnhan vien\b|\bthu ngan\b|\btho\b"), ("hạn dùng", r"\bhan dung\b|\bhet han\b|\bdate\b|\blo\b")]


def _k(s): return " ".join(dm.key(s).split())


class ShopVocab:
    """entity values of one shop by kind (key -> value), with regional / accent-free keys (built once per shop)."""
    def __init__(self, shop):
        V = qparse.Vocab(shop)
        self.tables = {"item": dict(V.items), "group": dict(V.groups), "customer": {k: v for k, v in V.customers.items() if not dm.is_walkin(v)},
                       "staff": dict(V.staff), "customer_group": dict(V.cgroups), "ingredient": dict(V.ingredients)}
        for k, code in V.staff_names.items(): self.tables["staff"].setdefault(k, code)
        self.codes = dict(V.codes)
        self.items_reg = dict(V.items_reg)
        P = shop.C.get("purchases")
        self.tables["supplier"] = {_k(x): x for x in (P["supplier"].dropna().unique() if P is not None and "supplier" in P.columns else []) if str(x).strip()}
        O = shop.O
        veh = set()
        for c in ("vehicle", "vehicle_line"):
            if c in O.columns: veh |= {x for x in O[c].dropna().unique() if str(x).strip()}
        self.tables["vehicle"] = {_k(x): x for x in veh}
        self.channels = {k: v for k, v in qparse.CHAN_WORDS}
        self.payments = {k: v for k, v in qparse.PAY_WORDS}
        has_ch = "channel" in O.columns and O["channel"].notna().any()
        has_pm = "payment_method" in O.columns and O["payment_method"].notna().any()
        self.tables["channel"] = {k: v for k, v in qparse.CHAN_WORDS} if has_ch else {}
        self.tables["payment"] = {k: v for k, v in qparse.PAY_WORDS} if has_pm else {}
        # units of measure of the shop's data (lines / items / conversions): base units first ('gói', 'ly', 'đĩa', 'kg')
        base, units = set(), set()
        for tab, cols in (("lines", ("base_unit", "unit")), ("items", ("base_unit", "unit")), ("conversions", ("base_unit", "unit"))):
            T = shop.C.get(tab)
            if T is None: continue
            for c in cols:
                if c in T.columns:
                    vals = {str(x).strip() for x in T[c].dropna().unique() if str(x).strip() and len(str(x)) <= 20}
                    (base if c == "base_unit" else units).update(vals)
        self.base_units = {_k(x) for x in base}
        self.tables["unit"] = {_k(x): x for x in sorted(units | base) if _k(x)}


def _spans(tk, table, min_len=3):
    """(start, end, key, value) of every table key occurring as whole words in the key text tk (longest first, no overlap)."""
    out, taken = [], []
    for k in sorted(table, key=len, reverse=True):
        if len(k) < min_len: continue
        for m in re.finditer(r"(?<!\S)" + re.escape(k) + r"(?!\S)", tk):
            if any(m.start() < e and s < m.end() for s, e in taken): continue
            out.append((m.start(), m.end(), k, table[k])); taken.append((m.start(), m.end()))
    return out


def link_entities(text, vocab, raw=None):
    """entities of the question (cleaned text; raw = the question as typed) linked to data values."""
    tk = _k(text); raw_k = _k(raw) if raw else tk; raw_l = str(raw or text).lower()
    ents, taken = [], []
    def add(s, e, kind, val, how, cands=()):
        if any(s < e2 and s2 < e for s2, e2 in taken): return
        taken.append((s, e)); ents.append({"text": tk[s:e], "kind": kind, "data_value": val, "how": how, "candidates": list(cands)[:3]})
    for kind in ("item", "ingredient", "customer", "staff", "supplier", "vehicle", "group", "customer_group", "channel", "payment"):
        tab = vocab.tables.get(kind) or {}
        for s, e, k, v in _spans(tk, tab, 2 if kind in ("channel", "payment") else 3):
            if kind in ("channel", "payment"): add(s, e, kind, v, "word"); continue
            how = "exact" if str(v).lower() in raw_l else ("no_accent" if k in raw_k else "cleaned")
            add(s, e, kind, v, how)
    for s, e, k, v in _spans(tk, vocab.codes, 4): add(s, e, "item", v, "code")
    # a unit of measure counted by the question ('bán được mấy gói', 'bao nhiêu ly', 'tính theo thùng'): only after a counting word
    for s, e, k, v in _spans(tk, getattr(vocab, "tables", {}).get("unit") or {}, 2):
        if re.search(r"(?:bao nhieu|bnhiu|\bbn|\bmay|\bso|\btheo|\bbang|tong cong|tat ca)(?: \S+)? $", tk[:s]):
            kd = "base_unit" if k in getattr(vocab, "base_units", set()) else "unit"
            if not any(x["kind"] == kd and x["data_value"] == v for x in ents): add(s, e, kd, v, "word")
    for s, e, k, v in _spans(dm.regional_key(text), vocab.items_reg, 4):        # regional words ('cá tràu' = 'cá lóc')
        if not any(x["data_value"] == v for x in ents): add(s, e, "item", v, "regional")
    # model lines / name prefixes: 'iPhone 16' names every 'iPhone 16 ... GB' item
    lines = {}
    for v in vocab.tables["item"].values():
        try: ml = industry.model_line(v)
        except Exception: ml = None
        if ml and _k(ml) != _k(v): lines.setdefault(_k(ml), ml)
    for s, e, k, v in _spans(tk, lines, 4):
        if not any(x["data_value"] == v for x in ents): add(s, e, "item", v, "abbrev")
    # typos: capitalised runs / 'bán <X>' phrases no table linked -> closest names (never chosen silently: candidates only)
    if qparse.unlinked_entity(raw or text, tk):
        words = [w for w in re.findall(r"[^\W\d_][\w\-]*", str(raw or text)) if w[:1].isupper()]
        phrase = _k(" ".join(words)) if len(words) >= 2 else ""
        if phrase and not any(phrase in x["text"] or x["text"] in phrase for x in ents):
            for kind in ("item", "customer", "staff", "supplier"):
                keys = list((vocab.tables.get(kind) or {}).keys())
                near = difflib.get_close_matches(phrase, keys, n=3, cutoff=0.75)
                if near:
                    ents.append({"text": phrase, "kind": kind, "data_value": vocab.tables[kind][near[0]], "how": "typo",
                                 "candidates": [vocab.tables[kind][x] for x in near]}); break
    return ents


_NOUN = r"(?:mat hang|mon|san pham|sp|khach(?: hang)?|nha cung cap|ncc|nhan vien|tho|dong (?:may|xe)|xe|hang|nhom|loai|(?:ngay|thang)(?= (?:\w+ ){0,2}nao\b))"
_SEL = r"\b(?:nao|nhat|nhut|top|chay|ai)\b"


_DEVICE_MAY = re.compile(r"\b(?:chiec|cai|ban|nhom|luot|dong|cac|nhung|so|loai|mua|thu|sua|doi|tra|gia|hang|tien)\s+may\b|\bmay\s+(?:cu|moi|dien thoai|tinh|anh|ban|lanh|giat|in|nay|do|kia|samsung|iphone|oppo|xiaomi|vivo|realme)\b")


def n_asks(t, clean=None):
    """how many quantities the question asks ('bao nhiêu', 'mấy'): 'mấy' and 'máy' (a phone) are both 'may' without accents,
    so the accented question is read when it has accents, else 'may' next to a device word is not counted."""
    n = len(re.findall(r"\bbao nhieu\b|\bbnhiu\b|\bbn\b", t))
    c = unicodedata.normalize("NFC", str(clean or "")).lower()
    if c and c != dm.strip_accents(c):
        m = len(re.findall(r"(?<!\w)mấy(?!\w)", c))
        if n and re.match(r"\W*mấy\s+(?:lô|đơn|mặt hàng|món|khách|máy|xe|phiếu)\b", c): m -= 1   # 'Mấy lô còn tồn ... đọng bao nhiêu
        return n + m                                                                        # tiền': the subject, one question
    return n + len(re.findall(r"\bmay\b", t)) - len(_DEVICE_MAY.findall(t))


def answer_type(t, clean=None):
    """the kind of answer the question asks (accent-free key t of the cleaned question; clean: the cleaned question itself,
    read for 'mấy' / 'máy'), or None."""
    na = n_asks(t, clean)
    asks_n = na > 0 or re.search(r"\bbao nhieu\b|\bbnhiu\b|\bbn\b", t)
    total = re.search(r"\bca\s+(?!sang|chieu|toi|dem|trua|hai\b)(?:\w+\s+){1,4}(?:lan|voi)\s+\w+|\bgop (?:ca|chung|lai|het)\b|\btinh ca\b|\bcong (?:lai|chung)\b", t)
    # ^ one total over several groups ('cả máy mới lẫn máy cũ', 'gộp cả', 'tính cả'): a number, not a list
    if re.search(r"\b(?:co|lieu)\b.*\b(?:khong|ko|hong|chua|phai khong|dung khong)\s*\??$", t) and not asks_n and not re.search(r"\bnao\b|\bai\b", t): return "boolean"
    if re.search(r"\bngay (?:nao|mo)\b|\bhom (?:nao|mo)\b|\bbua (?:nao|mo)\b|\bkhi nao\b|\bngay may\b|\bthang (?:nao|mo)\b", t) and not re.search(r"(?<![/\d])\b\d{1,2} (?:ngay|thang) nao\b", t): return "date"
    if re.search(r"\bphan tram\b|\bti le\b|\bty le\b|\btang truong\b|\b(?:bao nhieu|may|bn)\s*%|\bchiem bao nhieu\b", t): return "percent"
    if (na >= 2 and not total) or re.search(r"\b(?:moi|tung) ben\b|\bca hai\b|\blan luot\b", t): return "list"
    each = re.search(r"\bbang\b|\bkem (?:theo )?(?:so|tien|doanh thu)\b|\bcung (?:voi )?so\b", t) or re.search(r"\b(?:moi|tung)\s+(?:nguoi|khach|mat hang|mon|san pham|nha cung cap|nhan vien|tho|ncc)\b.*\bbao nhieu\b|\bva (?:so|tien|doanh thu)\b.*\b(?:cua )?(?:tung|moi)\b", t)
    m = re.search(r"\btop\s*(\d{1,2})\b|(?<![/\d.,])\b(\d{1,2}|hai|ba|bon|sau|bay|tam|chin|muoi)\s+" + _NOUN + r"\b(?![/\d])", t)
    if m and re.search(_SEL, t):
        n = m.group(1) or m.group(2); n = int(n) if n.isdigit() else qparse.NUMW.get(n, 2)
        if n >= 2: return "table" if each else "list"
    if re.search(r"\b(?:moi|tung|lan luot|theo)\s+(?:thang|ngay|tuan|quy)\b", t) and asks_n: return "list"
    # several groups asked at once ('chia theo tuổi nợ 0-7, 8-15, trên 15 ngày, mỗi nhóm bao nhiêu'; 'từng loại / mỗi kênh có bao nhiêu')
    if asks_n and not total and re.search(r"\b(?:moi|tung)\s+(?:nhom|loai|kenh|hinh thuc|ca|khung|muc|phan|khoang)\b|\bchia (?:theo|thanh|ra)\b", t): return "list"
    if re.search(r"\bnhung\b|\bcac (?:mat hang|mon|san pham|khach|nha cung cap)\b|\bliet ke\b|\bdanh sach\b", t) and re.search(r"\bnao\b|\bgi\b|\bliet ke\b|\bdanh sach\b", t):
        return "table" if each else "list"
    if re.search(r"\b" + _NOUN + r"\s+(?:\w+\s+){0,3}nao\b|\bla ai\b|\bai\b.*\bnhat\b|\bten (?:gi|la gi)\b|\bla gi\b", t) and not asks_n: return "string"
    if asks_n or re.search(r"\btong\b|\btrung binh\b|\bbinh quan\b|\bdem\b", t): return "number"
    return None


_GENERIC = {"mat", "hang", "san", "pham", "mon", "khach", "don", "nguoi"}
_UNIT_KINDS = ("unit", "base_unit")
# general Vietnamese shop words that say the same as a rulebook word ('bữa' = 'ngày', 'củ' = 'triệu', 'cháy hàng' = 'hết hàng')
_SYNONYMS = [(r"\bbữa\b", "ngày"), (r"\bcủ\b", "triệu"), (r"\bcháy hàng\b", "hết hàng"), (r"\bnhứt\b", "nhất"), (r"\bmô\b", "nào"),
             (r"\bthằng\b", "người"), (r"\bsp\b", "sản phẩm"), (r"\bncc\b", "nhà cung cấp"), (r"\bxịn\b", "sộp")]


def _toks(text):
    """(accented lowercase word, accent-free key) of every word of a text."""
    t = unicodedata.normalize("NFC", str(text)).lower()
    return [(w, " ".join(dm.key(w).split())) for w in re.findall(r"[^\W_]+", t)]


_PLAIN = {}


def _plain_words(rows):
    """accent-free words the rulebook itself writes without accents ('nay', 'trong', 'ghi'): real words, never a typing slip."""
    k = id(rows)
    if k not in _PLAIN:
        txt = " ".join(f"{a} {b}" for a, b in rows.values())
        _PLAIN[k] = {w for w, kk in _toks(txt) if w == kk and not w.isdigit()}
    return _PLAIN[k]


def rule_terms_of(question, rulebook_rows, entities=None):
    """rulebook rows (term -> definition) whose term the question uses, most specific first (at most 4).
    A term matches when its distinctive words all occur in the question: generic nouns ('mặt hàng', 'khách', 'đơn') are not
    required when the term has another word; a placeholder '<đơn vị>' matches a unit linked in the question. Words are
    compared with their accents when the question has them (accent-free 'mấy' / 'máy', 'đồng' / 'dòng' differ), else
    without; shop-chat synonyms count ('bữa' = 'ngày')."""
    q = unicodedata.normalize("NFC", str(question)).lower()
    for rx, rep in _SYNONYMS: q = re.sub(rx, lambda m: m.group(0) + " " + rep, q)
    qt = _toks(q)
    acc = {w for w, k in qt if w != k}; plain = {k for w, k in qt if w == k}; keys = {k for _, k in qt}
    accented = bool(acc)
    has_unit = any((e or {}).get("kind") in _UNIT_KINDS for e in (entities or ()))
    known = _plain_words(rulebook_rows)
    def has(w, k):
        if not accented: return k in keys
        if w in acc or (w == k and k in plain): return True           # the same accents ('nay' of 'hôm nay' is not 'này')
        return k in plain and k not in known and len(k) >= 2          # a word left without accents in a mixed question
                                                                      # ('luot xe'), unless it is a real accent-free word
    out = []; seen = set()
    for term, (shown, definition) in rulebook_rows.items():
        ph = re.findall(r"<[^>]*>", shown)
        if ph and not has_unit: continue
        words = [(w, k) for w, k in _toks(re.sub(r"<[^>]*>", " ", shown)) if (len(k) > 1 or w != k) and w not in _STOP_ACC
                 and not (w == k and k in _STOP)]
        req = list(words)                                              # a leading generic head noun is optional when two
        while req and req[0][1] in _GENERIC: req = req[1:]              # other words remain ('Mặt hàng được ưa chuộng nhất'
        if len(req) < 2: req = words                                    # = 'hàng nào được khách ưa chuộng nhất')
        if (len(words) + len(ph) >= 2 or (len(req) == 1 and words != req)) and req and all(has(w, k) for w, k in req):
            if definition in seen: continue
            seen.add(definition); out.append((len(req) + len(ph), {"term": shown, "definition": definition[:240]}))
    out.sort(key=lambda x: -x[0])
    return [x[1] for x in out[:4]]


_STOP = set("va hoac cua cho la mot cac nhung trong ky theo khi thi bang ve tu den de duoc co khong nhat".split())
_STOP_ACC = set("và hoặc của cho là một các những trong kỳ kì theo khi thì bằng về từ đến để được có không nhất".split())


def rulebook_rows(text):
    """{term key: (term as written, definition)} of the rulebook's table rows ('| Khách sộp nhất | ... |')."""
    out = {}
    for line in str(text).splitlines():
        if not line.strip().startswith("|"): continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2 or set(cells[0]) <= set("-: ") or dm.key(cells[0]) in ("luat", "khai niem", "thuat ngu", "cach noi", "tu", "cum tu") \
                or dm.key(cells[1]) in ("nghia", "dinh nghia", "y nghia", "cach tinh", "quy tac", "giai thich"): continue   # header rows
        for t in cells[0].split(","):
            if t.strip(): out[_k(t)] = (t.strip(), " ".join(" ".join(cells[1:]).split()))
    return out


def build_hints(question, as_of, shop, normalizer=None, rule_rows=None, vocab=None):
    """the hint bundle (see the module doc). shop: dm.Shop of the question's day; normalizer: qnorm.Normalizer of the shop;
    rule_rows: rulebook_rows(rulebook) of the split; vocab: ShopVocab(shop) (built here when not given)."""
    raw = str(question)
    clean = raw
    if normalizer is not None:
        try: clean = normalizer.clean(raw, as_of)["clean"]
        except Exception: clean = raw
    t = qparse.untelex(qparse.K(clean))
    h = {"clean": clean if clean.strip() != raw.strip() else None, "period": None, "periods": [], "entities": [], "answer_type": None,
         "rule_terms": [], "missing_data": [], "measure_words": []}
    for txt in (clean, raw):
        try: p, code = qparse.parse_period(txt, as_of)
        except Exception: p, code = None, None
        if p is not None:
            h["period"] = {"code": code, "from": str(p.start.date()), "to": str(p.end.date())}; break
    try:
        ms = qparse.month_list(clean, as_of)
        if len(ms) >= 2: h["periods"] = ms
    except Exception: pass
    if vocab is None:
        try: vocab = ShopVocab(shop)
        except Exception: vocab = None
    if vocab is not None:
        try: h["entities"] = link_entities(clean, vocab, raw)
        except Exception: h["entities"] = []
    h["answer_type"] = answer_type(t, clean)
    if rule_rows: h["rule_terms"] = rule_terms_of(clean, rule_rows, h["entities"])
    try:
        if qparse.subjective(t): h["missing_data"].append("a quality no column records (subjective)")
        why = qparse.missing_info(clean, t, shop) if shop is not None else None
        if why: h["missing_data"].append(why)
    except Exception: pass
    h["measure_words"] = [name for name, rx in MEASURES if re.search(rx, t)][:4]
    return h
