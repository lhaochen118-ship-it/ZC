#!/usr/bin/env python3
"""Rule-based reading of Vietnamese period expressions and of the common question templates -> a short program over
the dm toolkit (the same API the LLM-written programs use).

parse_period(text, today) -> (dm.Period, code) | (None, None)   ("tháng 6/2025", "hqua", "tuần rồi", "quý 4/2025", ...)
parse(question, today, shop) -> (template, code) | None          code assigns ANSWER, e.g.
    ANSWER = {"type": "number", "answer": revenue(month(2025, 6), item="Coca Cola 330ml")}

Used (1) by tests/eval_lib.py to check the canonical tables + metrics against the train answers without an LLM,
(2) to build few-shot (question -> program) examples from train, and (3) at run time as one more voter.
Only generic language patterns; nothing is keyed on business or question ids."""
import re, unicodedata
import pandas as pd
import dm


def K(s):
    """accent-free, lowercase key that keeps / - : . , % between digits; '15.000.000' -> '15000000'."""
    s = dm.strip_accents(str(s)).lower().replace("≥", " tro len ").replace(">=", " tro len ")
    s = re.sub(r"(?<=\d)[.,](?=\d{3}(?!\d))", "", s); s = re.sub(r"(?<=\d)[.,](?=\d{3}(?!\d))", "", s)
    s = re.sub(r"[^\w\s/:\-%.,]", " ", s)
    s = re.sub(r"(?<!\d)[.,]|[.,](?!\d)", " ", s)
    return re.sub(r"\s+", " ", s).strip()


UNIT_QTY_WORDS = set("""hom nay qua hqua bua tuan thang nam quy ngay truoc trc roi vua cuoi dau tu den het toi ke gan day nua hien tai
    trong vao cho minh em anh chi hoi xem giup gium check coi voi nhe nha ha vay a nhi the ta oi biet muon cua hang shop quan tiem
    la bao nhieu nhiu bnhiu bn may so tong cong tat ca duoc dc ra luong moi mat mon tinh theo cac san pham sp loai deu ban""".split())
QTY_WORDS = {"so", "may", "nhieu", "nhiu", "bnhiu", "bn", "mot", "moi", "tong", "theo"}   # '<word> kg' = kilogram ('hơn k' stays 'không')
NEG = {"k": "khong", "ko": "khong", "hok": "khong", "hong": "khong", "kh": "khong", "kg": "khong", "khg": "khong", "hem": "khong", "hk": "khong"}


def untelex(t):
    """undo Telex typing left in accent-free text ('bawngf' -> 'bang', 'toonf' -> 'ton', 'dduwowcj' -> 'duoc') and spell
    colloquial negations as 'khong'. Vietnamese syllables never end in f / s / r / x / j, so those are tone keys."""
    out = []
    for w in t.split():
        if out and (re.search(r"\d$", out[-1]) and w in ("kg", "k") or w == "kg" and out[-1] in QTY_WORDS):
            out.append(w); continue                                    # 'số kg', '5 kg', '50 k': a unit, not 'không'
        if re.fullmatch(r"[a-z]+", w) and len(w) > 1:
            v = w.replace("dd", "d").replace("aa", "a").replace("ee", "e").replace("oo", "o")
            v = re.sub(r"(?<=[aou])w", "", v).replace("uow", "uo")
            if len(v) > 2 and v[-1] in "fsrxj" and re.search(r"[aeiouy]", v[:-1]): v = v[:-1]
            w = v
        if w == "hong" and out and out[-1] in ("hoa", "hue"): out.append(w); continue   # 'hoa / huê hồng' = commission, not 'không'
        out.append(NEG.get(w, w))
    return dialect(" ".join(out))


def dialect(t):
    """central-Vietnamese words in accent-free text, only where unambiguous: 'thang ni' -> 'thang nay', 'tho mo' -> 'tho nao',
    'toi chu' -> 'toi nay' ('mo' alone could be 'mở', so it is rewritten only after a noun that asks 'which')."""
    t = re.sub(r"\b(thang|tuan|nam|bua|hom|ngay|quy|luc|dot)\s+ni\b", r"\1 nay", t)
    t = re.sub(r"\b(tho|ngay|thang|xe|dong xe|hang|mat hang|mon|khach|nhan vien|loai|nhom|sp|san pham|nha cung cap|ncc)\s+mo\b", r"\1 nao", t)
    t = re.sub(r"\b(toi|den|tu dau nam toi|tu dau thang toi)\s+chu\b", r"\1 nay", t)
    return slang(re.sub(r"\brua\b", "vay", t))


def slang(t):
    """shop slang and frequent typos in accent-free text, only where unambiguous: 'bq' / 'bìn quân' -> 'binh quan', 'trở lne' ->
    'tro len', 'cất hàng' / 'đánh hàng (về)' (buying stock) -> 'nhap hang', '<từ / đạt / trên> 10 củ' -> '10 trieu', '3tr7' -> '3.7 trieu'."""
    t = re.sub(r"\bbq\b", "binh quan", t); t = re.sub(r"\bbin quan\b", "binh quan", t); t = re.sub(r"\btrung bin\b", "trung binh", t)
    t = re.sub(r"\blne\b", "len", t); t = re.sub(r"\bdh\b", "don hang", t)                # 'đh' = đơn hàng
    t = re.sub(r"(\d(?:\s?(?:d|dong|k|nghin|trieu|tr|cu))?)\s+(?:cho|tro|cho|tro) ?(?:len|lne|lenn)\b|(\d)\s*trolen\b", lambda m: (m.group(1) or m.group(2)) + " tro len", t)
    t = re.sub(r"\bchung binh\b", "trung binh", t); t = re.sub(r"\bchung bin\b", "trung binh", t)   # tr -> ch
    t = re.sub(r"\bcomay\b", "co may", t); t = re.sub(r"\bbnhiu(?=\w)", "bao nhieu ", t)
    t = re.sub(r"\bnhut\b", "nhat", t); t = re.sub(r"\b(ban|mua|ban ra|ban duoc) bao nhieu nhat\b", r"\1 nhieu nhat", t)   # southern 'nhứt'
    t = re.sub(r"\b(?:cat|danh) hang(?: ve)?\b", "nhap hang", t)
    t = re.sub(r"\btieu thu (duoc|ra|het)\b", r"ban \1", t)                # 'tiêu thụ được' = 'bán được'
    t = re.sub(r"\bphai thanh toan\b", "phai tra", t)                        # 'khách phải thanh toán' = 'khách phải trả'
    # takings said in other words = revenue ('tổng thu từ món X', 'tiền thu từ việc bán X', 'thu nhập từ các đơn ...', 'quán làm
    # được tổng tiền bao nhiêu'); not 'thu nợ' / 'thu mua' / 'thu máy' (debts, trade-ins)
    t = re.sub(r"\btong (?:so )?(?:tien )?thu (?:duoc )?(?:tu |cua )(?!no\b|mua\b|may\b)", "doanh thu cua ", t)
    t = re.sub(r"\btien thu (?:duoc )?tu (?:viec )?ban\b", "doanh thu ban", t)
    t = re.sub(r"\bthu nhap (?:tu|cua) (?!no\b)", "doanh thu cua ", t)
    t = re.sub(r"\blam (?:ra |duoc |ra duoc )(?:tong (?:cong )?)?(bao nhieu|bnhiu) tien\b", r"ban duoc \1 tien", t)   # money said: not
    t = re.sub(r"\blam (?:ra |duoc )tong (?:cong )?tien (bao nhieu|bnhiu)\b", r"ban duoc \1 tien", t)                # 'làm được mấy phiếu'
    t = re.sub(r"\bban ra duoc\b", "ban duoc", t)
    t = re.sub(r"\bkhach (?:(trung binh|binh quan) )?chi tra\b", lambda m: "khach " + (m.group(1) + " " if m.group(1) else "") + "phai tra", t)
    t = re.sub(r"\b(tu|dat|tren|hon|duoi|it nhat|qua|vuot|khoang|toi|den|tro len) (\d+(?:[.,]\d+)?) ?cu\b", r"\1 \2 trieu", t)
    return re.sub(r"\b(\d+) ?tr ?(\d)\b", r"\1.\2 trieu", t)


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
    m = re.search(r"(?:tu )?" + MON + r"(\d{1,2})\s*(?:/|-|nam )\s*" + Y + r" (?:den|toi|-|cho den) (?:het )?" + MON + r"(\d{1,2})\s*(?:/|-|nam )\s*" + Y + r"\b", t)
    if m:                                                              # a range of months: 'th9/2025 đến hết th11/2025'
        a = pd.Timestamp(_y(m.group(2)), int(m.group(1)), 1); b = pd.Timestamp(_y(m.group(4)), int(m.group(3)), 1) + pd.offsets.MonthEnd(0)
        if a <= b: return dm.between(a, b), f"between({_dc(a)}, {_dc(b)})"
    if re.search(r"\btet\b", t):
        m = re.search(r"tet (?:nam |am lich |nguyen dan )?(20\d\d)", t)
        if m: return dm.tet(int(m.group(1))), f"tet({int(m.group(1))})"
        if re.search(r"tet nam (truoc|ngoai|roi)", t): return dm.tet(today.year - 1), f"tet({today.year - 1})"
        return dm.tet_of(today), "tet_of(TODAY)"
    if re.search(r"(?:thoi diem |ngay )?bat dau (?:cua )?nam (?:nay |ni )?(?:(?!tu )\S+ ){0,3}(?:den|toi) (?:het )?(?:ngay )?(?:hom nay|nay|bay gio|hien tai)", t):
        return dm.ytd(today), "ytd(TODAY)"                              # 'từ thời điểm bắt đầu năm nay cho đến ngày hôm nay'
    if re.search(r"(?:tu |ke tu )(?:thoi diem )?(?:bat dau|dau) tuan (?:nay |ni )?(?:(?!tu )\S+ ){0,3}(?:den|toi) (?:het )?(?:ngay )?(?:hom nay|nay|bay gio|hien tai)", t):
        return dm.this_week(today), "this_week(TODAY)"
    m = re.search(r"(?:tinh )?nguoc (?:lai|ve truoc) (\d+) ngay|(\d+) ngay (?:tro lai day|tro lai|tro ve truoc)\b|trong vong (\d+) ngay (?:qua|gan day|gan nhat|vua qua|tro lai day)\b", t)
    if m:                                                              # 'tính ngược lại 30 ngày', '7 ngày trở lại đây'
        n = int(next(g for g in m.groups() if g)); return dm.last_n_days(n, today), f"last_n_days({n}, TODAY)"
    if re.search(r"(dau thang nay|tu dau thang|tu mung 1|thang nay|thang ni|th nay|thang hien tai)", t): return dm.this_month(today), "this_month(TODAY)"
    if re.search(r"(tu dau nam|dau nam (toi|den)|nam nay tinh tu|nam nay (?:tinh )?(?:toi|den) (?:het )?(?:hom nay|nay|bay gio)|tu ngay 1/1|tu 1/1|tu mung 1 tet duong)", t): return dm.ytd(today), "ytd(TODAY)"
    m = re.search(r"(\d+) ngay (gan nhat|qua|vua qua|gan day|lien tiep gan nhat|cuoi|gan nhut|vua roi)", t)
    if m: return dm.last_n_days(int(m.group(1)), today), f"last_n_days({int(m.group(1))}, TODAY)"
    if re.search(r"tuan (roi|truoc|vua roi|vua qua|qua|trc)", t): return dm.last_week(today), "last_week(TODAY)"
    if re.search(r"tuan nay", t): return dm.this_week(today), "this_week(TODAY)"
    t = _year_fix(t)
    t = re.sub(r"\bq\s?([1-4])\b", r"quy \1", t)                         # 'Q1/2026', 'q3 2025'
    m = re.search(r"\bthang (muoi mot|muoi hai|mot|gieng|hai|ba|tu|nam|sau|bay|tam|chin|muoi|chap)\b(?! nay| truoc| roi| sau| toi)", t)
    if m and not re.search(r"\bthang \d", t):                            # 'tháng ba' (spelled month, no year)
        mo = SPELLED_M[m.group(1)]; return dm.month_noyear(mo, today), f"month_noyear({mo}, TODAY)"
    m = re.search(r"quy (\d)\s*(?:/|-|nam |cua nam )\s*" + Y, t)
    if m: return dm.quarter(_y(m.group(2)), int(m.group(1))), f"quarter({_y(m.group(2))}, {int(m.group(1))})"
    m = re.search(r"quy (\d)\b", t)
    if m: return dm.quarter_noyear(int(m.group(1)), today), f"quarter_noyear({int(m.group(1))}, TODAY)"
    m = re.search(r"(?:ngay )?(\d{1,2})/(\d{1,2})/(20\d\d)", t)
    if m: d = _date(*m.group(1, 2, 3)); return dm.day(d), f"day({_dc(d)})"
    m = re.search(r"\b" + MON + r"(\d{1,2})\s*(?:/|-|nam |cua nam )\s*" + Y + r"\b", t) or re.search(r"(?<![\d/])(\d{1,2})[/-](20\d\d|\d\d)\b(?![/\d])", t)
    if m: return dm.month(_y(m.group(2)), int(m.group(1))), f"month({_y(m.group(2))}, {int(m.group(1))})"
    m = re.search(r"\bnam (20\d\d)\b,? (?:(?!den |toi |tu |cho )\S+ ){0,2}?(?:thang|thg|th) (\d{1,2})\b(?![/\d])", t)
    if m and 1 <= int(m.group(2)) <= 12:                               # the year before its month: 'năm 2025 tháng 5', 'Năm 2026, tháng 5'
        return dm.month(int(m.group(1)), int(m.group(2))), f"month({int(m.group(1))}, {int(m.group(2))})"
    m = re.search(r"\b(?:thang|thg|th)\s*(\d{1,2})\b(?!/)", t)
    if m: return dm.month_noyear(int(m.group(1)), today), f"month_noyear({int(m.group(1))}, TODAY)"
    if re.search(r"(hom qua|hqua|bua qua|ngay qua\b|hom truoc|ngay hom qua)", t): return dm.yesterday(today), "yesterday(TODAY)"
    if re.search(r"(hom nay|hnay|bua nay|ngay nay|trong ngay|cuoi ngay)", t): return dm.today_p(today), "today_p(TODAY)"
    return None, None


def _spread_year(t):
    """'th7, th8, th9/2026' / 'tháng 5, 6, 7/2025' -> every month carries the year written after the last one."""
    def rep(m):
        y = m.group(2); ms = re.findall(r"\d{1,2}", m.group(1))
        return ", ".join(f"{x}/{y}" for x in ms) + m.group(3)
    return re.sub(r"\b((?:(?:thang|thg|th|t)\s*\d{1,2}\s*(?:,|va|\s)\s*)+(?:(?:thang|thg|th|t)\s*)?\d{1,2})\s*/\s*(20\d\d)(\b)", rep,
                  re.sub(r"\b(thang|thg|th|t)\s*(\d{1,2}(?:\s*[,\s]\s*(?:va\s+)?\d{1,2})+)\s*/\s*(20\d\d)", lambda m: ", ".join(f"{x}/{m.group(3)}" for x in re.findall(r"\d{1,2}", m.group(2))), t))


SPELLED_M = {"mot": 1, "gieng": 1, "hai": 2, "ba": 3, "tu": 4, "nam": 5, "sau": 6, "bay": 7, "tam": 8, "chin": 9, "muoi": 10,
             "muoi mot": 11, "muoi hai": 12, "chap": 12}


def _year_fix(t):
    """month expressions of a key text made explicit: a two-digit year after a month ('thang 7/26' -> 'thang 7/2026', 't2/25'),
    a year written once after a list ('thang 10, 11, 12 nam 2025', 'thang 3, thang 4, thang 5 nam 2025') -> every month."""
    t = re.sub(r"\b((?:thang|thg|th|t)\s*\d{1,2})\s*([/-])\s*(\d\d)\b(?![/\d])", lambda m: f"{m.group(1)}{m.group(2)}20{m.group(3)}", t)
    def rep(m):
        ms = re.findall(r"\d{1,2}", m.group(1)); return ", ".join(f"{x}/{m.group(2)}" for x in ms)
    return re.sub(r"\b((?:(?:thang|thg|th|t)\s*)?\d{1,2}(?:\s*(?:,|va|hay|voi|qua|roi|den|\s)\s*(?:(?:thang|thg|th|t)\s*)?\d{1,2})+)\s+nam\s+(20\d\d)\b", rep, t)


def month_list(text, today=None):
    """the months a question lists, in order: explicit 'M/YYYY' (years spread / completed), else a year-less list
    'tháng 4, tháng 5, tháng 6' -> month_noyear codes (rulebook: this year's month unless after today)."""
    ms = all_months(text)
    if ms: return ms
    t = K(text)
    m = re.search(r"\bthang\s*(\d{1,2})((?:\s*(?:,|va|hay|voi|qua|roi)?\s*(?:thang\s*)?\d{1,2}\b(?![/.,]?\d))+)", t)
    if not m: return []
    nums = [m.group(1)] + re.findall(r"\d{1,2}", m.group(2))
    return [f"month_noyear({int(x)}, TODAY)" for x in nums if 1 <= int(x) <= 12]


def all_months(text):
    """every explicit month 'M/YYYY' of the text in order -> list of code strings."""
    t = _spread_year(_year_fix(K(text))); out = []
    t = re.sub(r"\b(?:q|quy)\s?[1-4]\s*[/-]\s*20\d\d", " ", t)            # 'Q1/2026' is a quarter, not January
    for m in re.finditer(r"(?<![\d/])(\d{1,2})\s*[/-]\s*(20\d\d)\b|thang (\d{1,2}) nam (20\d\d)\b", t):
        mo, y = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        out.append(f"month({int(y)}, {int(mo)})")                     # 'tháng 4 năm 2026' as well as '4/2026'
    return out


def months_pos(text):
    """[(start, year, month)] of every explicit month 'M/YYYY' / 'tháng M năm YYYY' of the key text, in order."""
    out = []
    for m in re.finditer(r"(?<![\d/])(\d{1,2})\s*[/-]\s*(20\d\d)\b|thang (\d{1,2}) nam (20\d\d)\b", text):
        mo, y = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        out.append((m.start(), int(y), int(mo)))
    return out


NUMW = {"hai": 2, "ba": 3, "bon": 4, "nam": 5, "sau": 6, "bay": 7, "tam": 8, "chin": 9, "muoi": 10}   # spelled counts ('top năm món')


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
        self.items_reg = {dm.regional_key(x): x for x in self.items.values()}   # regional words ('cá tràu' = 'cá lóc')
        self.items_reg_plain = {dm.regional_key_plain(x): x for x in self.items.values()}   # the same, typed without accents
        self.codes = {dm.key(x): x for x in vals(L, "item_code")}
        self.groups = {dm.key(x): x for x in vals(L, "item_group")}
        self.cgroups = {dm.key(x): x for x in vals(O, "customer_group")}
        self.staff = {dm.key(x): x for x in vals(O, "staff")}
        self.customers = {dm.key(x): x for x in vals(O, "customer_name")}
        self.ingredients = {dm.key(x): x for x in vals(L, "active_ingredient")}
        self.staff_names = {}                                            # a staff list's full names -> the code orders carry
        S_ = shop.C.get("staff")
        if S_ is not None and {"staff", "staff_name"} <= set(S_.columns):
            for code, nm in zip(S_["staff"], S_["staff_name"]):
                if isinstance(nm, str) and nm.strip() and code is not None: self.staff_names[dm.key(nm)] = code

    @staticmethod
    def find(t, table, min_len=3):
        """longest table key occurring as whole words in t (accent-free key of the question)."""
        best = None
        for k, v in table.items():
            if len(k) >= min_len and (best is None or len(k) > len(best[0])) and re.search(r"(^|\s)" + re.escape(k) + r"(\s|$)", t): best = (k, v)
        return best[1] if best else None


PAY_WORDS = [("cod", "cod"), ("thu ho", "cod"), ("tien mat", "tiền mặt"), ("tien tuoi", "tiền mặt"), ("chuyen khoan", "chuyển khoản"), ("vi dien tu", "ví điện tử"),
             ("ghi no", "ghi nợ"), ("the ngan hang", "thẻ"), ("quet the", "thẻ")]
CHAN_WORDS = [("zalo", "zalo"), ("dien thoai", "điện thoại"), ("tai quay", "tại quầy"), ("tai cua hang", "tại quầy"), ("san", "sàn TMĐT"),
              ("shopee", "sàn TMĐT"), ("facebook", "facebook"), ("website", "website")]
UNITS = r"(kg|chai|goi|hop|thung|bao|vien|lon|ly|cai|tuyp|vi|phan|dia|cuc|m3|m2|bo|binh|cay|cuon|loc|lo|ong|mieng|to|noi|con|lan|tam|met|m|ket|qua|bich|lit|lit)"


def J(x): return repr(x) if not isinstance(x, str) else '"' + x.replace('"', '\\"') + '"'


def fargs(f):
    return "".join(f", {k}={J(v)}" for k, v in f.items())


def ans(t, expr): return f'ANSWER = {{"type": "{t}", "answer": {expr}}}'


SALE_FILLER = set("""hang ra het tong cong tat ca cac le si duoc dc cho khach cua tiem quan shop minh em anh chi nha nhe ha vay
    roi ay nay do""".split())
SALE_STOP = set("duoc dc duco dduoc dk duowc bao bnhiu bn may la thang tuan quy nam hom ngay bua tu trong vao het tong luc gio".split())
CAP_STOP = set("""thang thg th quy tuan nam tet hom ngay bua doanh thu dthu top tong cho shop quan tiem cua hang ad chi anh em minh toi
    ban tu den trong vao dau cuoi bao nhieu binh quan trung so voi va hay la co mot moi cac nhung nay qua truoc sau ca sang chieu""".split()) - {"chi", "anh", "em", "minh", "toi"}
# (pronouns stay out of the stop list: 'Lương Chí Đức' is a name)


def unlinked_entity(q, t):
    """a sold thing / a name the shop vocabulary did not link: 'bán <X> được bao nhiêu', 'tiền bán <X>' with X not filler, or a
    run of capitalised words inside the question ('Canh chua cá tràu', 'Đoàn Bảo Phúc'). A shop-wide template would then answer
    for the whole shop instead of X."""
    for m in re.finditer(r"\b(?:tien ban|ban)\s+((?:\S+\s*){0,8})", t):
        words = []
        for w in m.group(1).split():
            if w in SALE_STOP or re.search(r"\d", w): break
            words.append(w)
        if [w for w in words if w not in SALE_FILLER]: return True
    toks = re.findall(r"[^\W\d_][\w\-]*", str(q)); run = []
    for i, w in enumerate(toks + ["."]):
        if w[:1].isupper() and not w.isupper() and dm.key(w) not in CAP_STOP and i < len(toks): run.append(i); continue
        if len(run) >= 2 and (run[0] > 0 or len(run) >= 3): return True
        run = []
    return False


def _kinds_of(shop):
    """which public-rulebook industry tables the shop's mapping has: 'repair' (tickets with work / plate), 'phone' (devices /
    instalments / trade-ins / warranty book), 'baby' (vouchers / points / members / brands)."""
    C, O = shop.C, shop.O; k = set()
    if any(c in O.columns and O[c].notna().any() for c in ("work", "plate", "vehicle")): k.add("repair")
    if any(C.get(x) is not None and len(C.get(x)) for x in ("instalments", "tradeins", "warranty")) or \
            ("tradein" in O.columns and O["tradein"].notna().any()) or (shop.L is not None and "imei" in shop.L.columns and shop.L["imei"].notna().any()):
        k.add("phone")
    if (C.get("customers") is not None and "registered" in C["customers"].columns) or "points_discount" in O.columns or \
            (C.get("items") is not None and "brand" in C["items"].columns):
        k.add("baby")
    return k


JOB_WORDS = r"(?:thay|va|sua|lam|rua|ve sinh|bao duong|chinh|can|tang)\s+\S+(?:\s+\S+)?|nhot\S*|lap\b|ruot\b|bugi|bu gi|curoa|sen\b"


_WD = {"hai": 0, "2": 0, "ba": 1, "3": 1, "tu": 2, "4": 2, "nam": 3, "5": 3, "sau": 4, "6": 4, "bay": 5, "7": 5}


def named_weekdays(q):
    """weekday indices (0 = Monday = thứ 2 ... 6 = Sunday = chủ nhật) a question names: Vietnamese numbers the days from
    thứ 2 (Monday), so 'thứ 3' is Tuesday = 1 (public dev_012, v25: every LLM sample took 'thứ 3' for weekday 2)."""
    k = dm.key(str(q)); out = {_WD[m] for m in re.findall(r"\bthu (hai|ba|tu|nam|sau|bay|[2-7])\b", k)}
    if re.search(r"\bchu nhat\b|\bcn\b", k): out.add(6)
    return out


def weekday_consistent(q, code):
    """False when a program filters / averages by a weekday index the question does not name (weekday_avg(p, 2) for
    'thứ 3'); True when the question names no weekday or the program uses none."""
    wd = named_weekdays(q)
    if not wd or not code: return True
    used = {int(x) for x in re.findall(r"weekday_avg\([^()]*?(?:\([^()]*\)[^()]*?)*?,\s*(?:weekday\s*=\s*)?(\d)", code)}
    used |= {int(x) for x in re.findall(r"(?:weekday|dayofweek)\s*(?:==|in\s*[\[(])\s*(\d)", code)}
    return not used or used <= wd


def named_vehicle(q):
    """the vehicle model a repair-shop question names after 'xe' / 'chiếc' in capitals ('bao nhiêu xe Air Blade khác nhau',
    'bn chiec SH Mode khac nhau', 'chiếc xe Wave Alpha'), else None."""
    mv = re.search(r"\b(?:xe|chi[eế]c)\s+((?:[A-Z][\w\-]*|\d{2,3}\b)(?:\s+(?:[A-Z][\w\-]*|\d{2,3}\b))*)", str(q))
    if mv and not re.match(r"(?:Th|T|Q)\d", mv.group(1)): return mv.group(1).strip()
    return None


def parse_industry(q, t, tk, P, p, today, shop, has, kinds):
    """templates of the public rulebook's industries (src/industry.py toolkit); None = not one of them."""
    import industry
    d = until_day(q, today)
    O = shop.O
    def name_in(table_vals, min_len=3):
        best = None
        for v in table_vals:
            k = dm.key(v)
            if len(k) >= min_len and re.search(r"(^|\s)" + re.escape(k) + r"(\s|$)", tk) and (best is None or len(k) > len(dm.key(best))): best = v
        return best
    if "repair" in kinds:
        S_ = shop.C.get("staff"); full = list(S_["staff"].dropna()) if S_ is not None and "staff" in S_.columns else []
        staff = name_in(full) or (name_in([x for x in O["staff"].dropna().unique()]) if "staff" in O.columns else None)
        veh = None
        if "vehicle" in O.columns:
            lab = industry.vehicle_lines(O["vehicle"]); veh = name_in(list(lab), 2)
        if veh is None:      # the question names a model ('bao nhiêu xe Air Blade khác nhau') that this mapping cannot see (no
            veh = named_vehicle(q)                     # orders.vehicle): keep the filter (public dev_012, v25: 368 Air Blades = every vehicle)
        sf = f", staff={J(staff)}" if staff else ""
        if has(r"hoa hong|hue hong") and P and S_ is not None and "commission_pct" in S_.columns:
            if staff: return "ind_commission", ans("number", f"commission({P}, staff={J(staff)})")
            if has(r"bang|tung tho|moi tho|cac tho|xep"): return "ind_commission_table", ans("table", f"commission({P})")
            if has(r"tho nao|ai|tho mo") and has(r"cao nhat|nhieu nhat|nhat"): return "ind_commission_top", ans("string", f"commission({P})['rows'][0][0]")
            return None
        if has(r"khach quen|tu (\d+) lan|(\d+) lan tro len") and has(r"\bxe\b") and P and "plate" in O.columns:
            m = re.search(r"(?:tu |it nhat )?(\d+) lan", t); n = int(m.group(1)) if m else 2
            return "ind_regular_vehicles", ans("number", f"n_vehicles({P}, min_paid={n})")
        if has(r"xe .*khac nhau|chiec .*khac nhau|chiec xe|bao nhieu xe|may xe|bnhiu xe|bn xe|moi xe tinh|(?:bao nhieu|may|bnhiu|bn) chiec") and P and not has(r"luot|phieu") \
                and "plate" in O.columns:                                   # a named model on a mapping without orders.vehicle:
                                                                                # unanswerable there -> the vehicle topic moves it
            return "ind_vehicles", ans("number", f"n_vehicles({P}" + (f", vehicle={J(veh)}" if veh else "") + ")")
        if has(r"dong xe") and has(r"nhieu tien|doanh thu|dthu") and has(r"nhat") and P:
            return "ind_top_vehicle", ans("string", f"top_vehicles({P}, 1, by='revenue')[0]")
        mw = re.search(r"\b(thu (?:hai|ba|tu|nam|sau|bay|[2-7])|chu nhat)\b", t)
        if mw and has(r"trung binh|binh quan") and has(r"luot|phieu|\bxe\b") and P and not has(r"tien|doanh thu|dthu"):
            wd = {"thu hai": 0, "thu 2": 0, "thu ba": 1, "thu 3": 1, "thu tu": 2, "thu 4": 2, "thu nam": 3, "thu 5": 3, "thu sau": 4, "thu 6": 4,
                  "thu bay": 5, "thu 7": 5, "chu nhat": 6}.get(mw.group(1))
            if wd is not None: return "ind_weekday_tickets", ans("number", f"weekday_avg({P}, {wd}, by='orders')")
        if has(r"trung binh|binh quan") and has(r"phieu") and P:                    # 'không tính xe bảo hành, bình quân 1 phiếu'
            return "ind_avg_ticket", ans("number", f"avg_ticket({P}{sf})")
        if has(r"bao hanh") and P and ("warranty" in O.columns or "note" in O.columns):
            if has(r"%|phan tram|ti le|ty le|chiem"): return "ind_warranty_rate", ans("percent", f"warranty_rate({P})")
            if has(r"bao nhieu|may|bnhiu|\bbn\b|so") : return "ind_warranty_visits", ans("number", f"warranty_visits({P})")
        if has(r"trung binh|binh quan") and has(r"phieu") and has(r"tien") and P and not has(r"thu (?:hai|ba|tu|nam|sau|bay|\d)|chu nhat"):
            return "ind_avg_ticket", ans("number", f"avg_ticket({P}{sf})")
        if has(r"tho (?:nao|mo)|\bai\b|tho .*nhat") and has(r"nhat") and P and not has(r"hoa hong"):
            by = "n_days" if has(r"cham chi|cham nhat|sieng") else "revenue" if has(r"tien|doanh thu|dthu") else "tickets" if has(r"luot|phieu|xe") else None
            if by: return "ind_top_staff", ans("string", f"top_staff({P}, 1, by={J(by)})[0]")
        if staff and P and has(r"luot|phieu|\bxe\b") and not has(r"tien|doanh thu|dthu"):
            return "ind_staff_tickets", ans("number", f"n_tickets({P}, staff={J(staff)})")
        if staff and P and has(r"tien|doanh thu|dthu"): return "ind_staff_revenue", ans("number", f"revenue({P}, staff={J(staff)})")
        m = re.search(r"(?:luot|phieu)(?: xe)?(?: sua)?(?: co)?\s+(" + JOB_WORDS + r")", t) or re.search(r"(" + JOB_WORDS + r")\s+(?:cho )?(?:bao nhieu|may|bnhiu|bn) (?:luot|phieu)", t)
        if m and P and not has(r"trung binh|binh quan"):
            job = industry.job_query(m.group(1))
            if job: return "ind_job", ans("number", f"job_tickets({P}, {J(job)})")
        if has(r"(?:bao nhieu|may|bnhiu|bn|dem) (?:luot|phieu)") and P and not has(r"trung binh|binh quan|thu (?:hai|ba|tu|nam|sau|bay)"):
            return "ind_tickets", ans("number", f"n_tickets({P})")
        return None
    if "phone" in kinds:
        I = shop.C.get("instalments")
        cust_vals = list(I["customer_name"].dropna().unique()) if I is not None and "customer_name" in I.columns else []
        cust = name_in(cust_vals)
        cf = f", customer={J(cust)}" if cust else ""
        has_inst = I is not None and len(I) > 0
        W_ = shop.C.get("warranty"); has_w = W_ is not None and len(W_) > 0
        has_ti = ("tradein" in O.columns and O["tradein"].notna().any()) or (shop.C.get("tradeins") is not None and len(shop.C["tradeins"]))
        if has(r"tra gop|\bgop\b|ky gop|tien gop") and has_inst:
            if has(r"toi han|den han") and has(r"may ky|bao nhieu ky|so ky") and cust: return "ind_terms_due", ans("number", f"terms_due({d}, customer={J(cust)})")
            if has(r"(?:may|bao nhieu|bnhiu|bn|so) (?:khach|nguoi)") and has(r"qua han|tre han|cham"): return "ind_overdue_customers", ans("number", f"n_overdue_customers({d})")
            asks_money = has(r"tien|bao nhieu|bnhiu|\bbn\b|\btong\b|gia tri")   # an amount is asked in money words; without them
            if asks_money and has(r"qua han|tre han|cham gop|cham dong"):      # the question lost its head ('ko mi ngui' = 'có mấy
                return "ind_inst_overdue", ans("number", f"instalment_overdue({d}{cf})")   # người'): no rule answer
            if asks_money and has(r"con lai|con thieu|chua gop|chua tra|con no"):
                return "ind_inst_remaining", ans("number", f"instalment_remaining({d}{cf})")
            return None
        if has_ti and has(r"(?<!doanh )\bthu (?:lai |mua |vao |duoc )?may(?: cu)?\b|tien may cu|may cu (?:thu|tru)|thu cu doi moi") and \
                not has(r"ban (?:ra |duoc |dc )?(?:bao nhieu|may|bnhiu|bn) (?:may|chiec)|dien thoai cu|trung binh moi may|gia bao nhieu|moi may gia"):
            if has(r"khach phai tra|phai tra"):
                if has(r"trung binh|binh quan|moi don|mot don|1 don"): return "ind_avg_payable", ans("number", f"avg_payable({P})") if P else None
                return "ind_payable", ans("number", f"payable_total({P})") if P else None
            if has(r"%|phan tram|ti le|chiem") and has(r"don"): return "ind_tradein_share", ans("percent", f"tradein_share({P})") if P else None
            if has(r"tien|bao nhieu|bnhiu|cong") and P: return "ind_tradein", ans("number", f"tradein_total({P})")
        if has(r"khach phai tra|phai tra") and P and not has(r"\bno\b") and "total_payable" in O.columns and O["total_payable"].notna().any():
            if has(r"trung binh|binh quan|moi don|mot don|1 don"): return "ind_avg_payable", ans("number", f"avg_payable({P})")
            return "ind_payable", ans("number", f"payable_total({P})")
        if has(r"bao hanh") and P and has_w:
            if has(r"%|phan tram|ti le|chiem|so voi"): return "ind_warranty_rate", ans("percent", f"warranty_rate({P})")
            if has(r"dong may|may nao") and has(r"nhieu|nhat"): return "ind_top_warranty", ans("string", f"top_warranty_lines({P}, 1)[0]")
            if has(r"may ban") and has(r"(?:bao nhieu|may|bnhiu|bn) (?:chiec )?may"): return "ind_devices_warranty", ans("number", f"devices_with_warranty({P}, {d})")
            if has(r"luot|bao nhieu|may|bnhiu"): return "ind_warranty_visits", ans("number", f"warranty_visits({P})")
            return None
        dev = has(r"\bmay\b(?! cu thu)|chiec|dien thoai") and P
        kind = "máy cũ" if has(r"may cu|dien thoai cu|\bcu\b") and not has(r"ca may moi lan may cu|gop ca may moi|moi lan cu|moi va cu|moi voi may cu") else \
               "máy mới" if has(r"may moi") and not has(r"ca may moi|moi lan|moi va cu|moi voi") else None
        kf = f", kind={J(kind)}" if kind else ""
        if dev and has(r"dong may") and has(r"nhieu|nhat|top"):
            m = re.search(r"(\d+) dong may", t); k = int(m.group(1)) if m and int(m.group(1)) <= 20 else 1
            return ("ind_top_lines", ans("list", f"top_lines({P}, {k}, by='qty'{kf})")) if k > 1 else ("ind_top_line", ans("string", f"top_lines({P}, 1, by='qty'{kf})[0]"))
        if dev and has(r"%|phan tram|chiem|ti le"):
            brand = next((b for b in ("Samsung", "Apple", "iPhone", "Xiaomi", "Oppo", "Vivo", "Realme", "Nokia", "Honor", "Huawei") if re.search(r"\b" + dm.key(b) + r"\b", tk)), None)
            by = "qty" if has(r"so may|tong so may|bao nhieu may|chiec") else "revenue"
            if brand: return "ind_device_share", ans("percent", f"device_share({P}, brand={J(brand)}, by={J(by)})")
            if kind: return "ind_device_share", ans("percent", f"device_share({P}, kind={J(kind)}, by={J(by)})")
            return None
        if dev and has(r"trung binh|binh quan") and has(r"gia|tien"): return "ind_avg_device", ans("number", f"avg_device_price({P}{kf})")
        if dev and has(r"mat hang nao|san pham nao|hang nao|may nao") and has(r"nhieu (?:may|chiec)|nhat") and not has(r"tien|doanh thu|dthu|bao hanh"):
            return "ind_top_device_item", ans("string", f"top_lines({P}, 1, by='qty', level='item'{kf})[0]")
        if dev and has(r"(?:bao nhieu|may|bnhiu|bn|so) (?:may|chiec|dien thoai)") and not has(r"tien|doanh thu|dthu"):
            lines = sorted({industry.model_line(n) for n in (shop.L["item_name"].dropna().unique() if shop.L is not None else []) if industry.model_line(n)}, key=len, reverse=True)
            line = name_in(lines)
            if line: return "ind_devices_line", ans("number", f"devices_sold({P}, line={J(line)}{kf})")
            V = Vocab(shop); it = V.find(tk, V.items)
            if it is not None: return None                                 # one item: the item templates
            if re.search(r"\b(iphone|ip|samsung|galaxy|xiaomi|redmi|oppo|vivo|realme|nokia|honor|huawei|pixel|tecno|infinix)\b", tk):
                return None                  # a model / brand this mapping's names do not show: not every device (dev_012-like miss)
            return "ind_devices", ans("number", f"devices_sold({P}{kf})")
        return None
    if "baby" in kinds and P:
        if has(r"voucher|vau cho|vâu chơ|ma giam") and has(r"giam|tru|tien|bao nhieu"): return "ind_voucher", ans("number", f"voucher_total({P})")
        if has(r"\bdiem\b") and has(r"tru|giam|dung|doi"): return "ind_points", ans("number", f"points_total({P})")
        if has(r"phi ship|phi van chuyen|phi giao|tien ship"): return "ind_shipping", ans("number", f"shipping_total({P})")
        if has(r"thanh vien|dang ky") and has(r"moi"): return "ind_new_members", ans("number", f"new_members({P})")
        szm = re.search(r"\b(?:size|co) (nb|s|m|l|xl|xxl|xxxl)\b", t); som = re.search(r"\bso (\d)\b", t)
        size = szm.group(1).upper() if szm else (f"Số {som.group(1)}" if som and has(r"sua") else None)
        it = shop.C.get("items"); brands = list(it["brand"].dropna().unique()) if it is not None and "brand" in it.columns else []
        brand = name_in(brands, 2)
        if has(r"bich|goi") and has(r"bim|ta\b") and has(r"%|phan tram|chiem|ti le") and (size or brand):
            return "ind_pack_share", ans("percent", f"pack_share({P}" + (f", size={J(size)}" if size else "") + (f", brand={J(brand)}" if brand else "") + ")")
        if (size or brand) and has(r"\bbim\b|\bta\b") and has(r"(?:bao nhieu|may|bnhiu|bn|tong cong|tong) (?:bich|goi)") and not has(r"%|phan tram|chiem|ti le|tien|doanh thu"):
            return "ind_diaper_qty", ans("number", f"diaper_qty({P}" + (f", size={J(size)}" if size else "") + (f", brand={J(brand)}" if brand else "") + ")")
        if has(r"hang nao|hang gi") and has(r"nhieu|nhat") and size:
            g = Vocab(shop).find(tk, Vocab(shop).groups); gf = f", group={J(g)}" if g else ""
            return "ind_top_brand", ans("string", f"top_brands({P}, 1, by='qty', size={J(size)}{gf})[0]")
        if size and size.startswith("Số") and has(r"(?:bao nhieu|may|bnhiu|bn) (?:lon|hop)") and not has(r"tien|doanh thu"):
            g = Vocab(shop).find(tk, Vocab(shop).groups); gf = f", group={J(g)}" if g else ""
            return "ind_number_qty", ans("number", f"size_qty({P}, {J(size)}{gf})")
    return None


# qualities a shop's data never records (rulebook §7: unanswerable) unless the rulebook defines them; the rulebook's own ranking
# terms (trung thành, sộp, hay đổi ý, ưa chuộng, chăm chỉ, đắt khách, chính, bán giỏi, bán chạy, lời ...) are never in this list
SUBJECTIVE = r"\b(ngon|than thien|kho tinh|de tinh|uy tin|dang tin cay|tin cay|chat luong|trung thuc|dep|nhiet tinh|vui ve|lich su|de thuong|" \
             r"hai long|tu te|chu dao|sach se|nhanh nhen|dang yeu|tan tam|chuyen nghiep|tot bung|de chiu|kho chiu|dep trai|xinh|thoai mai|" \
             r"lanh nghe|tay nghe|gioi nghe|le phep|quy men|yeu men|men mo|duoc long|cham soc khach|hieu y|khen)\b"


def subjective(t):
    """a 'which ... most <quality>' question about a quality no data column records ('sản phẩm nào ngon nhất', 'nhà cung cấp uy
    tín', 'khách thân thiện nhất', 'thợ nhiệt tình nhất'): True when it names such a quality and asks for the top one."""
    if re.search(SUBJECTIVE + r"\s+(?:\w+\s+)?(?:nhat|nhut)\b", t): return True        # '<quality> (tốt) nhất' ('Siro ăn ngon ...
    if re.search(SUBJECTIVE + r"(?:\s+\w+){0,4}\s+(?:khong|ko|hong)\b", t) and re.search(r"\bco\b", t): return True   # 'có tận tâm … không'
    if re.search(r"\b(?:hay|duoc) khen\b|\bkhen (?:\w+ )?(?:nao|ai)\b", t): return True                              # praise is not recorded
    m = re.search(r"\b(uy tin|dang tin cay|tin cay)\b", t)                                  # gần hết hạn nhất' is a name)
    return bool(m and re.search(r"\b(la ai|nao|ai)\b", t))


def _has_col(shop, *cols, tables=("orders", "lines", "items", "customers", "returns")):
    for tn in tables:
        df = shop.O if tn == "orders" else shop.L if tn == "lines" else shop.C.get(tn)
        if df is not None and any(c in df.columns and df[c].notna().any() for c in cols): return True
    return False


def missing_info(q, t, shop):
    """rulebook §7: the question needs a kind of information the shop's data does not record -> the reason, else None.
    Only kinds the canonical tables name explicitly (customer gender / address, reasons, item groups, time of day) and
    questions outside the data (medical advice)."""
    ql = unicodedata.normalize("NFC", str(q)).lower()
    if re.search(r"\bkhách (?:hàng )?(?:nữ|nam)\b(?!\s*\d)|\bgiới tính\b|\b(?:phái|giới) (?:nữ|nam)\b", ql) or re.search(r"\bkhach (?:hang )?nu\b", t):
        if not _has_col(shop, "gender", "customer_gender"): return "gender"
    if re.search(r"\b(?:ở|tại|o|tai) (?:đường|phố|quận|phường|duong|pho|quan|phuong)\b|\bđịa chỉ\b|\bdia chi\b|\bkhách (?:ở|đường)\b", ql):
        if not _has_col(shop, "address", "street", "district"): return "address"
    if re.search(r"\bvì sao\b|\blý do\b|\bnguyên nhân\b|\bvi sao\b|\bly do\b|\bnguyen nhan\b", ql) or \
            re.search(r"\b(?:huỷ|hủy|trả lại|trả món|trả hàng|trả)\b[^?]{0,30}\bvì\b(?! ví| điện tử)", ql):
        if not _has_col(shop, "reason", "cancel_reason", "return_reason"): return "reason"
    if re.search(r"\bnhom (?!khach|tuoi|nguoi|thanh vien|ngay|no\b)(?:hang |mon |san pham |sp )?\w", t) and not re.search(r"\bnhom (?:khach|tuoi)", t):
        if not _has_col(shop, "item_group", tables=("lines", "items")): return "item group"
    if re.search(r"\bmấy giờ\b|\bgiờ nào\b|\bkhung giờ\b|\bmay gio\b|\bgio nao\b|\bkhung gio\b|\bgiờ cao điểm\b", ql):
        ot = shop.O["order_time"] if "order_time" in shop.O.columns else None
        tt = pd.to_datetime(ot, errors="coerce").dropna() if ot is not None else pd.Series(dtype="datetime64[ns]")
        if not len(tt) or ((tt.dt.hour == 0) & (tt.dt.minute == 0)).mean() > 0.98: return "time of day"
    if re.search(r"\b(?:bà bầu|mẹ bầu|phụ nữ có thai|người đang mang thai|trẻ sơ sinh|trẻ em|em bé)\b.{0,40}\b(?:dùng|uống|ăn|xài)\b.{0,40}\b(?:được không|được ko|có được|có sao không|an toàn)", ql) \
            or re.search(r"\b(?:nên|phải)\s+(?:uống|dùng|bôi|xịt)\b.{0,30}\b(?:mỗi lần|một lần|mấy viên|bao nhiêu viên|liều|mấy lần)", ql) \
            or re.search(r"\b(?:nen|phai) (?:uong|dung) .{0,30}\b(?:moi lan|may vien|bao nhieu vien|lieu)\b", t):
        return "medical advice"
    if re.search(r"\bnhom khach|\bkhach (?:hang )?(?:nhom|thuoc nhom)\b|\bkhach (?:hang )?(?:than thiet|vip|si)\b", t) and not _has_col(shop, "customer_group", tables=("orders", "customers")):
        return "customer group"
    return None


_MONEY_Q = re.compile(r"(?<![\d/])(\d{1,3}(?:[.,]\d{3})+|\d+(?:[.,]\d+)?)\s*(?:d|dong|vnd|k|nghin|ngan|trieu|tr|cu)\b|(?<![\d/.,])(\d{1,3}(?:[.,]\d{3}){2,})(?![\d/])")
_NO_MONEY_TPL = ("price_effect", "price_on", "n_days_rev_ge", "n_orders_rev_ge", "subjective", "unrecorded", "price_change")


def _amounts(code):
    return {float(x) for x in re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w])", code)}


_ENT_NOUN = r"(?:mat hang|mon|san pham|sp|hang|khach(?: hang)?|nha cung cap|ncc|dong (?:may|san pham|hang)|ngay|thang|nhan vien|tho|nhom(?: hang)?|loai)"
_COUNT_CALL = re.compile(r"\b(?:n_\w+|len)\(")


def answer_kind_ok(t, tpl, code, raw=None):
    """the kind of answer the question asks (which one / how many / what share / how much money / which period) against the
    kind the program gives. A conflict means the reading missed the question's head (paraphrases of the public questions,
    k=0 split: 11 of 74 wrong answers caught, no right one lost; no canonical train / public answer is affected):
    'mặt hàng nào ... nhất' answered with a number, 'tỷ lệ / %' with a count, 'bao nhiêu tiền' with a count, 'bao nhiêu
    mặt hàng' with a name, 'tính ngược lại 30 ngày' with today, 'bao nhiêu phần' with revenue, revenue without any word
    of selling, 'thu nhập' read as 'nhập hàng', 'doanh thu chia cho số đơn' read as a total."""
    m = re.search(r'"type": "(\w+)"', code or ""); ty = m.group(1) if m else None
    count_cue = re.search(r"\b(?:bao nhieu|may|bnhiu|bn|so luong|dem)\b", t)
    if ty in ("number", "percent") and re.search(r"\b" + _ENT_NOUN + r"\s+nao\b|\bla ai\b|\bai la\b|\bai (?:\w+ ){0,4}nhat\b", t) \
            and not count_cue and not re.search(r"\bco\b.*\bnao\b.*\b(?:khong|ko|chua)\b", t): return False
    if ty == "number" and re.search(r"\b(?:ty le|ti le|phan tram)\b|(?<![\d.,])\s*%", t) and not re.search(r"\bbao nhieu tien\b", t): return False
    money_ask = re.search(r"\bbao nhieu tien\b", t) or (re.search(r"\btong (?:so )?tien\b(?! (?:hang )?(?:tu|tren|duoi|it nhat|hon|tro len))", t)
                                                       and re.search(r"\bbao nhieu\b", t))
    if ty == "number" and _COUNT_CALL.search(code or "") and money_ask and not re.search(r"\b(?:bao nhieu|may|bnhiu)\s+" + _ENT_NOUN + r"\b", t):
        return False
    if ty == "string" and re.search(r"\b(?:bao nhieu|may|bnhiu)\s+" + _ENT_NOUN + r"\b", t) \
            and not re.search(r"\b" + _ENT_NOUN + r"\s+nao\b|\bla ai\b|\bla gi\b", t): return False
    if "today_p(" in (code or "") and re.search(r"\b\d+\s*ngay\b|\bdau (?:nam|thang|tuan)\b|\btro lai day\b|\bnguoc lai\b|\bbat dau\b", t): return False
    if tpl == "revenue" and re.search(r"\b(?:bao nhieu|may|bnhiu)\s+" + UNITS + r"\b", t) and not re.search(r"\btien\b|doanh thu|dthu", t): return False
    if tpl == "revenue" and not re.search(r"\b(?:ban|doanh thu|dthu|doanh so|thu|kiem|bill|tien hang|duoc|revenue|sales)\b", t): return False
    for x in (t, raw or t):                                              # raw: before the slang ('thu nhập từ' -> 'doanh thu của')
        if (tpl or "").startswith("purchase") and re.search(r"\bthu nhap\b", x) and not re.search(r"(?<!thu )\bnhap\b", x): return False
    if re.search(r"\bchia (?:cho|deu|trung binh)\b", t) and tpl in ("n_orders", "revenue", "n_customers", "purchase_total"): return False
    return True


def consistent(q, t, tpl, code):
    """precision guard: a rule answer is kept only when it uses what the question states. A money amount written in the
    question that the program never uses ('đạt từ 40.000.000 đồng chở lên' read as plain revenue), a count of days / customers /
    orders answered with money, are misreadings (typing noise hid the qualifier): no rule answer (the LLM or abstain decides)."""
    if tpl not in ("subjective", "unrecorded") and not answer_kind_ok(t, tpl, code, K(q)): return False
    if tpl in _NO_MONEY_TPL or tpl.startswith("ind_"): return True
    for m in _MONEY_Q.finditer(t):
        raw = m.group(1) or m.group(2); unit = m.group(0)[len(raw):].strip()
        v = float(re.sub(r"[.,]", "", raw)) if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", raw) else float(raw.replace(",", "."))
        mult = 1e6 if unit.startswith(("trieu", "tr", "cu")) else 1e3 if unit.startswith(("k", "nghin", "ngan")) else 1
        val = v * mult
        if val < 1000: continue                                          # '3 lon', '2 đợt': not money
        if not any(abs(a - val) < 0.5 or abs(a - v) < 0.5 for a in _amounts(code)): return False
    if tpl == "revenue" and re.search(r"\b(?:moi|mot|1|tung) (?:don|bill|hoa don|phieu)\b|\b(?:binh quan|trung binh|chung binh)\b", t):
        return False                                                     # an average per order read as the total
    money_tpl = tpl in ("revenue", "aov", "purchase_total", "gross_profit", "net_profit", "debt_total", "stock_value", "share", "rev_pair", "months_list")
    tp = dm.strip_accents(unicodedata.normalize("NFC", str(q)).lower())    # punctuation kept: 'bao nhiêu, đơn qua …' is two clauses
    if money_tpl and re.search(r"\b(?:may|bao nhieu|bnhiu|bn|so|dem) (?:ngay|bua|hom|khach|nguoi|don|lan|luot|phieu|mat hang|mon|san pham)\b", tp) \
            and not re.search(r"\bbao nhieu tien\b|\bdoanh thu\b.{0,20}\b(?:moi|tung) ngay\b", tp):
        return False
    return True


def names_used(q, code, shop, tpl=""):
    """every person the question names (a staff member's full name or code, a customer of >= 2 words) and the item it names
    must appear in the program: a shop-wide answer to 'thợ Đoàn Bảo Phúc sửa được bao nhiêu xe' or 'bao nhiêu lượt thay dầu
    láp' answers another question."""
    V = getattr(shop, "_qp_vocab", None)
    if V is None:
        V = Vocab(shop)
        try: shop._qp_vocab = V
        except Exception: pass
    tk = dm.key(q); code_k = dm.key(code or "")
    def named(k): return len(k) >= 6 and re.search(r"(^|\s)" + re.escape(k) + r"(\s|$)", tk)
    for k, sc in V.staff_names.items():                                 # a staff list: the program filters on the code or the name
        if named(k) and k not in code_k and dm.key(str(sc)) not in code_k: return False
    for k, v in V.staff.items():
        if len(k.split()) >= 2 and named(k) and k not in code_k: return False
    for k, v in V.customers.items():
        if len(k.split()) >= 2 and not dm.is_walkin(v) and named(k) and k not in code_k: return False
    if tpl.startswith("ind_"): return True                              # industry books name their own things (tasks: 'sửa xe')
    item = V.find(tk, V.items, min_len=6)
    if item is not None and len(dm.key(item).split()) >= 2 and dm.key(item) not in code_k and "item" not in (code or ""): return False
    return True


def parse(q, today, shop=None):
    """(template, code) for the common templates; None when unsure (also when the precision guard rejects the reading)."""
    r = _parse(q, today, shop)
    if r is None: return None
    t = untelex(K(q))
    if not consistent(q, t, r[0], r[1]): return None
    if shop is not None and r[0] not in ("subjective", "unrecorded"):
        try:
            if not names_used(q, r[1], shop, r[0]): return None
        except Exception: pass
    return r


def _parse(q, today, shop=None):
    """(template, code) for the common templates; None when unsure."""
    t0 = K(q); t = untelex(t0); tk = dm.key(q)                                    # Telex typos undone first ('quys 3/2025')
    if re.search(r"(?i)(?<!\w)[oô]\s?-?\s?đờ(?!\w)", str(q)): t = re.sub(r"\bo ?do\b", "order", t)   # 'o đờ' = order ('ở đó' keeps its accents)
    p, P = parse_period(t, today)
    if P is None: p, P = parse_period(q, today)
    def has(*pats): return any(re.search(x, t) or re.search(x, t0) for x in pats)
    if subjective(t) or subjective(t0): return "subjective", ans("unanswerable", "None")
    if shop is not None:
        why = missing_info(q, t, shop)
        if why: return "unrecorded", ans("unanswerable", "None")
    if shop is not None:                                                # the public rulebook's industries (their own tables)
        kinds = _kinds_of(shop)
        if kinds:
            r = parse_industry(q, t, tk, P, p, today, shop, has, kinds)
            if r is not None: return r
            if "repair" in kinds and has(r"luot|phieu|\btho\b|\bxe\b") and not has(r"doanh thu|dthu|bao nhieu tien|bnhiu tien|tien hang"): return None
    V = Vocab(shop) if shop is not None else None
    f = {}; item = None
    if V is not None:
        item = V.find(tk, V.items) or V.find(tk, V.codes) or V.find(dm.regional_key(q), V.items_reg) or V.find(dm.regional_key_plain(q), V.items_reg_plain, min_len=8)
        if item is not None: f["item"] = item
        m = re.search(r"nhom (?:hang |mon |san pham )?(.+)", tk)
        if m and item is None:
            cg = V.find(m.group(1), V.cgroups, min_len=2)
            if has(r"khach") and cg is not None: f["customer_group"] = cg
            else:
                g = V.find(m.group(1), V.groups)
                if g is not None: f["group"] = g
        if item is None and "group" not in f and "customer_group" not in f and V.groups:
            ws = tk.split()
            for i, w in enumerate(ws):                                   # the exact name of a group right after 'món' / 'loại' /
                if w in ("mon", "loai", "cac", "dong") or (w == "hang" and i > 0 and ws[i - 1] == "mat"):   # 'các' / 'mặt hàng'
                    for n in (4, 3, 2, 1):
                        cand = " ".join(ws[i + 1:i + 1 + n])
                        if len(cand) >= 4 and cand in V.groups: f["group"] = V.groups[cand]; break
                    if "group" in f: break
        m = re.search(r"(?:nhan vien|nguoi ban|nv)(?: ma)? (nv\s*\d+)", tk) or re.search(r"(?<!\w)(nv\s?\d{2,})(?!\w)", tk)
        if m: f["staff"] = V.find(m.group(1).replace(" ", ""), V.staff) or m.group(1).upper().replace(" ", "")
        m = re.search(r"hoat chat (.+)", tk)
        if m and V.ingredients:
            ing = V.find(m.group(1), V.ingredients)
            if ing is not None: f["where"] = f"lambda d: d['active_ingredient'].map(key) == key({J(ing)})"
    def _mask(txt, name):
        if not isinstance(name, str): return txt
        k = dm.key(name)
        if len(k) < 4: return txt
        out = txt
        for kk in dict.fromkeys((k, untelex(k), K(name), untelex(K(name)))):   # the text went through K / untelex ('cheese' -> 'chese')
            out = re.sub(r"(^|\s)" + re.escape(kk) + r"(\s|$)", r"\1qxq\2", out)
        if out == txt:                                                  # matched through regional words: the span of txt
            ws, kk = txt.split(), dm.regional_key(name)
            n = len(k.split())
            for a in range(len(ws)):
                for b in (a + n - 1, a + n, a + n + 1):
                    if 0 < b <= len(ws) and dm.regional_key(" ".join(ws[a:b])) == kk:
                        return " ".join(ws[:a] + ["qxq"] + ws[b:])
        return out
    t_unmasked = t
    for _v in (item, f.get("group"), f.get("customer_group")):
        t = _mask(t, _v); t0 = _mask(t0, _v)
    pays = [lab for w, lab in PAY_WORDS if re.search(r"\b" + w + r"\b", t)]
    chans = [lab for w, lab in CHAN_WORDS if re.search(r"(qua|ban|tai|kenh|dat|tren)\s+(?:\w+\s+)?" + w + r"\b(?! pham)", t) or (w in ("zalo", "shopee") and w in t)
             or (w in ("tai quay", "tai cua hang") and re.search(r"\b" + w + r"\b", t))]
    if re.search(r"\btai tiem\b", t) and "tại quầy" not in chans: chans.append("tại quầy")
    if re.search(r"\bsan thuong mai dien tu\b|\bsan tmdt\b", t) and "sàn TMĐT" not in chans: chans.append("sàn TMĐT")
    chans = list(dict.fromkeys(chans))
    if isinstance(item, str):
        ik_ = dm.regional_key(item) + " " + dm.key(item)
        chans = [c for c in chans if not any(re.search(r"\b" + w + r"\b", ik_) for w, lab in CHAN_WORDS if lab == c)]
    if has(r"ca sang") and not has(r"ca chieu"): f["shift"] = "sáng"
    elif has(r"ca chieu") and not has(r"ca sang"): f["shift"] = "chiều"
    if has(r"ke don|can don bac si|theo don"):
        neg = bool(re.search(r"khong ke don|khong can don|tru thuoc ke don|tru .*ke don|khong theo don|khong phai thuoc ke don", t))
        f["where"] = "lambda d: ~d['prescription'].fillna(False).astype(bool)" if neg else "lambda d: d['prescription'].fillna(False).astype(bool)"
    fa = lambda d=None: fargs({k: v for k, v in (d if d is not None else f).items() if k != "where"}) + (f", where={f['where']}" if "where" in (d if d is not None else f) else "")
    rev = has(r"doanh thu|dthu|revenue|ban duoc .*tien|ban dc .*tien|thu ve|tien hang|tien ban|bao nhieu tien|bnhiu tien|bn tien|nhiu tien|nhieu tien|mang ve|thu duoc")
    two_part = len(re.findall(r"bao nhieu|bnhiu|\bbn\b|\bla nhieu\b", t)) >= 2 or len(re.findall(r"doanh thu|dthu", t)) >= 2 or has(r"moi loai")
    # ^ two quantities asked ('X là bao nhiêu, Y là bao nhiêu', 'mỗi loại'): no single-number template answers it
    debt = has(r"\bno\b|cong no|tra no|mua chiu|ban chiu|chua tra|thu no|ghi no|con thieu|thieu tien")
    # ---------------- debt (§4)
    if debt and not has(r"nhap"):
        d = until_day(q, today)
        if has(r"ngay phai tra|han (?:phai )?tra|han thanh toan") and has(r"(?:bao nhieu|may|bnhiu|bn|so) don") and shop is not None \
                and "due_date" in shop.O.columns and shop.O["due_date"].notna().any() and not has(r"qua han|tien|bao nhieu tien"):
            if has(r"muon hon|sau hom nay|con sau|chua den han|sau ngay hom nay|tre hon"): return "due_after", ans("number", f"n_due_orders(d={d}, after=True)")
            if P and not P.startswith(("today_p", "day(")): return "due_in", ans("number", f"n_due_orders({P}, d={d})")
        if has(r"ngay phai tra|han (?:phai )?tra|den han|qua han|ky han"): return None    # due dates: no shop-wide template
        m = re.search(r"tuoi no (?:la |da )?(?:qua|hon|tren) (\d+) ngay", t)
        if m: return "debt_older", ans("number", f"debt_older_than({d}, {int(m.group(1))})")   # 'tuổi nợ quá N ngày' -> one amount
        if has(r"tuoi no"): return "debt_aging", ans("list", f"debt_aging({d})")
        m = re.search(r"(?:hon|qua|tren) (\d+) ngay", t)
        if m and has(r"don dat|dat (da )?(hon|qua)|tuoi"): return "debt_older", ans("number", f"debt_older_than({d}, {int(m.group(1))})")
        m = re.search(r"(\d+) khach", t) or re.search(r"top (\d+)", t)
        if m and has(r"nhieu nhat|nhiu nhat|lon nhat|cao nhat"): return "top_debtors", ans("table", f"top_debtors({d}, {int(m.group(1))})")
        if has(r"no them|ghi no them|mua chiu them") and has(r"tra"):
            cust = V.find(tk, V.customers) if V else None
            return "debt_flow", ans("list", f"debt_flow({P}" + (f", customer={J(cust)}" if cust else "") + ")") if P else None
        if has(r"tra bot no|tra no|thu no|thu duoc .*no|thu hoi|khach tra|doi duoc|doi no") and P and not has(r"con no|chua tra"): return "debt_repaid", ans("number", f"debt_flow({P})[1]")
        if (has(r"bao nhieu khach|so khach|may khach|bnhiu khach|bn khach|bao nhieu nguoi|may nguoi|nhieu nguoi|bnhiu nguoi") or has(r"\bdem\b.*khach")) \
                and has(r"con no|chua tra|cong no|con thieu|thieu tien"): return "n_debtors", ans("number", f"n_debtors({d})")
        if has(r"tra gop|\bgop\b|ky han|ky gop"): return None                 # instalments: not the shop's order debts
        if V is not None and V.find(tk, V.customers) is not None: return None    # one customer's debt: no shop-wide template
        if has(r"con no|chua tra|tong no|cong no|con thieu|thieu tien|no chua thu|khach no"): return "debt_total", ans("number", f"debt_total({d})")
        return None
    # ---------------- price effect
    if has(r"tang gia|len gia|doi gia|tang len|\blen \d") and item is not None:
        m = re.search(r"(\d+) ngay (?:dau|ke tu|tu)", t) or re.search(r"(\d+) ngay", t)
        n = int(m.group(1)) if m else None
        dd = re.search(r"(\d{1,2})/(\d{1,2})/(20\d\d)", t)
        pr = re.findall(r"(\d{3,})\s*(?:d|dong)\b", t)
        if n is None: return None
        if dd: return "price_effect", ans("percent", f"price_effect({J(item)}, {n}, date={_dc(_date(*dd.group(1, 2, 3)))})")
        if pr: return "price_effect", ans("percent", f"price_effect({J(item)}, {n}, new_price={float(pr[-1])})")
        return None
    # ---------------- purchases
    ims = has(r"nhap tru ban|tru so ban|chenh lech nhap|nhap\b.{0,30}\btru\b.{0,30}\bban\b")
    if has(r"nhap") and not ims:
        if has(r"binh quan|trung binh") and item is not None and P:
            u = re.search(r"(?:mot|1|moi) " + UNITS + r"\b", t)
            return "avg_purchase", ans("number", f"avg_purchase_price({J(item)}, {P}" + (f", unit={J(u.group(1))}" if u else "") + ")")
        if has(r"nha cung cap|ncc") and P:
            by = "receipts" if has(r"chinh|nhieu phieu|nhieu lan") else "amount"
            return "top_supplier", ans("string", f"top_supplier({P}, by={J(by)})")
        mq = re.search(r"nhap (?:ve |them |vao )?(?:duoc )?(?:bao nhieu|may|bnhiu|bn|tong cong) " + UNITS + r"\b", t)
        if mq and item is not None and P and not has(r"tien|gia"):
            return "purchased_qty", ans("number", f"purchased_qty_base({J(item)}, {P})")
        if P and has(r"tong|bao nhieu tien|het bao nhieu|gia tri|tien hang|bnhiu|bn\b|nhieu tien|tien nhap"): return "purchase_total", ans("number", f"purchase_total({P})")
        return None
    if ims and item is not None and P: return "import_minus_sales", ans("number", f"import_minus_sales({J(item)}, {P})")
    # ---------------- lots / expiry
    if has(r"han dung|han su dung|\bhsd\b|het han\b|can date|\blo (?:hang|nao|ton|con)|(?:may|bao nhieu|so|cac) lo (?:con|ton|hang|sap)") or re.search(r"(?i)\blô\b", q):   # 'lọ' = bottle
        d = "TODAY"
        ms = re.findall(r"den het (?:ngay )?(\d{1,2})/(\d{1,2})/(20\d\d)", t) or re.findall(r"(?:toi|den) (?:het )?(?:ngay )?(\d{1,2})/(\d{1,2})/(20\d\d)", t)
        until = _dc(_date(*ms[-1])) if ms else None                       # the LAST date: 'tính đến hết ngày D, ... hạn dùng từ D đến hết D2'
        n = re.search(r"(\d+) ngay toi", t)
        if until is None and n: until = f"TODAY + {int(n.group(1))} * D"
        if has(r"gan het han nhat|het han som nhat|gan het han nhut|can date nhat|gan date nhat|het han gan nhat") and item is not None: return "nearest_expiry", ans("date", f"nearest_expiry({J(item)}, {d})")
        if until is not None:
            if has(r"tien von|gia tri|bao nhieu tien|bnhiu tien"): return "expiring_value", ans("number", f"money((lots_expiring({d}, {until})['remaining'] * lots_expiring({d}, {until})['unit_cost_base']).sum())")
            if has(r"may lo|bao nhieu lo|so lo|bnhiu lo"): return "expiring_lots", ans("number", f"len(lots_expiring({d}, {until}))")
            if has(r"(?:may|bao nhieu|bnhiu|bn|so) (?:mat hang|san pham|loai|thuoc|sp)\b"): return "n_items_expiring", ans("number", f"len(items_expiring({d}, {until}))")
            if has(r"(?:hang|san pham|mat hang|thuoc|sp|loai) nao|nhung (?:san pham|mat hang|hang|thuoc)|liet ke|\bke (?:giup|ra|ten|het)|cac (?:san pham|mat hang|thuoc|loai)"):
                return "items_expiring", ans("list", f"items_expiring({d}, {until})")
        return None
    if has(r"het hang|chay hang|dut hang|het sach") and not has(r"het han\b|nhap|ban chay"):
        d = until_day(q, today)
        if has(r"(?:may|bao nhieu|bnhiu|bn|so) (?:mat hang|san pham|loai|sp)\b"): return "n_out_of_stock", ans("number", f"len(out_of_stock({d}))")
        if has(r"(?:hang|san pham|mat hang|sp|loai) nao|nhung (?:san pham|mat hang|hang)|liet ke"): return "out_of_stock", ans("list", f"out_of_stock({d})")
    # ---------------- stock
    if has(r"gia von|tien von") and has(r"kho|\bton\b|hang ton"): return "stock_value", ans("number", f"stock_value({until_day(q, today)})")
    if has(r"ton kho|con ton|trong kho|con lai trong kho|ton cuoi|kho con|con bao nhieu|con may") and not rev:
        d = until_day(q, today)
        if has(r"gia von|tien von|gia tri") : return "stock_value", ans("number", f"stock_value({d})")
        if item is not None:
            m = re.search(r"(?:may|bao nhieu|bnhiu|bn) " + UNITS + r" (?:chan|nguyen)", t) or \
                re.search(r"(?:may|bao nhieu|bnhiu|bn) ((?:xe|cuon|thung|bao|hop|loc|vi|ket|bo) \w*\d\w*) (?:chan|nguyen)", t)
            if m is None and has(r"\bchan\b|\bnguyen\b") and has(r"\ble\b"):
                m = re.search(r"(?:may|bao nhieu|bnhiu|bn) " + UNITS + r"\b", t)
            if m: return "stock_split", ans("list", f"stock_split({J(item)}, {d}, {J(m.group(1))})")
            return "stock", ans("number", f"stock_at({J(item)}, {d})")
        return None
    # ---------------- deliveries
    dlv = shop is not None and shop.C.get("deliveries") is not None and len(shop.C["deliveries"]) > 0
    m = re.search(r"(?:tu|it nhat|toi thieu) (\d) dot|(\d) dot tro len|thanh (\d) dot", t)
    if dlv and m and has(r"\bdon\b") and P and has(r"giao|cho"):
        n = int(next(g for g in m.groups() if g)); return "orders_by_trips", ans("number", f"n_orders_by_trips({P}, {until_day(q, today)}, {n})")
    if dlv and item is not None and P and has(r"da (?:cho|giao)|xe (?:da )?(?:giao|cho)|cho di|giao di") and has(r"don dat|khach dat|dat (?:thang|trong|tu|ngay)|cac don"):
        return "delivered_qty", ans("number", f"delivered_qty({P}, TODAY, item={J(item)})")
    if has(r"dot (giao|hang)|chuyen hang|luot giao|dot cho") and P: return "deliveries", ans("number", f"n_delivery_trips({P})")
    # ---------------- profit
    if has(r"lai rong|loi nhuan rong|tru het chi phi|tru chi phi|loi that|lai thuc") and P: return "net_profit", ans("number", f"net_profit({P})")
    if has(r"lai gop|loi gop|lai nhat|loi nhat|lai nhieu nhat|loi nhieu nhat|lai cao nhat|lai gop cao nhat|cho loi|mang lai lai") and P:
        if has(r"hang nao|mat hang|san pham nao|mon nao|\bsp\b|nao") : return "top_profit_item", ans("string", f"top_items({P}, 1, by='profit')[0]")
        return "gross_profit", ans("number", f"gross_profit({P}{fa()})")
    # ---------------- unsold items
    m = re.search(r"(?:it nhat|tu|tren|o|trong) (\d+) don (?:hang )?khac nhau|co mat (?:trong|o) (?:it nhat )?(\d+) don|o (\d+) don (?:hang )?khac nhau"
                  r"|(?:xuat hien|co mat|duoc mua|nam|duoc goi|goi|gop mat|duoc ban|duoc dat) (?:trong|chong|o) (?:it nhat |tu |tro len |toi thieu )?(\d+) (?:don|bill|hoa don)", t)
    if m and P and has(r"mat hang|mon|san pham"):
        n = int(next(g for g in m.groups() if g)); return "items_in_n_orders", ans("number", f"int((item_sales({P}, by='n_orders') >= {n}).sum())")
    if has(r"\bbi e\b|ban khong duoc|ban ko duoc|ban ko dc|khong ban (?:duoc|ra duoc|ra) (?:cai|lan|mon) nao|khong ai mua|e han|\bees\b|mat hang e\b|mon e\b|hang e\b|\be\b.*khong ban|\be\b.*ko ban|\be\b.*k ban") and P and has(r"mat hang|mon|san pham"):
        return "unsold", ans("number", f"len(unsold_items({P}))")
    # ---------------- price on a day
    if has(r"don gia|gia ban|gia bao nhieu|gia bnhiu|ban ra voi|voi don gia|gia mot|gia 1") and item is not None and p is not None and p.days == 1:
        return "price_on", ans("number", f"price_on({J(item)}, {P.replace('day(', '').rstrip(')') if P.startswith('day(') else 'TODAY'})")
    # ---------------- cancel
    if has(r"\bhuy\b"):
        if has(r"\bvi\b(?! dien tu)|ly do|nguyen nhan"): return None        # a reason ('huỷ vì giao trễ'): not a column ('ví điện tử' is)
        if has(r"ti le|ty le|%|phan tram") and P: return "cancel_rate", ans("percent", f"cancel_rate({P})")
        if has(r"(?:bao nhieu|may|so|bnhiu|bn) (?:don|bill|hoa don|order)|dem") and P and not has(r"khach"): return "n_cancelled", ans("number", f"len(orders({P}, valid=None, status='huỷ'))")
        if has(r"khach") and has(r"nhieu|nhat") and P: return "top_canceller", ans("string", f"top_customers({P}, 1, by='n_cancelled')[0]")
        return None
    # ---------------- monthly lists / trends / best month
    ms = all_months(q); mpos = months_pos(_spread_year(_year_fix(K(q)))); ml = month_list(q)
    # 'T7 cao hơn T6, rồi T8 lại cao hơn T7, đúng không': every stated pair, in the stated direction
    MR = r"(?:thang |thg |th |t)?(\d{1,2})\s*[/-]\s*(20\d\d)"
    prs = re.findall(MR + r"\s+(?:(?:lai|deu|van|co|da|cung)\s+)?(cao hon|thap hon|nhieu hon|it hon|lon hon|nho hon)\s+(?:so voi\s+)?" + MR, _year_fix(K(q)))
    if prs and rev and has(r"dung (?:vay |the )?(?:khong|ko|hong|ha)|phai (?:khong|ko|hong)|co dung|co phai|dung chua|\?\s*$") and len(prs) * 2 >= len(ms):
        cmp_ = lambda c: ">" if c in ("cao hon", "nhieu hon", "lon hon") else "<"
        return "rev_cmp", ans("boolean", "all([" + ", ".join(f"revenue(month({int(y1)}, {int(m1)})) {cmp_(c)} revenue(month({int(y2)}, {int(m2)}))"
                                                     for m1, y1, c, m2, y2 in prs) + "])")
    if has(r"tang lien tuc|luon cao hon|thang sau .*cao hon|thang sau .*hon thang (?:truoc|trc)|luon hon|tang dan|co phai .*cao hon") and rev:
        if len(ms) >= 2 and has(r"\b(tu|giua) "):
            if len(ms) == 2: return "increasing", ans("boolean", f"increasing(month_seq({ms[0][6:-1]}, {ms[1][6:-1]}))")
        if len(ms) >= 2: return "increasing", ans("boolean", f"increasing([{', '.join(sorted(dict.fromkeys(ms), key=lambda c: tuple(map(int, re.findall(r'\d+', c)))))}])")
        if len(ml) >= 2: return "increasing", ans("boolean", f"increasing([{', '.join(ml)}])")
        return None
    if has(r"lan luot|moi thang|tung thang|theo thu tu|tu thang cu|3 thang lien") and rev and not pays and not chans:
        if len(ms) == 2 and has(r"(tu|ket thuc|den) "):
            if has(r"ket thuc"): return "months_list", ans("list", f"monthly_revenue(months_back({ms[-1]}, 3))")
            return "months_list", ans("list", f"monthly_revenue(month_seq({ms[0][6:-1]}, {ms[1][6:-1]}))")
        m = re.search(r"(\d) thang (?:lien nhau |lien tiep |gan nhat )?(?:ket thuc|tinh (?:toi|den)|den|cho den|toi)", t)
        if m and ms: return "months_list", ans("list", f"monthly_revenue(months_back({ms[-1]}, {int(m.group(1))}))")
        if len(ms) >= 2: return "months_list", ans("list", f"monthly_revenue([{', '.join(ms)}])")
        if len(ml) >= 2: return "months_list", ans("list", f"monthly_revenue([{', '.join(ml)}])")
        return None
    if rev and len(dict.fromkeys(ml)) >= 3 and not two_part and not has(r"\b(?:tu|giua)\b.*\b(?:den|toi)\b|cong lai|tong|gop|ca (?:ba|3|bon|4)|trung binh|binh quan|"
                                                                         r"tang|giam|nhat|so voi|so vs|hon|thang nao|%|phan tram|chiem"):
        return "months_list", ans("list", f"monthly_revenue([{', '.join(dict.fromkeys(ml))}])")
    if has(r"thang nao") and rev and len(ml) >= 2:
        seq = f"month_seq({ms[0][6:-1]}, {ms[-1][6:-1]})" if len(ms) == 2 and has(r"\b(tu|den|toi)\b") else f"[{', '.join(ml)}]"
        return "best_month", ans("date", f"best_month({seq})")
    # ---------------- growth
    if has(r"tang hay giam|thay doi|tang truong|len hay xuong|len xuong|cao hay thap|hon hay kem|hon kem|tang giam|tang/giam|giam/tang|lech bao nhieu %|bao nhieu phan tram so|% so (?:voi|vs|cung)|chenh .{0,40}(?:bao nhieu|may|bnhiu|bn) (?:%|phan tram)|"
           r"(?:tang|giam|chenh|lech) (?:may|bao nhieu|bnhiu|bn) (?:phan tram|%)|nhieu hay it hon|it hay nhieu hon|cao hay thap hon|"
           r"\bhon\b.{0,40}(?:may|bao nhieu|bnhiu|bn) (?:%|phan tram)|(?:may|bao nhieu|bnhiu|bn) (?:%|phan tram) so (?:voi|vs)") and (rev or has(r"don")):
        meas = "revenue" if rev else "n_orders"
        if has(r"ca chieu") and has(r"ca sang"): return "growth_shift", ans("percent", f"growth({meas}({P}, shift='chiều'), {meas}({P}, shift='sáng'))") if P else None
        if has(r"truoc tet") and P: return "growth_tet", ans("percent", f"growth({meas}({P}), {meas}(before_tet({P[4:-1] if P.startswith('tet(') else 'TODAY.year'})))") if P.startswith("tet") else None
        if has(r"tet nam (truoc|ngoai|roi)") and has(r"tet nam nay|tet nay") : return "growth_tet_yoy", ans("percent", f"growth({meas}(tet_of(TODAY)), {meas}(tet(TODAY.year - 1)))")
        if has(r"cung ky thang (truoc|roi|trc|lien truoc)|cung so ngay (?:dau |cua )?thang (?:truoc|roi|trc)") or \
                (has(r"dau thang|thang nay|th nay") and has(r"cung ky|cung so ngay|so (?:voi|vs) (?:cung so ngay (?:dau |cua )?)?thang (?:truoc|trc|roi|lien truoc)|thang (?:truoc|trc|roi)")):
            return "growth_mtd", ans("percent", f"growth({meas}(this_month(TODAY){fa()}), {meas}(prev_month_same_days(TODAY){fa()}))")
        if len(ms) == 2 and len(mpos) == 2 and ms[0] != ms[1]:
            sv = re.search(r"\bso (?:voi|vs|cung)\b", K(q))
            after = [i for i, (st, _, _) in enumerate(mpos) if sv and st >= sv.end()]
            bi = after[0] if after else min((0, 1), key=lambda i: mpos[i][1:])   # 'so với A' -> base A; else the earlier month
            new, base = ms[1 - bi], ms[bi]
            return "growth2", ans("percent", f"growth({meas}({new}{fa()}), {meas}({base}{fa()}))")
        if P and P.startswith("month") and has(r"thang (truoc|lien truoc|lien trc|trc|ngay truoc|truoc do|lien ke|ngay truoc do)"): return "growth_prev", ans("percent", f"growth({meas}({P}{fa()}), {meas}(prev_month({P}){fa()}))")
        if P and P.startswith("month") and has(r"cung (ky|thang) nam (ngoai|truoc|roi)|nam ngoai|nam truoc|nam roi"): return "growth_yoy", ans("percent", f"growth({meas}({P}{fa()}), {meas}(same_month_last_year({P}){fa()}))")
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
    _aov_q = has(r"trung binh|binh quan") and has(r"\bdon\b|\bbill\b|hoa don") and not has(r"khach nao|\bai\b|nhom khach|khach hang (?:than thiet|vip|si|le)")
    _item_q = has(r"(?:\bhang|mat hang|san pham|\bmon|\bsp|\bthuoc) (?:nao|gi)\b") and not has(r"khach (?:hang )?nao|\bai\b")
    if (has(r"khach") or has(r"\bai\b (?:mua|chi|dat)")) and P and not has(r"dat khach") and not _aov_q and not _item_q:
        m = re.search(r"(?:tu|it nhat|tro len|toi thieu) (\d+) (?:don|bill|hoa don|lan)", t)
        if m:
            if has(r"phan tram|%|ti le"): return "pct_customers", ans("percent", f"pct_customers({P}, {int(m.group(1))})")
            return "n_customers_min", ans("number", f"n_customers({P}, min_orders={int(m.group(1))})")
        by = "n_cancelled" if has(r"hay doi y|doi y (?:nhieu|nhat)|hay huy") else "n_orders" if has(r"nhieu (lan|don)|nhiu (lan|don)|so (?:luong )?don|nhieu don|don hang (?:nhieu|cao) nhat|so lan mua") else "n_days" if has(r"trung thanh|nhieu ngay") else "aov" if has(r"\bsop\b") else "revenue" if has(r"nhieu tien|nhiu tien|chi tieu|tien nhieu|mua nhieu nhat|cao nhat|lon nhat|nhieu nhat|nhiu nhat|tong tien mua") else None
        if by:
            m = re.search(r"(?<![\d/])([1-9]|1\d|20) khach", t)
            if m: return "top_customers", ans("list", f"top_customers({P}, {int(m.group(1))}, by={J(by)})")
            if has(r"\bai\b|khach nao|khach hang nao|la ai|nguoi nao"): return "top_customer", ans("string", f"top_customers({P}, 1, by={J(by)})[0]")
        if "customer_group" in f and (rev or has(r"gia tri")) and has(r"trung binh|binh quan") and has(r"don"): return "aov", ans("number", f"aov({P}{fa()})")
        if "customer_group" in f and rev: return "revenue", ans("number", f"revenue({P}{fa()})")
        return None
    # ---------------- staff
    if has(r"nhan vien|nguoi ban|\bnv\b") and P:
        if "staff" not in f and V is not None:
            sn = V.find(tk, V.staff_names)
            if sn is not None: f["staff"] = V.staff_names[dm.key(sn)] if dm.key(sn) in V.staff_names else sn
        if "staff" in f and (rev or has(r"tien hang|cong lai")): return "revenue", ans("number", f"revenue({P}{fa()})")
        if has(r"nhat|gioi nhat|cao nhat"): return "top_staff", ans("string", "rank(staff_sales(" + P + "), 1)[0]")
        return None
    # ---------------- items: top / qty / size
    m = re.search(r"(?:top |liet ke |^|cho minh |cho toi )(\d+) (?:mat hang|hang|san pham|mon|sp|best)", t) or re.search(r"top (\d+)", t) or re.search(r"(\d+) (?:mat hang|mon|san pham|hang|sp) (?:ban )?(?:chay|nhieu|hot|best)", t) \
        or re.search(r"(?<![\d/.,])(?<!thang )(?<!thg )(?<!th )(?<!ngay )(?<!nam )(?<!quy )(?<!tuan )(\d{1,2}|hai|ba|bon|nam|sau|bay|tam|chin|muoi) (?:mat hang|san pham|mon|sp)\b", t) or re.search(r"\btop (hai|ba|bon|nam|sau|bay|tam|chin|muoi)\b", t)
    nN = (int(m.group(1)) if m.group(1).isdigit() else NUMW[m.group(1)]) if m else None
    if nN is not None and nN > 50: m = nN = None                       # 'Tết 2026 hàng bán nhiều nhất': a year, not top-2026
    best = has(r"ban chay|chay nhat|ban nhieu nhat|ban dc nhieu nhat|ban duoc nhieu nhat|hot nhat|best seller|ban nhiu nhat|ban dc nhiu nhat|nhieu nhat|nhiu nhat",
               r"doanh thu (?:cao|lon) nhat|nhieu tien nhat|ua chuong nhat|duoc (?:khach )?ua chuong|nhieu " + UNITS + r" nhat")
    if best and P and (has(r"mat hang|hang nao|san pham|mon nao|\bsp\b|mon |hang gi|la gi|la chi|hang chi|best seller|top|nhung") or m):
        by = "revenue" if has(r"nhieu tien|nhiu tien|doanh thu|thu ve") else "n_orders" if has(r"nhieu don|ua chuong") else "qty"
        g = {k: v for k, v in f.items() if k in ("group",)}
        if nN and nN > 1: return "top_items", ans("list", f"top_items({P}, {nN}, by={J(by)}{fa(g)})")
        if nN is None and has(r"nhung (?:mon|mat hang|san pham|sp|hang)|cac (?:mon|mat hang|san pham)"): return None   # plural, no count
        return "top_item", ans("string", f"top_items({P}, 1, by={J(by)}{fa(g)})[0]")
    if has(r"size (nb|s|m|l|xl|xxl|xxxl)\b") and P and not rev and not has(r"%|phan tram|chiem|ti le|ty le"):
        sz = re.search(r"size (nb|s|m|l|xl|xxl|xxxl)\b", t).group(1).upper(); return "size_qty", ans("number", f"size_qty({P}, {J(sz)})")
    if P and not rev and (item is not None or "group" in f) and (re.search(r"(bao nhieu|may|bnhiu|bn|so|tong cong|total) " + UNITS + r"\b", t)
                                                               or re.search(r"(bao nhieu|may|bnhiu|bn|so|tong cong|total) " + UNITS + r"\b", t_unmasked)
                                                               or has(r"ban ra bao nhieu|ban duoc bao nhieu|ban dc bao nhieu|so luong")) \
            and not has(r"giao|cho di|da cho|van chuyen|so voi|so vs|tang hay giam|chenh|nhap"):
        return "item_qty", ans("number", f"item_qty({P}{fa()})")
    if P and not rev and item is None and not f and not pays and not chans and not two_part:
        # 'bán được bao nhiêu <đơn vị>' over the whole shop (rulebook: only items whose base unit is that unit)
        mu = re.search(r"\bban (?:ra |duoc |dc )*(?:tong cong |tat ca |het |tong )?(?:duoc |dc )?(?:la )?(?:bao nhieu|bnhiu|bn|may) " + UNITS + r"\b", t) or \
             re.search(r"(?:^|\s)(?:tong )?so " + UNITS + r" (?:da )?ban (?:ra|duoc|dc)\b", t)
        _t_np = re.sub(r"\d+ ngay (?:gan nhat|gan nhut|lien tiep gan nhat)", " ", t)
        if mu and not any(re.search(x, _t_np) for x in (r"(?:mat hang|san pham|\bmon|\bsp|loai|hang) (?:nao|gi)|(?:bao nhieu|may|so|bnhiu|bn) (?:mat hang|san pham|mon|sp|loai)\b",
                          r"\bkhach|nhan vien|\bnv\b|\bdon\b|hoa don|\bbill|nhap|\bton\b|con lai|tra hang|tra lai|\bhuy|trung binh|binh quan|moi ngay|"
                          r"nhat\b|ngay nao|thang nao|gio nao|hom nao|tang|giam|so voi|so vs|%|phan tram|size|nhom")):
            u = mu.group(1)                                                # every other word: period, filler, 'cộng mọi mặt hàng'
            tu = re.sub(r"\d+ ngay (?:gan nhat|gan nhut|qua|vua qua|gan day|lien tiep gan nhat|cuoi|vua roi)|tinh (?:luon|ca) hom nay(?: thi)?", " ", t)
            if not [w for w in tu.split() if not re.search(r"\d", w) and w not in UNIT_QTY_WORDS and w != u]:   # (an unlinked
                return "unit_qty", ans("number", f"item_qty({P}, base_unit={J(u)})")                          # item: no rule)
    # ---------------- supplier / deliveries / profit / stock words (phrasings the sections above do not read)
    if has(r"nha cung cap|\bncc\b") and P and has(r"\bchinh\b|nhieu phieu|nhieu lan|nhap nhieu|nhieu tien nhat|nhieu nhat|lon nhat"):
        by = "receipts" if has(r"\bchinh\b|nhieu phieu|nhieu lan") else "amount"
        return "top_supplier", ans("string", f"top_supplier({P}, by={J(by)})")
    if has(r"\bgiao\b") and has(r"(?:may|bao nhieu|bnhiu|bn|tong cong|tong) (?:\w+ )?dot\b") and P and dlv:
        return "deliveries", ans("number", f"n_delivery_trips({P})")
    if has(r"loi nhuan gop|lai gop") and P and has(r"(?:hang|mat hang|san pham|mon|sp) (?:nao|gi)|la gi|la hang") and has(r"nhat|nhieu nhat|cao nhat|lon nhat"):
        return "top_profit_item", ans("string", f"top_items({P}, 1, by='profit')[0]")
    if has(r"loi nhuan gop|lai gop") and P and not has(r"nao|mat hang|san pham"): return "gross_profit", ans("number", f"gross_profit({P}{fa()})")
    # ---------------- revenue variants
    if has(r"co may ngay|bao nhieu ngay|bnhiu ngay|bn ngay|\bmay ngay|may bua|co may hom|so ngay|dem so ngay|bao nhieu bua|nhieu ngay dat|nhiu ngay|co nhieu ngay") and P:
        x = money_in(t, r"(?:tu|dat|tren|it nhat|tro len)\s*")
        if x and has(r"tro len"): return "n_days_rev_ge", ans("number", f"n_days_revenue_at_least({P}, {x:.0f})")
    if has(r"\bdon\b|\bbill\b|hoa don|\border\b") and has(r"tro len") and P and (rev or has(r"gia tri")):
        x = money_in(t, r"(?:tu|dat|tren|it nhat|tri gia|gia tri|tro len)\s*")
        if x: return "n_orders_rev_ge", ans("number", f"n_orders_revenue_at_least({P}, {x:.0f}{fa()})")
    if has(r"trung binh|binh quan") and has(r"\bdon\b|\bbill\b|hoa don|\border\b") and (rev or has(r"gia tri")) and P and not has(r"nhap|moi ngay"):
        return "aov", ans("number", f"aov({P}{fa()})")
    mw = re.search(r"(?:trung binh|binh quan) (?:moi |mot |1 )?(?:ngay )?(thu (?:hai|ba|tu|nam|sau|bay|[2-7])|chu nhat|\bcn\b)", t)
    if mw and P and not two_part:                                       # rulebook: divided by the number of such days of the period
        w = mw.group(1); wd = {"thu hai": 0, "thu 2": 0, "thu ba": 1, "thu 3": 1, "thu tu": 2, "thu 4": 2, "thu nam": 3, "thu 5": 3, "thu sau": 4,
                              "thu 6": 4, "thu bay": 5, "thu 7": 5, "chu nhat": 6, "cn": 6}.get(w)
        by = "revenue" if rev else "orders" if has(r"\bdon\b|bill|hoa don|luot|phieu|order") else "qty" if re.search(r"(?:bao nhieu|may|bnhiu|bn) " + UNITS + r"\b", t) else None
        if wd is not None and by: return "weekday_avg", ans("number", f"weekday_avg({P}, {wd}, by={J(by)}{fa()})")
    if has(r"ngay nao|ngay may|hom nao|bua nao") and P and (rev or has(r"cao nhat|nhieu nhat|dat khach")):
        by = "orders" if has(r"nhieu don|dat khach|don nhat") else "revenue"
        return "best_day", ans("date", f"best_day({P}, by={J(by)})")
    if has(r"gio cao diem|gio nao") and P: return "peak_hour", ans("number", f"peak_hour({P})")
    if rev and P and (len(pays) == 2 or len(chans) == 2) and (has(r"\bva\b|voi|lan luot|moi loai") or two_part):
        key_ = "payment" if len(pays) == 2 else "channel"; vals = pays if len(pays) == 2 else chans
        words = PAY_WORDS if key_ == "payment" else CHAN_WORDS
        pos = {lab: min((t.find(w) for w, l2 in words if l2 == lab and t.find(w) >= 0), default=999) for lab in vals}
        order = sorted(vals, key=lambda v: pos[v])
        return "rev_pair", ans("list", f"[revenue({P}, {key_}={J(order[0])}), revenue({P}, {key_}={J(order[1])})]")
    multi = len(dict.fromkeys(ms)) >= 2 and not P.startswith("between") if P else False   # several months, the template reads one
    count_q = bool(re.search(r"(?i)đếm", str(q))) or has(r"\bdem (?:so|gium|giup|xem|coi|thu)\b|\bco may\b|(?:bao nhieu|bnhiu|\bbn\b|\bmay|\bso) (?:ngay|bua|hom|don|lan|khach|nguoi|phieu|luot|mat hang|mon|san pham|loai)\b")
    yes_no = has(r"\bhon\b|dung (?:vay |the )?(?:khong|ko|hong)|phai (?:khong|ko|hong)|co dung|co phai|\bluon (?:cao |thap |nhieu |it )?hon\b")
    if rev and P and not two_part and not multi and not count_q and not yes_no and not has(r"tang|giam|so voi|so vs|\btop\b|(?<!gan )(?<!moi )nhat|%|phan tram|ti trong|chiem|lan luot|\bva\b|moi loai|tung|trung binh|binh quan"):
        g = dict(f)
        if not f and not pays and not chans and (unlinked_entity(q, t) or has(r"\btho\b|\bphieu\b")): return None   # a mechanic / a ticket
        if len(pays) == 1 and has(r"tra bang|thanh toan|tra qua|bang"): g["payment"] = pays[0]
        if len(chans) == 1: g["channel"] = chans[0]
        if not g and has(r"mat hang|san pham|\bsp\b|nhom|kenh|tra bang|thanh toan"): return None
        return "revenue", ans("number", f"revenue({P}{fa(g)})")
    if has(r"so don|may don|bao nhieu don|bnhiu don|bn don|so bill|bao nhieu bill|may bill|bao nhieu hoa don|nhieu don|tong don|ddown|don hang|\bdh\b|\bo do\b|dem so don|bao nhieu don hang|chot duoc|bao nhieu order|may order|so order") and P \
            and not two_part and not multi and not yes_no and not has(r"tra hang|khach|tro len|giao|%|ti le|phan tram|trung binh|binh quan|\bchiu\b|\bno\b") \
            and (not has(r"co mua|co ban") or "group" in f or "item" in f):
        g = dict(f)
        if not g and has(r"mat hang|san pham|\bsp\b|nhom|kenh"): return None
        if len(chans) == 1: g["channel"] = chans[0]
        if len(pays) == 1 and has(r"tra bang|thanh toan"): g["payment"] = pays[0]
        return "n_orders", ans("number", f"n_orders({P}{fa(g)})")
    return None
