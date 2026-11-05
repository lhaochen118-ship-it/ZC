#!/usr/bin/env python3
"""Question cleaning for DataMate: generic Vietnamese chat normalisation (nothing about any business or question id).

    nz = Normalizer.for_split(split_dir)            # vocabulary: accented training questions + the split's rulebook
    r = nz.clean(question, as_of_date)
    r["clean"]    readable question: accents restored, abbreviations expanded, Telex slips fixed, dialect, months
    r["key"]      accent-free lowercase key (what the rule-based reader qparse matches)
    r["period"]   first period expression -> {"start", "end", "code"} (qparse.parse_period), or None
    r["changes"]  [[from, to, rule], ...]

Steps (in order, each logged in `changes`):
  1. unicode NFC, emoji / decorative symbols removed, spaces collapsed
  2. chat abbreviations, whole tokens only ('dthu' -> 'doanh thu', 'bnhiu' -> 'bao nhiêu', 'hqua' -> 'hôm qua', 'đc' -> 'được')
  3. Telex slips, accepted only when the decoded word is a known word ('dduwowcj' -> 'được', 'toongr' -> 'tổng')
  4. central-Vietnamese words in unambiguous contexts ('tháng ni' -> 'tháng này', 'thợ mô' -> 'thợ nào', 'rứa' -> 'vậy')
  5. accents restored on accent-free words: Viterbi over the accented forms seen in the vocabulary, bigram model with
     interpolated smoothing ('doanh thu thang nay' -> 'doanh thu tháng này'); unknown words (names, codes) are kept
  6. month shorthands ('th7/2025', 'T11/2025', '6-2026', '4/26') -> 'tháng 7/2025'
The original question stays authoritative: the agent shows `clean` next to it, never instead of it.
CLI: qnorm.py <split_dir> <out.jsonl>      (cleans every question of the split)"""
import os, re, sys, json, math, unicodedata
from collections import Counter, defaultdict

ABBR = {
    "hqua": "hôm qua", "hwa": "hôm qua", "homqua": "hôm qua", "hnay": "hôm nay", "homnay": "hôm nay", "bữa ni": "hôm nay",
    "dthu": "doanh thu", "đthu": "doanh thu", "dt": None,                     # 'dt' is ambiguous (doanh thu / điện thoại): kept
    "bn": "bao nhiêu", "bnhiu": "bao nhiêu", "bnhieu": "bao nhiêu", "bnh": "bao nhiêu", "baonhieu": "bao nhiêu",
    "nhiu": "nhiêu", "nhìu": "nhiều", "nhiu tiền": "nhiêu tiền",
    "đc": "được", "dc": "được", "đươc": "được", "dk": "được", "đk": "được",
    "ko": "không", "kh": "không", "hok": "không", "hông": "không", "kô": "không", "k0": "không", "khg": "không", "hem": "không",
    "trc": "trước", "vs": "với", "z": "vậy", "zậy": "vậy", "dzậy": "vậy", "dzay": "vậy", "zay": "vậy",
    "sp": "sản phẩm", "đh": "đơn hàng", "dh": "đơn hàng", "ntn": "như thế nào", "nhứt": "nhất",
    "kòn": "còn", "kả": "cả", "wán": "quán", "mik": "mình", "mk": "mình", "cko": "cho", "xố": "số", "kó": "có", "mí": "mấy",
    "trog": "trong", "jum": "giùm", "júp": "giúp", "zô": "vô", "zo": "vô", "tíh": "tính", "từg": "từng", "ngừi": "người",
    "tỉ nệ": "tỉ lệ", "nượt": "lượt", "lựơt": "lượt", "xửa": "sửa", "cũa": "của", "nheiu": "nhiêu", "ddown": "đơn",
    "ck": "chuyển khoản", "tm": "tiền mặt", "ncc": "nhà cung cấp", "kh lẻ": "khách lẻ",
    "xi ô đi": "COD", "xi o đi": "COD", "xiôđi": "COD", "si ô đi": "COD",                  # COD said in Vietnamese
    "o đờ": "đơn", "ô đờ": "đơn", "o-đờ": "đơn", "ođờ": "đơn", "oder": "đơn",              # 'order' said in Vietnamese
}
ABBR = {k: v for k, v in ABBR.items() if v}
MULTI = sorted([k for k in ABBR if " " in k], key=len, reverse=True)
DIALECT_NI = {"tháng", "tuần", "năm", "bữa", "hôm", "ngày", "quý", "lúc", "đợt", "kỳ", "thang", "tuan", "nam", "bua", "hom", "ngay", "quy", "luc", "dot", "ky"}
DIALECT_MO = {"thợ", "ngày", "tháng", "xe", "hàng", "món", "khách", "loại", "nhóm", "người", "tho", "ngay", "thang", "hang", "mon",
              "khach", "loai", "nhom", "nguoi", "dòng", "dong", "sp"}
COMMON = """bao nhiêu tháng năm ngày tuần hôm qua nay doanh thu tiền bán được đơn hàng khách tổng số trước sau với của cửa tiệm quán
nào nhất nhiều ít cao thấp tăng giảm trung bình phần trăm tỉ lệ huỷ hủy trả nợ còn hết giá nhập vốn lãi lời mặt món nhóm
cho biết xem giúp giùm mình tôi em anh chị vậy nhỉ không có là gì mấy những các mỗi một hai ba từ đến tới trong ngoài liền
đầu cuối kỳ quý tết thứ chủ nhật sáng chiều tối ca giờ lượt phiếu xe thợ máy điện thoại nhân viên người chiếm so sánh cùng
lần đợt giao chưa đã đang rồi thì mà và hay hoặc bằng hơn kém thêm bớt khoảng gần nhất chạy ế tồn kho thuốc kê đơn hoạt chất
lô hạn dùng size gói hộp chai lon thùng bao cây viên vỉ tuýp ly cà phê trà sữa bánh nước đá cơm mì cháo phở bỏ tính đếm liệt kê
những ai đắt rẻ trị giá vào ra mua chi tiêu nhà cung cấp ship phí điểm thành viên voucher đăng ký góp cũ mới biển dòng
bảo hành sửa làm việc vị lẻ sỉ quen trung thành sộp đổi ý ưa chuộng chăm chỉ cao điểm khung này""".split()
WORD = re.compile(r"[^\W\d_]+", re.U)
_TONE = {"s": "́", "f": "̀", "r": "̉", "x": "̃", "j": "̣"}
_VOW = [("aa", "â"), ("aw", "ă"), ("ee", "ê"), ("oo", "ô"), ("ow", "ơ"), ("uw", "ư"), ("dd", "đ")]


def qnorm_key(s):
    """accent-free lowercase word / number tokens of a name ('Snack Lay's vị tự nhiên 56g' -> ['snack', 'lay', 's', ...])."""
    return re.findall(r"[^\W_]+", strip_accents(str(s)).lower())


def strip_accents(s):
    s = unicodedata.normalize("NFD", s).replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFC", "".join(c for c in s if unicodedata.category(c) != "Mn"))


def telex_all(w):
    """every decoding of a Telex-typed token, one per tone position ('gias' -> ['gía', 'giá']); [] if not Telex.
    The caller keeps the decoding that is a known word (onsets like 'gi', 'qu' make a fixed tone rule unreliable)."""
    s = w.lower(); tone = ""
    if len(s) > 2 and s[-1] in _TONE and re.search(r"[aeiouy]", s[:-1]): tone, s = _TONE[s[-1]], s[:-1]
    t = s
    for a, b in _VOW: t = t.replace(a, b)
    t = t.replace("uơ", "ươ")
    if t == s and not tone: return []
    if not tone: return [unicodedata.normalize("NFC", t)]
    return [unicodedata.normalize("NFC", t[:i + 1] + tone + t[i + 1:]) for i, ch in enumerate(t) if ch in "aeiouyăâêôơư"]


def telex(w):
    """decode one Telex-typed token ('tieenf' -> 'tiền'); None if it does not look like Telex."""
    s = w.lower(); tone = ""
    if len(s) > 2 and s[-1] in _TONE and re.search(r"[aeiouy]", s[:-1]): tone, s = _TONE[s[-1]], s[:-1]
    t = s
    for a, b in _VOW: t = t.replace(a, b)
    t = t.replace("uơ", "ươ")
    if t == s and not tone: return None
    if tone:
        vs = [i for i, ch in enumerate(t) if ch in "aeiouyăâêôơư"]
        if not vs: return None
        pri = [i for i in vs if t[i] in "ăâêôơư"]
        i = pri[-1] if pri else (vs[-2] if len(vs) >= 2 and vs[-1] == len(t) - 1 else vs[-1])
        t = t[:i + 1] + tone + t[i + 1:]
    return unicodedata.normalize("NFC", t)


def _case_like(src, dst):
    if strip_accents(dst).lower() == src.lower() and dst.lower() == src.lower(): return src   # unchanged word: as written
    if len(src) > 1 and src.isupper(): return dst.upper()
    return dst[:1].upper() + dst[1:] if src[:1].isupper() else dst


def _recase(src, dst):
    """dst's letters in src's casing (ALL CAPS / Capitalised / lower): the user's casing, the data's accents."""
    if len(src) > 1 and src.isupper(): return dst.upper()
    return dst[:1].upper() + dst[1:].lower() if src[:1].isupper() else dst.lower()


_TONES = "\u0300\u0301\u0303\u0309\u0323"


def word_sig(w):
    """a word up to its tone mark's position ('huỷ' and 'hủy' are the same word)."""
    d = unicodedata.normalize("NFD", w.lower())
    return "".join(c for c in d if c not in _TONES), "".join(sorted(c for c in d if c in _TONES))


def _not_after_thang(f):
    """month rewrite unless the text already says 'tháng' right before the match."""
    def g(m):
        return m.group(0) if strip_accents(m.string[:m.start()]).lower().rstrip().endswith(("thang", "thg")) else f(m)
    return g


MONTHS = [
    (re.compile(r"\b(?:th|thg|t)\s?(\d{1,2})\s*[/\-]\s*(20\d\d|\d\d)\b", re.I),
     lambda m: f"tháng {int(m.group(1))}/{m.group(2) if len(m.group(2)) == 4 else '20' + m.group(2)}"),
    (re.compile(r"(?<![\d/\-.])(\d{1,2})\s*-\s*(20\d\d)\b"), _not_after_thang(lambda m: f"tháng {int(m.group(1))}/{m.group(2)}")),
    (re.compile(r"(?<![\d/\-.])(\d{1,2})/(2[3-9])(?![\d/])"), _not_after_thang(lambda m: f"tháng {int(m.group(1))}/20{m.group(2)}")),
    (re.compile(r"\b(?:th|thg)\s?(\d{1,2})\b(?!\s*(?:giờ|h\b|ngày|/))", re.I), lambda m: f"tháng {int(m.group(1))}"),
    (re.compile(r"\b(?:th|thg)\s+(trước|này|sau|rồi|tới|nay|liền trước|kế tiếp)\b", re.I), lambda m: f"tháng {m.group(1)}"),
]


_QWORDS = re.compile(r"\b(nhất|hơn|hay ít|ít hay|nào|gì|mấy|bao nhiêu|bao lâu|ai|sao|đâu|không|chưa|top|liệt kê|xếp hạng|"
                     r"có phải|so với|so sánh|nhiêu)\b")


class Normalizer:
    def __init__(self, texts):
        """texts: accented Vietnamese text (rulebook, training questions); only sentences with accents are learnt."""
        self.uni, self.bi, self.forms = Counter(), Counter(), defaultdict(Counter)
        for t in list(texts) + [" ".join(COMMON)]:
            for line in re.split(r"[\n.;:!?()|]+", unicodedata.normalize("NFC", t)):
                ws = [w.lower() for w in WORD.findall(line)]
                if not ws or sum(not w.isascii() for w in ws) < 0.5 * len(ws): continue   # typed with accents (not 1 'đ' in a plain line)
                for i, w in enumerate(ws):
                    self.uni[w] += 1; self.forms[strip_accents(w)][w] += 1
                    if i: self.bi[(ws[i - 1], w)] += 1
        # typos inside accented questions ('toongr', 'dduwowcj') must not become vocabulary: drop an ASCII word whose
        # Telex decoding is a known accented word, and every ABBR key
        bad = {w for w in self.uni if (w.isascii() and any(t != w and t in self.uni for t in telex_all(w))) or w in ABBR}
        for w in bad:
            del self.uni[w]; f = self.forms.get(strip_accents(w))
            if f is not None:
                f.pop(w, None)
                if not f: del self.forms[strip_accents(w)]
        self.bi = Counter({k: v for k, v in self.bi.items() if k[0] not in bad and k[1] not in bad})
        self.N = sum(self.uni.values()) or 1
        self.vocab = {w for w in self.uni}

    # ---- typo repair (generic Vietnamese typing noise: consonants swapped by pronunciation, a letter dropped / doubled, a
    #      final n / ng, two words typed without the space). Only for a word that is neither known nor part of a name.
    _INIT = [("tr", "ch"), ("ch", "tr"), ("s", "x"), ("x", "s"), ("d", "gi"), ("gi", "d"), ("d", "v"), ("v", "d"), ("r", "d"), ("d", "r"),
             ("r", "gi"), ("gi", "r"), ("l", "n"), ("n", "l"), ("ng", "nh"), ("k", "c"), ("c", "k"), ("ph", "f"), ("q", "c")]
    _FINAL = [("ng", "n"), ("n", "ng"), ("c", "t"), ("t", "c"), ("nh", "n"), ("n", "nh"), ("ch", "t"), ("i", "y"), ("y", "i")]
    _LETTERS = "aăâeêioôơuưyhngcthđmnrsvx"

    def _edits(self, w):
        out = set()
        for a, b in self._INIT:
            if w.startswith(a): out.add(b + w[len(a):])
        for a, b in self._FINAL:
            if w.endswith(a): out.add(w[:-len(a)] + b)
        if len(w) >= 3:
            for i in range(len(w)):                                     # one letter dropped / doubled (no free substitution:
                out.add(w[:i] + w[i + 1:])                              # 'xả' -> 'cả', 'vô' -> 'có' are other words)
                for c in self._LETTERS: out.add(w[:i] + c + w[i:])
        return {x for x in out if len(x) >= 2}

    def typo_fix(self, w, prev=None, nxt=None):
        """the known word a typed word most likely means ('tền' -> 'tiền', 'dảm' -> 'giảm', 'xo' -> 'so', 'chong' -> 'trong',
        'đơng' -> 'đơn', 'nhiềuđơn' -> 'nhiều đơn'), or None. Accents of the typed word are a weak hint: a candidate with the
        same accent-free letters, or one edit away, scored by its frequency and its bigrams with the neighbours."""
        lw = w.lower()
        if len(lw) < 2 or lw in self.vocab or not re.fullmatch(r"[^\W\d_]+", lw): return None
        def score(c):
            sc = math.log(1 + self.uni.get(c, 0))
            if prev: sc += 2.0 * math.log(1 + self.bi.get((prev, c), 0))
            if nxt: sc += 2.0 * math.log(1 + self.bi.get((c, nxt), 0))
            return sc
        cands = {}
        base = strip_accents(lw)
        if lw.isascii():                                                  # typed without accents: letters only (the accent
            if base in self.forms: return None                            # restoration step decides the accents)
            for e in self._edits(lw):
                if e in self.forms and sum(self.forms[e].values()) >= 5:
                    cands[e] = math.log(1 + sum(self.forms[e].values())) + 2.0 * math.log(1 + sum(self.bi.get((prev, c), 0) for c in self.forms[e]) if prev else 1)
        else:
            for c, n in (self.forms.get(base) or {}).items():             # same letters, other accents ('mội' -> 'mỗi')
                cands[c] = score(c) + 1.0
            for e in self._edits(lw):
                if e in self.vocab and self.uni.get(e, 0) >= 3: cands[e] = max(cands.get(e, -1e9), score(e) + 0.5)
                for c in (self.forms.get(strip_accents(e)) or {}):         # an edit + other accents ('dảm' -> 'giảm')
                    if self.uni.get(c, 0) >= 3: cands[c] = max(cands.get(c, -1e9), score(c) - 1.0)
        for i in range(2, len(lw) - 1):                                   # two words glued ('nhiềuđơn', 'ngàynào')
            a, b = lw[:i], lw[i:]
            if a in self.vocab and b in self.vocab and self.uni[a] >= 5 and self.uni[b] >= 5:
                cands[a + " " + b] = 0.5 * (score(a) + score(b)) + math.log(1 + self.bi.get((a, b), 0))
        def ctx(c):                                                       # a correction must fit its neighbours strongly: known
            ws = c.split(); first, last = ws[0], ws[-1]                   # bigrams worth >= log 21 together ('bao nhiêu tền' ->
            fs = self.forms.get(first, {first: 1}) if c.isascii() else {first: 1}   # 262 x 'nhiêu tiền'); a valid word outside
            ls = self.forms.get(last, {last: 1}) if c.isascii() else {last: 1}      # the vocabulary ('có tận tâm', 'đường hậu
            if len(ws) == 2: return self.bi.get((ws[0], ws[1]), 0) >= 1           # giang') rarely has that much support
            bp = max([self.bi.get((prev, f), 0) for f in fs] or [0]); bn = max([self.bi.get((l, nxt), 0) for l in ls] or [0])
            return math.log(1 + bp) + math.log(1 + bn) >= math.log(21)
        cands = {c: v for c, v in cands.items() if ctx(c)}
        if not cands: return None
        best = max(cands, key=lambda c: (cands[c], c))
        return best if cands[best] >= math.log(1 + 3) else None

    def set_entities(self, names):
        """names written in the business's data (items, customers, addresses, labels ...). Two uses:
        - linking: a span of the question that spells an accented name (accents, case and punctuation aside; the whole
          name of >= 2 words, or >= 3 consecutive words of it: 'bia heineken lon' in 'Bia Heineken lon 330ml') gets the
          name's accents (see link);
        - protection: a word typed next to its neighbour in a name ('Plusssz Max', 'Sino SP D20', 'Galaxy Z') is part
          of that name: no Telex decoding, abbreviation or context rule touches it."""
        idx = defaultdict(lambda: defaultdict(Counter)); idx_a = defaultdict(lambda: defaultdict(Counter))
        self.name_bigrams = set(); self.name_upper = set()
        for n in {unicodedata.normalize("NFC", str(n)) for n in names}:
            lw = WORD.findall(n)
            self.name_upper.update(w for w in lw if len(w) > 1 and w.isupper())
            kl = [strip_accents(w).lower() for w in lw]; self.name_bigrams.update(zip(kl, kl[1:]))
            ws = re.findall(r"[^\W_]+", n)
            if len(ws) < 2 or len(ws) > 24: continue
            kw = [strip_accents(w).lower() for w in ws]; d = idx_a if n.isascii() else idx
            if len(" ".join(kw)) >= 6: d[" ".join(kw)][" ".join(w.lower() for w in ws)][" ".join(ws)] += 1
            for a in range(len(ws)):                                   # parts of >= 3 consecutive words
                for b in range(a + 3, min(len(ws) - (a == 0), a + 8) + 1):
                    d[" ".join(kw[a:b])][" ".join(w.lower() for w in ws[a:b])][" ".join(ws[a:b])] += 1
        # key -> (name words lowercase, word written lowercase in the data, accent-free name). A name written entirely
        # without accents ('Bia Heineken lon') only lends the accent-free words seen as such in accented text ('lon', not 'nuoc')
        self.ent = {}
        for d, plain in ((idx_a, True), (idx, False)):                  # accented names win over accent-free ones
            for k, sp in d.items():
                low = max(sp, key=lambda v: sum(sp[v].values()))        # the commonest spelling (accents)
                self.ent[k] = (low.split(), [any(v.split()[x][:1].islower() for v in sp[low]) for x in range(len(k.split()))], plain)
        self.maxn = max((len(k.split()) for k in self.ent), default=0)
        self.entities = bool(self.ent)

    def protected(self, words):
        """per word (letters only, as typed): inside a name of the data (see set_entities)."""
        nb = getattr(self, "name_bigrams", ()); keys = [strip_accents(w).lower() for w in words]
        return [bool(nb) and ((i > 0 and (keys[i - 1], keys[i]) in nb) or (i + 1 < len(keys) and (keys[i], keys[i + 1]) in nb))
                for i in range(len(keys))]

    def link(self, s, typed=()):
        """the data's accents on spans of s that spell one of its names (longest first, no overlaps). Only words the user
        did not accent change, and only their accents: the user's casing and separators stay. A span whose user-typed
        accents differ from the name's is another word ('khách Lê Bảo Hoàng' is not the label 'Khách lẻ'), and so is a
        capitalised word inside the sentence where the data writes the word in lowercase (unless the user capitalised the
        whole span: 'Bia Heineken Lon'). typed: words the user accented."""
        if not getattr(self, "entities", None): return s, []
        words = [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"[^\W_]+", s)]
        keys = [strip_accents(w).lower() for _, _, w in words]; typed = {word_sig(w) for w in typed}
        cands = []
        for i in range(len(words)):
            for j in range(min(len(words), i + self.maxn), i + 1, -1):
                e = self.ent.get(" ".join(keys[i:j]))
                if e is None: continue
                nw, lowinit, plain = e; qw = [w for _, _, w in words[i:j]]
                if any(word_sig(q) in typed and word_sig(q) != word_sig(x) for q, x in zip(qw, nw)): continue
                if not all(q[:1].isupper() or q[:1].isdigit() for q in qw) and \
                        any(i + x > 0 and q[:1].isupper() and not q.isupper() and lowinit[x] for x, q in enumerate(qw)): continue
                cands.append((j - i, i, j, nw, plain))
        used = [False] * len(words); repl = []
        for _, i, j, nw, plain in sorted(cands, key=lambda c: (-c[0], c[1])):
            if any(used[i:j]): continue
            for x in range(i, j): used[x] = True
            for (a, b, q), e in zip(words[i:j], nw):
                if word_sig(q) == word_sig(e) or word_sig(q) in typed: continue   # same word / the user's own accents
                if plain and e not in self.uni: continue
                # (an accent-free word of an accented name is accent-free on purpose: 'men vi sinh', 'Lay's', 'lau sàn')
                repl.append((a, b, _recase(q, e)))
        ch = [[s[a:b], n, "entity"] for a, b, n in sorted(repl)]
        for a, b, n in sorted(repl, reverse=True): s = s[:a] + n + s[b:]
        return s, ch

    @staticmethod
    def entity_texts(biz, max_unique=5000):
        """names written in a business's own data (items, groups, customers, staff ...): its text columns. The accented
        ones teach the vocabulary and are linked; all of them protect their words (set_entities)."""
        out = []
        for name, df in biz.tables.items():
            for c in df.columns:
                if c.startswith("_"): continue
                import pandas as pd
                if not (df[c].dtype == object or pd.api.types.is_string_dtype(df[c])): continue   # pandas 3: 'str' dtype
                vals = df[c].dropna().astype(str).unique()
                if len(vals) > max_unique: continue
                out += [v for v in vals if len(v) >= 3]
        return out

    @classmethod
    def for_split(cls, split_dir, train_questions=None, exclude_business=None, extra_texts=(), extra_weight=2):
        """vocabulary = the split's rulebook + accented training questions (never test answers) + extra texts (the
        business's own entity names, weighted: they decide the accents of product / customer names)."""
        texts = [open(os.path.join(split_dir, "rulebook.md"), encoding="utf-8").read()] + [t for t in extra_texts if not t.isascii() for _ in range(extra_weight)]
        ph = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qnorm_phrases.txt")
        if os.path.isfile(ph):                                         # general shop phrasing (weight 3), see the file
            texts += [l for l in open(ph, encoding="utf-8").read().replace(";", "\n").splitlines() if l.strip() and not l.startswith("#")] * 3
        tq = train_questions or os.path.join(os.path.dirname(os.path.abspath(__file__)), "qnorm_train_questions.txt")
        if os.path.isfile(tq):
            for l in open(tq, encoding="utf-8"):
                b, _, q = l.rstrip("\n").partition("\t")
                if b != exclude_business: texts.append(q)
        nz = cls(texts); nz.set_entities(extra_texts); return nz

    # -- accent restoration
    def _lp(self, prev, w):
        pu = (self.uni[w] + 0.1) / (self.N + 0.1 * len(self.uni))
        if prev is None: return math.log(pu)
        return math.log((self.bi[(prev, w)] + 2.0 * pu) / (self.uni[prev] + 2.0))

    def restore(self, words):
        """Viterbi over accented candidates of accent-free lowercase words; other words are fixed."""
        cands = []
        for w in words:
            if any(ord(c) > 127 for c in w) or w not in self.forms and w not in self.uni: cands.append([w])
            else:
                cs = list(self.forms.get(w, {}).keys()) or [w]
                cands.append(cs if (w in cs or w not in self.uni) else cs + [w])
        best = [{c: (self._lp(None, c), None) for c in cands[0]}]
        for i in range(1, len(words)):
            col = {}
            for c in cands[i]:
                p, s = max(((best[-1][pc][0] + self._lp(pc, c), pc) for pc in best[-1]), key=lambda x: x[0])
                col[c] = (p, s)
            best.append(col)
        out = [max(best[-1], key=lambda c: best[-1][c][0])]
        for i in range(len(words) - 1, 0, -1): out.append(best[i][out[-1]][1])
        return out[::-1]

    def restore_only(self, text):
        """accents restored on an accent-free sentence, words unchanged (measurement of the bigram model)."""
        parts = re.split(r"([^\W\d_]+)", unicodedata.normalize("NFC", text)); ws = parts[1::2]
        if ws:
            rest = self.restore([w.lower() for w in ws])
            parts[1::2] = self._context_fixes([_case_like(w, r) for w, r in zip(ws, rest)], ws, restored=True, seps=parts[0::2])
        return "".join(parts)

    # -- context rules after accent restoration (each limited to a safe context; see question_cleaning/README.md)
    @staticmethod
    def _context_fixes(words, typed, restored, seps=None, protect=None):
        """seps[i] = the text right before word i (numbers live there: '50 k' is 50 thousand, not 'không');
        protect[i]: word i belongs to a name of the data (never rewritten)."""
        out = list(words); low = [w.lower() for w in out]; n = len(out); seps = seps or [""] * (n + 1)
        for i, w in enumerate(low):
            nxt = " ".join(low[i + 1:i + 4]); prv = low[i - 1] if i else ""
            if protect and protect[i]: continue
            if w == "nhiêu" and prv != "bao" and re.search(r"\b(nhất|hơn|hay ít)\b", nxt): out[i] = _case_like(out[i], "nhiều")
            elif w == "nhiêu" and prv != "bao": out[i] = _case_like(out[i], "bao nhiêu")   # colloquial 'nhiêu' = 'bao nhiêu'
            elif w in ("hà", "ha") and i == n - 1 and typed[i][:1].islower(): out[i] = "hả"   # final question particle, not a name
            elif w == "lệ" and restored and prv not in ("tỉ", "tỷ") and typed[i].isascii(): out[i] = _case_like(out[i], "lẻ")
            elif w == "j" and i > 0 and typed[i] == "j": out[i] = "gì"           # lowercase only: 'Vitamin K', 'Galaxy J'
            elif w == "k" and i > 0 and typed[i] == "k" and not re.search(r"\d\s*$", seps[i]): out[i] = "không"
            elif w == "len" and re.search(r"\d\s*(đ|d|đồng|dong|k)?\s*$", (seps[i - 1] + low[i - 1] if i and low[i - 1] in ("đ", "đồng", "d", "dong", "k") else "") + seps[i]): out[i] = "lên"
            elif w == "tạ" and i == n - 1: out[i] = "ta"                           # final particle 'ta', not 'tạ' (100 kg)
            elif w == "hồng" and i == n - 1 and typed[i] == "hong" and "có" in low[:i] and not _QWORDS.search(" ".join(low[:i])):
                out[i] = "không"                                                   # 'có ... hong' = 'có ... không'
            elif w == "nhiều" and restored and typed[i].lower() == "nhieu" and prv != "bao" and \
                    not _QWORDS.search(" ".join(low[:i] + low[i + 1:])): out[i] = _case_like(typed[i], "bao nhiêu")
            # ^ a question needs a question word: 'bán được nhieu tiền' asks 'bao nhiêu'; 'nhiều ... nhất / hơn' stays
            elif i == 0 and typed[i] in ("e", "E") and n > 1: out[i] = _case_like(typed[i], "ê")   # opening 'e' = 'ê' (hey), not 'ế'
            elif w == "chi" and i == n - 1 and prv == "là": out[i] = "gì"            # central 'là chi' = 'là gì'
            # context words, only on words typed WITHOUT accents (the user's accents are never overridden)
            elif not typed[i].isascii(): continue
            elif w == "mọi" and i + 1 < n and low[i + 1] == "loại" and re.search(r"\bbao nhiêu\b", " ".join(low[i + 2:i + 5])): out[i] = _case_like(out[i], "mỗi")
            elif w == "số" and i + 1 < n and low[i + 1] == "ngày" and re.fullmatch(r"\s*\d+\s*", seps[i + 1] or "") \
                    and re.search(r"\bvới\b", " ".join(low[i + 2:])): out[i] = _case_like(out[i], "so")   # 'so 30 ngày ... với 30 ngày'
            elif w == "tục" and re.search(r",\s*$", seps[i]): out[i] = "tức"                          # ', tức hạn dùng từ ...'
            elif w == "nhưng" and i + 1 < n and low[i + 1] in ("đơn", "mặt", "sản", "khách", "lô", "món", "nhân", "hàng", "ngày"): out[i] = _case_like(out[i], "những")
            elif w == "vị" and i + 1 < n and low[i + 1] in ("giao", "lỗi", "hết", "mua", "khách", "sai", "hỏng", "trễ", "hư"): out[i] = _case_like(out[i], "vì")
            elif w == "mã" and i + 2 < n and low[i + 1] == "khách" and low[i + 2] in ("chưa", "không", "vẫn", "còn"): out[i] = _case_like(out[i], "mà")
        return out

    # -- the whole cleaning
    def clean(self, q, as_of=None):
        changes = []
        s = unicodedata.normalize("NFC", str(q))
        s2 = "".join(c for c in s if unicodedata.category(c) not in ("So", "Sk", "Cn") and c not in "️‍")
        if s2 != s: changes.append([s, s2, "symbols"]); s = s2
        s = re.sub(r"\s+", " ", s).strip()
        for k in MULTI:                                                 # multi-word abbreviations
            s2 = re.sub(r"(?<!\w)" + re.escape(k) + r"(?!\w)", ABBR[k], s, flags=re.I)
            if s2 != s: changes.append([k, ABBR[k], "abbr"]); s = s2
        parts = re.split(r"([^\W\d_]+)", s)                           # words at odd positions
        words = parts[1::2]; prev = None; outw = []
        prot = self.protected(words)                                    # words of the data's names: as typed
        glued = [bool(re.search(r"\d$", parts[2 * i]) or re.match(r"\d", parts[2 * i + 2])) for i in range(len(words))]
        # ^ letters attached to a number are a unit or a code ('2.92m', '60x60', 'CT-11A', 'NV01', '35.000đ'): as typed
        for i, w in enumerate(words):
            lw = w.lower(); r = None; rule = None
            inner_cap = i > 0 and w[:1].isupper() and not re.search(r"[.?!:;]\s*$", parts[2 * i])   # a capitalised word inside
            keep = prot[i] or glued[i] or (len(w) > 1 and w.isupper() and w in getattr(self, "name_upper", ()))   # the sentence: a name
            if glued[i] and lw == "d" and re.search(r"\d{3}$", parts[2 * i]): r, rule = "đ", "unit"      # '35.000d' = '35.000đ'
            elif keep: pass
            elif lw in ABBR: r, rule = ABBR[lw], "abbr"
            elif lw not in self.vocab and lw not in self.forms and not inner_cap:
                ts = [t for t in telex_all(lw) if t in self.vocab]      # only an exact known accented word
                if ts: r, rule = max(ts, key=lambda t: self.uni[t]), "telex"
            if keep: pass
            elif r is None and lw in ("ni",) and prev in DIALECT_NI: r, rule = "này", "dialect"
            elif r is None and (lw == "mô" or (lw == "mo" and prev in DIALECT_MO)): r, rule = "nào", "dialect"
            elif r is None and lw in ("rứa", "chừ"): r, rule = {"rứa": "vậy", "chừ": "giờ"}[lw], "dialect"   # not 'răng' (= teeth)
            elif r is None and not inner_cap and lw not in self.vocab and (not lw.isascii() or lw not in self.forms):
                nxt = words[i + 1].lower() if i + 1 < len(words) else None
                r = self.typo_fix(lw, prev, nxt)
                if r is not None: rule = "typo"
            if r is not None: changes.append([w, r, rule]); outw.append(_case_like(w, r))
            else: outw.append(w)
            prev = (r or lw).split()[-1].lower()
        # accent restoration on the accent-free words of the (possibly expanded) sentence
        toks = " ".join(outw).split(" ") if outw else []
        flat = [t.lower() for t in toks]
        typed = [w for w in words if w.lower() not in ABBR and w.lower() not in ("đ", "đồng")]   # as typed: abbreviations and
                                                                           # the currency sign ('35.000đ') are no evidence of accents
        accented = sum(1 for w in typed if not w.isascii()) / max(1, len(typed))
        if flat and accented < 0.15:                                       # typed without accents: restore them
            rest = self.restore(flat)
        else: rest = flat
        k = 0; final = []
        for i, w in enumerate(outw):
            n = len(w.split(" ")); new = rest[k:k + n]; k += n
            if glued[i]: final.append(w); continue
            orig = w.split(" ")
            fixed = " ".join(_case_like(o, x) for o, x in zip(orig, new))
            if fixed != w: changes.append([w, fixed, "accent"])
            final.append(fixed)
        final = self._context_fixes(final, [w for w in outw], restored=flat and accented < 0.15, seps=parts[0::2],
                                    protect=[a or b for a, b in zip(prot, glued)])
        for w0, w1 in zip(outw, final):
            if w0 != w1 and not any(c[1] == w1 for c in changes): changes.append([w0, w1, "context"])
        parts[1::2] = final
        s = "".join(parts)
        for rx, f in MONTHS:                                            # month shorthands
            s2 = rx.sub(f, s)
            if s2 != s: changes.append(["month", s2, "month"]); s = s2
        s = re.sub(r"\s+", " ", s).strip()
        s, ch = self.link(s, typed=[w for w in words if not w.isascii()]); changes += ch   # the data's accents on its names
        out = {"clean": s, "changes": changes, "key": strip_accents(s).lower(), "period": None}
        if as_of is not None:
            try:
                import qparse
                p, code = qparse.parse_period(s, as_of)
                if p is not None: out["period"] = {"start": str(p.start.date()), "end": str(p.end.date()), "code": code}
            except Exception: pass
        return out


def build_train_vocab(train_split, out_path):
    """the accented training questions used as vocabulary (one 'business<TAB>question' per line)."""
    with open(out_path, "w", encoding="utf-8") as f:
        for l in open(os.path.join(train_split, "questions.jsonl"), encoding="utf-8"):
            q = json.loads(l)
            if any(ord(c) > 127 for c in q["question"]): f.write(f"{q['business_id']}\t{q['question'].strip()}\n")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    split, out = sys.argv[1], sys.argv[2]
    nz = Normalizer.for_split(split)
    with open(out, "w", encoding="utf-8") as f:
        for l in open(os.path.join(split, "questions.jsonl"), encoding="utf-8"):
            q = json.loads(l); r = nz.clean(q["question"], q["as_of_date"])
            f.write(json.dumps({"id": q["id"], "business_id": q["business_id"], "as_of_date": q["as_of_date"], "question": q["question"], **r},
                               ensure_ascii=False) + "\n")
