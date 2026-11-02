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


class Normalizer:
    def __init__(self, texts):
        """texts: accented Vietnamese text (rulebook, training questions); only sentences with accents are learnt."""
        self.uni, self.bi, self.forms = Counter(), Counter(), defaultdict(Counter)
        for t in list(texts) + [" ".join(COMMON)]:
            for line in re.split(r"[\n.;:!?()|]+", unicodedata.normalize("NFC", t)):
                ws = [w.lower() for w in WORD.findall(line)]
                if not ws or not any(any(ord(c) > 127 for c in w) for w in ws): continue
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

    @staticmethod
    def entity_texts(biz, max_unique=5000):
        """accented names written in a business's own data (items, groups, customers, staff ...): its text columns."""
        out = []
        for name, df in biz.tables.items():
            for c in df.columns:
                if c.startswith("_"): continue
                import pandas as pd
                if not (df[c].dtype == object or pd.api.types.is_string_dtype(df[c])): continue   # pandas 3: 'str' dtype
                vals = df[c].dropna().astype(str).unique()
                if len(vals) > max_unique: continue
                out += [v for v in vals if len(v) >= 3 and not v.isascii()]
        return out

    @classmethod
    def for_split(cls, split_dir, train_questions=None, exclude_business=None, extra_texts=(), extra_weight=2):
        """vocabulary = the split's rulebook + accented training questions (never test answers) + extra texts (the
        business's own entity names, weighted: they decide the accents of product / customer names)."""
        texts = [open(os.path.join(split_dir, "rulebook.md"), encoding="utf-8").read()] + [t for t in extra_texts for _ in range(extra_weight)]
        tq = train_questions or os.path.join(os.path.dirname(os.path.abspath(__file__)), "qnorm_train_questions.txt")
        if os.path.isfile(tq):
            for l in open(tq, encoding="utf-8"):
                b, _, q = l.rstrip("\n").partition("\t")
                if b != exclude_business: texts.append(q)
        return cls(texts)

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
    def _context_fixes(words, typed, restored, seps=None):
        """seps[i] = the text right before word i (numbers live there: '50 k' is 50 thousand, not 'không')."""
        out = list(words); low = [w.lower() for w in out]; n = len(out); seps = seps or [""] * (n + 1)
        for i, w in enumerate(low):
            nxt = " ".join(low[i + 1:i + 4]); prv = low[i - 1] if i else ""
            if w == "nhiêu" and prv != "bao" and re.search(r"\b(nhất|hơn|hay ít)\b", nxt): out[i] = _case_like(out[i], "nhiều")
            elif w in ("hà", "ha") and i == n - 1 and typed[i][:1].islower(): out[i] = "hả"   # final question particle, not a name
            elif w == "lệ" and restored and prv not in ("tỉ", "tỷ") and typed[i].isascii(): out[i] = _case_like(out[i], "lẻ")
            elif w == "j" and i > 0: out[i] = "gì"
            elif w == "k" and i > 0 and not re.search(r"\d\s*$", seps[i]): out[i] = "không"
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
        for w in words:
            lw = w.lower(); r = None; rule = None
            if lw in ABBR: r, rule = ABBR[lw], "abbr"
            elif lw not in self.vocab and lw not in self.forms:
                ts = [t for t in telex_all(lw) if t in self.vocab]      # only an exact known accented word
                if ts: r, rule = max(ts, key=lambda t: self.uni[t]), "telex"
            if r is None and lw in ("ni",) and prev in DIALECT_NI: r, rule = "này", "dialect"
            elif r is None and (lw == "mô" or (lw == "mo" and prev in DIALECT_MO)): r, rule = "nào", "dialect"
            elif r is None and lw in ("rứa", "chừ"): r, rule = {"rứa": "vậy", "chừ": "giờ"}[lw], "dialect"   # not 'răng' (= teeth)
            if r is not None: changes.append([w, r, rule]); outw.append(_case_like(w, r))
            else: outw.append(w)
            prev = (r or lw).split()[-1].lower()
        # accent restoration on the accent-free words of the (possibly expanded) sentence
        toks = " ".join(outw).split(" ") if outw else []
        flat = [t.lower() for t in toks]
        typed = [w for w in words if w.lower() not in ABBR]                # words as typed (abbreviations aside)
        accented = sum(1 for w in typed if not w.isascii()) / max(1, len(typed))
        if flat and accented < 0.15:                                       # typed without accents: restore them
            rest = self.restore(flat)
        else: rest = flat
        k = 0; final = []
        for w in outw:
            n = len(w.split(" ")); new = rest[k:k + n]; k += n
            orig = w.split(" ")
            fixed = " ".join(_case_like(o, x) for o, x in zip(orig, new))
            if fixed != w: changes.append([w, fixed, "accent"])
            final.append(fixed)
        final = self._context_fixes(final, [w for w in outw], restored=flat and accented < 0.15, seps=parts[0::2])
        for w0, w1 in zip(outw, final):
            if w0 != w1 and not any(c[1] == w1 for c in changes): changes.append([w0, w1, "context"])
        parts[1::2] = final
        s = "".join(parts)
        for rx, f in MONTHS:                                            # month shorthands
            s2 = rx.sub(f, s)
            if s2 != s: changes.append(["month", s2, "month"]); s = s2
        s = re.sub(r"\s+", " ", s).strip()
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
