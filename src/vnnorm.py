#!/usr/bin/env python3
"""Normalise informal Vietnamese chat questions for the LLM (generic language rules, nothing about any business):
  - whole-token chat abbreviations: hqua -> hôm qua, hnay -> hôm nay, dthu -> doanh thu, bnhiu -> bao nhiêu, đc -> được,
    th7/2025 -> tháng 7/2025, trc -> trước, z -> vậy, ...
  - telex typing slips: tieenf -> tiền, dduwowcj -> được, nawm -> năm (accepted only when the decoded word is a known
    Vietnamese word of the vocabulary: rulebook + common words below).
The agent shows the normalised form next to the original question (the original stays authoritative).
Usage: python3 vnnorm.py "<question>" """
import re, sys, unicodedata

ABBR = {
    "hqua": "hôm qua", "hwa": "hôm qua", "homqua": "hôm qua", "hum": "hôm", "hnay": "hôm nay", "homnay": "hôm nay",
    "dthu": "doanh thu", "dthu.": "doanh thu",
    "bn": "bao nhiêu", "bnhiu": "bao nhiêu", "bnhieu": "bao nhiêu", "bnh": "bao nhiêu", "nhiu": "nhiêu", "nhìu": "nhiều",
    "bao nhiu": "bao nhiêu",
    "đc": "được", "dc": "được", "đươc": "được", "duoc": "được",
    "ko": "không", "hok": "không", "hông": "không", "kô": "không", "k0": "không",
    "trc": "trước", "truoc": "trước", "vs": "với", "z": "vậy", "zậy": "vậy", 
    "sp": "sản phẩm", "đh": "đơn hàng", "nv": "nhân viên", "ntn": "như thế nào", "nhứt": "nhất", "nhứ": "nhất",
    "kòn": "còn", "kả": "cả", "wán": "quán", "mik": "mình", "cko": "cho", "xố": "số", "kó": "có", "chừ": "bây giờ",
    "tuan": "tuần", "thang": "tháng", "ngay": "ngày", "hom": "hôm", "roi": "rồi", 
    "trog": "trong", "mí": "mấy", 
}
# "th7/2025", "t11/2025", "th 7", "T12/2025": month abbreviations before a number
MONTH = re.compile(r"\b(th|t|thg)\s?(?=\d{1,2}\s*[/\-]\s*\d{2,4}\b|\d{1,2}\b(?!\s*(giờ|h\b|ngày)))", re.I)

COMMON = """bao nhiêu tháng năm ngày tuần hôm qua nay doanh thu tiền bán được đơn hàng khách tổng số trước sau với của cửa tiệm quán
nào nhất nhiều ít cao thấp tăng giảm trung bình phần trăm tỉ lệ huỷ hủy trả nợ còn hết giá nhập vốn lãi lời mặt món nhóm
cho biết xem giúp giùm mình tôi em anh chị vậy nhỉ không có là gì mấy những các mỗi một hai ba từ đến tới trong ngoài liền
đầu cuối kỳ quý tết thứ chủ nhật sáng chiều tối ca giờ lượt phiếu xe thợ máy điện thoại nhân viên người chiếm so sánh cùng
lần đợt giao chưa đã đang rồi thì mà và hay hoặc bằng hơn kém thêm bớt khoảng gần nhất chạy ế tồn kho thuốc kê đơn hoạt chất
lô hạn dùng size gói hộp chai lon thùng bao cây viên vỉ tuýp ly ca phê trà sữa bánh nước đá cơm mì cháo phở bỏ tính đếm liệt kê
những ai được đắt rẻ trị giá vào ra mua chi tiêu nhà cung cấp ship phí điểm thành viên voucher đăng ký góp cũ mới biển dòng
bảo hành sửa làm việc đơn vị lẻ sỉ quen trung thành sộp đổi ý ưa chuộng chăm chỉ cao điểm khung""".split()

_TONE = {"s": "́", "f": "̀", "r": "̉", "x": "̃", "j": "̣"}
_VOW = [("aa", "â"), ("aw", "ă"), ("ee", "ê"), ("oo", "ô"), ("ow", "ơ"), ("uw", "ư"), ("dd", "đ")]


def _telex(w):
    """Decode one telex-typed token ('tieenf' -> 'tiền'); None if it does not look like telex."""
    s = w.lower(); tone = ""
    if len(s) > 2 and s[-1] in _TONE and re.search(r"[aeiouyăâêôơư]|aa|aw|ee|oo|ow|uw", s[:-1]): tone, s = _TONE[s[-1]], s[:-1]
    t = s
    for a, b in _VOW: t = t.replace(a, b)
    t = t.replace("uơ", "ươ")
    if t == s and not tone: return None
    if tone:                                                          # tone mark on the main vowel
        vs = [i for i, ch in enumerate(t) if ch in "aeiouyăâêôơư"]
        if not vs: return None
        pri = [i for i in vs if t[i] in "ăâêôơư"]
        i = pri[-1] if pri else (vs[-2] if len(vs) >= 2 and vs[-1] == len(t) - 1 else vs[-1])
        t = t[:i + 1] + tone + t[i + 1:]
    return unicodedata.normalize("NFC", t)


_VOCAB = None


def vocab(extra_text=""):
    global _VOCAB
    if _VOCAB is None or extra_text:
        words = set(COMMON) | set(re.findall(r"[^\W\d_]+", unicodedata.normalize("NFC", extra_text.lower())))
        _VOCAB = {w for w in words if any(ord(c) > 127 for c in w) or w in COMMON}
    return _VOCAB


def normalize(q, extra_text=""):
    V = vocab(extra_text) if extra_text or _VOCAB is None else _VOCAB
    s = unicodedata.normalize("NFC", q)
    s = MONTH.sub("tháng ", s)
    s = re.sub(r"\bbao\s+nhiu\b", "bao nhiêu", s, flags=re.I)

    def tok(m):
        w = m.group(0); lw = w.lower()
        if lw in ABBR: return ABBR[lw]
        if lw not in V:
            t = _telex(lw)
            if t and t in V: return t
        return w
    return re.sub(r"(?<!\w)[^\W\d_]+(?!\w)", tok, s)                    # whole words only ("NV001" stays)


if __name__ == "__main__":
    print(normalize(" ".join(sys.argv[1:])))
