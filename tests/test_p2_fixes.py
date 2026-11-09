#!/usr/bin/env python3
"""Plan 2 round 1 (docs/12_PLAN2.md WP-P2-01..03): general fixes of the hints, the question cleaning and the vote, on
synthetic questions written for these rules (no question of any test split).
Usage: test_p2_fixes.py <train_split_dir>"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import hints, qnorm, qparse, hybrid

split = sys.argv[1]
bad = []
def check(name, got, want):
    if got != want: bad.append((name, got, want))

# 1. answer type: 'máy' (a phone) is not a second 'mấy'; one total over several groups is a number
K = lambda s: qparse.untelex(qparse.K(s))
for q, want in [("Tháng 5 bán được mấy máy?", "number"), ("bán được bao nhiêu máy Oppo tháng này", "number"),
                ("tháng 3 bán được mấy máy cũ và mấy máy mới", "list"), ("doanh thu tháng 6 và tháng 7 mỗi tháng bao nhiêu", "list"),
                ("tháng này bán mấy chiếc, gộp cả máy mới với máy cũ", "number"), ("Tháng 6/25, tháng 7/25 hay tháng 8/25, tháng nào doanh thu cao nhất?", "date"),
                ("3 ngày nào bán nhiều tiền nhất tháng này", "list"),
                ("Mấy lô còn tồn hạn dùng tới hết 06/12/2026 đang đọng bao nhiêu tiền vốn?", "number"),
                ("tháng này số ly size L bán được là bao nhiêu, tính mọi loại đồ uống?", "number"),
                ("doanh thu mỗi loại hình thức thanh toán bao nhiêu", "list"),
                ("khách còn nợ bao nhiêu ở nhóm tuổi nợ 0–7 ngày, 8–15 ngày và trên 15 ngày", "list")]:
    check(f"answer_type {q}", hints.answer_type(K(q), q), want)
check("answer_type accent-free máy", hints.answer_type(K("thang nay ban duoc bao nhieu chiec may cu"), None), "number")

# 2. rulebook terms: accents compared, head nouns optional, unit placeholder
rows = hints.rulebook_rows("| Khái niệm | Định nghĩa |\n|---|---|\n| Mặt hàng được ưa chuộng nhất | có mặt trong nhiều đơn nhất |\n"
                           "| Dòng máy | phần tên trước GB |\n| Bán được bao nhiêu <đơn vị> | cộng số lượng các mặt hàng có đơn vị cơ bản là <đơn vị> |\n"
                           "| Tháng này | từ ngày 1 đến hết hôm nay |\n| Doanh thu một đơn | tổng thành tiền |\n")
terms = lambda q, ents=None: [t["term"] for t in hints.rule_terms_of(q, rows, ents)]
check("ưa chuộng", terms("tháng 7 hàng nào được khách ưa chuộng nhất?"), ["Mặt hàng được ưa chuộng nhất"])
check("mấy ... đồng is not Dòng máy", terms("tháng 3 có mấy ngày bán từ 7.000.000 đồng trở lên?"), [])
check("unit placeholder", terms("tuần này bán được bao nhiêu gói?", [{"kind": "base_unit", "data_value": "gói"}]), ["Bán được bao nhiêu <đơn vị>"])
check("no unit linked", terms("tuần này bán được bao nhiêu tiền?"), [])
check("hôm nay is not tháng này", terms("tháng 4/2026, tính đến hôm nay"), [])
check("doanh thu alone is not một đơn", terms("doanh thu tháng này bao nhiêu"), ["Tháng này"])

# 3. question cleaning: 'quán' (the shop) vs a name, 'máy' vs 'mấy', 'nhiều' in a selection, 'tháng này'
nz = qnorm.Normalizer.for_split(split)
for q, want in [("doanh thu cua quan thang 12/2025 duoc bao nhieu", "doanh thu của quán tháng 12/2025 được bao nhiêu"),
                ("binh quan 1 don duoc bao nhieu", "bình quân 1 đơn được bao nhiêu"),
                ("ban duoc bao nhieu may cu", "bán được bao nhiêu máy cũ"),
                ("ton kho duoc may hop chan va may goi le", "tồn kho được mấy hộp chẵn và mấy gói lẻ"),
                ("ncc nao nhiu tien nhut", "nhà cung cấp nào nhiều tiền nhất"),
                ("Th nay ban duoc bao nhieu", "tháng này bán được bao nhiêu"),
                ("nam nay ban duoc bao nhieu", "năm nay bán được bao nhiêu"),
                ("cuoi ngay hom nay co may mat hang dang chay hang", "cuối ngày hôm nay có mấy mặt hàng đang cháy hàng")]:
    check(f"clean {q}", nz.clean(q)["clean"].lower(), want.lower())
# a half-accented question gets the accents of its plain words (only with a neighbour's support), particles stay
c = nz.clean("tính tới hết ngày 9/6/2025, tien khach còn nợ là bao nhiêu nha")["clean"]
check("mixed: tiền khách", "tiền khách còn nợ" in c, True); check("mixed: nha stays", c.endswith("nha"), True)

# 4. format of the hinted type, review of missing data only (mode 2)
check("fmt list->number", hybrid.fmt_fix({"type": "list", "answer": [24000000]}, "number"), {"type": "number", "answer": 24000000})
check("fmt keeps 2 numbers", hybrid.fmt_fix({"type": "list", "answer": [1, 2]}, "number"), {"type": "list", "answer": [1, 2]})
check("fmt iso -> date", hybrid.fmt_fix({"type": "string", "answer": "2025-07"}, "date"), {"type": "date", "answer": "2025-07"})
check("fmt name stays", hybrid.fmt_fix({"type": "string", "answer": "Cà phê"}, "date"), {"type": "string", "answer": "Cà phê"})
check("review 2 flags only", hybrid.review_msg({"missing_data": []}, {"type": "number", "answer": 0}, ["zero result"], 2), None)
check("review 2 missing data", bool(hybrid.review_msg({"missing_data": ["reason"]}, {"type": "number", "answer": 3}, [], 2)), True)
check("review 1 flags", bool(hybrid.review_msg({"missing_data": []}, {"type": "number", "answer": 0}, ["zero result"], 1)), True)

# 5. round 2 (WP-P2-05): what is counted, thresholds (money in đồng; 'từ X đến Y' is a range, 'nhiều hơn 30' a comparison)
for q, c, th in [("tháng 10 có mấy bữa quán bán được từ 9.100.000 đồng trở lên?", "ngày", [{"op": ">=", "value": 9100000, "of": "đồng"}]),
                 ("có bao nhiêu mặt hàng có mặt trong ít nhất 50 đơn", "mặt hàng", [{"op": ">=", "value": 50, "of": "đơn"}]),
                 ("mấy ngày bán từ 10 củ trở lên", "ngày", [{"op": ">=", "value": 10000000, "of": "đồng"}]),
                 ("trung bình mỗi thứ Tư sửa mấy lượt xe", "lượt xe / phiếu", []),
                 ("từ 01/07/2025 đến 15/07/2025 bán bao nhiêu tiền", None, []),
                 ("30 ngày sau bán nhiều hơn 30 ngày trước mấy %", None, [])]:
    t = K(q); check(f"counted {q}", hints.counted(t), c); check(f"thresholds {q}", hints.thresholds(t), th)

# 6. round 3: a repair job after a work verb (never 'tháng' read as 'thắng'), a catalogue size ('số 4', 'size M')
import pandas as pd
class _Shop:
    O = pd.DataFrame({"work": ["thay nhớt, vá ruột"]})
    C = {"lines": pd.DataFrame({"size": ["Số 4", "Size M", None]})}
ents = lambda q: [(e["kind"], e["data_value"]) for e in hints.job_size_entities(K(q), _Shop())]
check("job after a verb", ents("tháng 9 tiệm thay dầu láp cho bao nhiêu lượt xe"), [("job", "thay nhớt láp")])
check("tháng is not thắng", ents("tháng 8 thợ An lãnh bao nhiêu tiền"), [])
check("size số 4", ents("tháng này bán bao nhiêu lon sữa bột số 4"), [("size", "Số 4")])
check("size M", ents("bỉm size M bán được mấy bịch"), [("size", "Size M")])

for b in bad: print("MISMATCH", b)
print("PASS" if not bad else "FAIL")
