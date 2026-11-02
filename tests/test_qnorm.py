#!/usr/bin/env python3
"""Question cleaning (src/qnorm.py rules, src/qnorm_llm.py check / merge / job) without a GPU.
Usage: test_qnorm.py <train_split_dir> <scratch_dir>"""
import os, sys, json, shutil, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import qnorm, qnorm_llm

split, scratch = sys.argv[1], sys.argv[2]
nz = qnorm.Normalizer.for_split(split)
cases = {   # rule cases found while reviewing the 3000 training questions (generic, not tied to a business)
    "bữa qua toongr cộng bán dduwowcj bnhiu lon": "bữa qua tổng cộng bán được bao nhiêu lon",
    "gias trị trung bình một đơn tháng 9/2025": "giá trị trung bình một đơn tháng 9/2025",
    "Th3/2026 thợ mô làm nhiều lựơt xe nhất?": "tháng 3/2026 thợ nào làm nhiều lượt xe nhất?",
    "thang 2/2026 ngay nao cua hang ban duoc nhieu tien nhat": "tháng 2/2026 ngày nào cửa hàng bán được nhiều tiền nhất",
    "ton kho gaviscon hop 24 goi cuoi ngay hom nay duoc may hop chan va may goi le": "tồn kho gaviscon hộp 24 gói cuối ngày hôm nay được mấy hộp chẵn và mấy gói lẻ",
    "khach nao mua nhieu don nhat thang 12/2025 ha": "khách nào mua nhiều đơn nhất tháng 12/2025 hả",
    "thang 6/2025 top 5 hang ban chay nhat la j vay": "tháng 6/2025 top 5 hàng bán chạy nhất là gì vậy",
    "gia 50 k mot chai la bao nhieu": "giá 50 k một chai là bao nhiêu",
    "Ngày 01/11/2025 cửa hàng bán Kem đánh răng P/S 180g": "Ngày 01/11/2025 cửa hàng bán Kem đánh răng P/S 180g",
    "Tháng 6/2026 khách Tô Phúc Huy ghi nợ thêm bao nhiêu": "tháng 6/2026 khách Tô Phúc Huy ghi nợ thêm bao nhiêu",
    "giờ mở cửa 1/1 là mấy giờ": "giờ mở cửa 1/1 là mấy giờ",
    "Cho tôi hỏi 6-2026 trung bình mỗi đơn": "Cho tôi hỏi tháng 6/2026 trung bình mỗi đơn",
    "ê trong tháng 2/2025, số ly size l bán được là nhiêu": "ê trong tháng 2/2025, số ly size l bán được là bao nhiêu",
    "ngay 10/09/2025 soda tang gia tu 35.000đ len 37.000đ": "ngày 10/09/2025 soda tăng giá từ 35.000đ lên 37.000đ",
    "doanh thu cac don tra bang tien mat va chuyen khoan moi loai bao nhieu": "doanh thu các đơn trả bằng tiền mặt và chuyển khoản mỗi loại bao nhiêu",
    "so 30 ngay ke tu ngay tang gia voi 30 ngay ngay truoc": "so 30 ngày kể từ ngày tăng giá với 30 ngày ngay trước",
    "thang 3/2026 co bao nhieu don bi huy vi giao tre": "tháng 3/2026 có bao nhiêu đơn bị huỷ vì giao trễ",
    "cac don dat da hon 7 ngay ma khach chua tra het": "các đơn đặt đã hơn 7 ngày mà khách chưa trả hết",
    "tính mọi loại đồ uống bao nhiêu ly": "tính mọi loại đồ uống bao nhiêu ly",
}
bad = [(q, nz.clean(q)["clean"], want) for q, want in cases.items() if nz.clean(q)["clean"].lower() != want.lower()]
# names of the data (synthetic here): protected from Telex / abbreviations / context rules, linked for accents only,
# the user's casing and accents kept (exact comparison)
names = ["Plusssz Max Multivitamin viên sủi tuýp 20 viên", "Ống luồn dây điện Sino SP D20 cây 2.92m", "Khách lẻ", "Lê Bảo Hoàng",
         "BÁNH CROISSANT", "Bio-acimin Gold men vi sinh hộp 30 gói", "Vitamin K2 Mk7 hộp 30 viên", "Chống thấm Kova CT-11A tường 20kg",
         "Dầu gió xanh Con Ó chai 12ml"]
nzn = qnorm.Normalizer.for_split(split, extra_texts=names)
name_cases = {
    "Tồn kho Plusssz Max Multivitamin viên sủi tuýp 20 viên còn bao nhiêu": "Tồn kho Plusssz Max Multivitamin viên sủi tuýp 20 viên còn bao nhiêu",
    "ke tu khi plusssz max multivitamin vien sui tuyp 20 vien len gia": "kể từ khi plusssz max multivitamin viên sủi tuýp 20 viên lên giá",
    "Số cây Ống luồn dây điện Sino SP D20 cây 2.92m bán đc": "Số cây Ống luồn dây điện Sino SP D20 cây 2.92m bán được",
    "tháng này khách Lê Bảo Hoàng nợ thêm bao nhiêu": "tháng này khách Lê Bảo Hoàng nợ thêm bao nhiêu",
    "giá nhập một hộp Bánh croissant là bao nhiêu": "giá nhập một hộp Bánh croissant là bao nhiêu",
    "ton kho bio-acimin gold men vi sinh hop 30 goi": "tồn kho bio-acimin gold men vi sinh hộp 30 gói",
    "vitamin k2 mk7 ban duoc bao nhieu hop": "vitamin k2 mk7 bán được bao nhiêu hộp",
    "so thung chong tham kova CT-11A tuong 20kg ban duoc": "số thùng chống thấm kova CT-11A tường 20kg bán được",
    "dau gio xanh con o chai 12ml gan het han nhat": "dầu gió xanh con ó chai 12ml gần hết hạn nhất",   # the data's 'Con Ó', not 'còn'
    "e doanh thu hom qua la bao nhieu": "ê doanh thu hôm qua là bao nhiêu",
}
bad += [(q, nzn.clean(q)["clean"], want) for q, want in name_cases.items() if nzn.clean(q)["clean"] != want]
cases.update(name_cases)
for b in bad: print("RULE MISMATCH", b)
# LLM check: only accents may be added; the user's accents and every word / number stay
C = qnorm_llm.check
assert C("ban duoc bao nhieu tien", "bán được bao nhiêu tiền")
assert not C("ban duoc bao nhieu tien", "bán được bao nhiêu tiền vậy"), "an added word must be rejected"
assert not C("thang 7/2025 doanh thu", "tháng 8/2025 doanh thu"), "a changed number must be rejected"
assert not C("tháng 7 doanh thu", "thăng 7 doanh thu"), "a word the user accented must not change"
assert not C("doanh thu", "doanh thu\nGiải thích: ..."), "explanations must be rejected"
m = qnorm_llm.merge("tháng 7 Coca Cola 330ml ban duoc bao nhieu", "tháng 7 Côca Côla 330ml bán được bao nhiêu")
assert m == "tháng 7 Coca Cola 330ml bán được bao nhiêu", m
# a question typed with accents keeps its deliberate accent-free words ('so số' = compare, 'ca chiều' = shift)
q = "Xem giúp so số đĩa cơm bán ra trong 7 ngày, với 7 ngày trước"
assert qnorm_llm.merge(q, "Xem giúp số số đĩa cơm bán ra trong 7 ngày, với 7 ngày trước", q) == q
# the job end to end with a fake LLM that returns the user's accents (accented questions) / the rules' text
mini = os.path.join(scratch, "mini_qn"); shutil.rmtree(mini, ignore_errors=True); os.makedirs(os.path.join(mini, "data"))
for f in ("rulebook.md", "calendar.csv"): shutil.copy(os.path.join(split, f), mini)
qs = [json.loads(l) for l in open(os.path.join(split, "questions.jsonl"), encoding="utf-8") if json.loads(l)["business_id"] == "biz_101"][:15]
os.symlink(os.path.abspath(os.path.join(split, "data", "biz_101")), os.path.join(mini, "data", "biz_101"))
open(os.path.join(mini, "questions.jsonl"), "w", encoding="utf-8").writelines(json.dumps(q, ensure_ascii=False) + "\n" for q in qs)
orig = {qnorm.strip_accents(q["question"]): q["question"] for q in qs}
rf = os.path.join(scratch, "fake_qn.py")
open(rf, "w", encoding="utf-8").write("import json\nORIG = " + repr(orig) + "\ndef responder(conv):\n    t = conv[-1]['content']\n    return ORIG.get(t, t)\n")
out = os.path.join(scratch, "out_qn")
r = subprocess.run([sys.executable, os.path.join(os.path.dirname(qnorm.__file__), "qnorm_llm.py"), mini, out, "--fake-responder", rf],
                   capture_output=True, text=True, timeout=600)
assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
acc = json.load(open(os.path.join(out, "accuracy.json")))
rows = [json.loads(l) for l in open(os.path.join(out, "questions_clean.jsonl"), encoding="utf-8")]
assert len(rows) == len(qs) and acc["llm_question_acc"] == 1.0, acc     # the fake LLM returns the true accents
# the copy of the module in question_cleaning/module/ (for the user to read) must be identical to src/
mod = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "question_cleaning", "module")
for f in ("qnorm.py", "qnorm_llm.py", "qnorm_phrases.txt", "qnorm_train_questions.txt"):
    a_, b_ = os.path.join(os.path.dirname(qnorm.__file__), f), os.path.join(mod, f)
    if not (os.path.isfile(b_) and open(a_, "rb").read() == open(b_, "rb").read()): bad.append(("module copy differs", f, "run question_cleaning/make.py"))
print("rules:", len(cases) - len(bad), "/", len(cases), "| job:", acc)
print("PASS" if not bad else "FAIL")
