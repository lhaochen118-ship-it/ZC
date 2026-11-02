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
}
bad = [(q, nz.clean(q)["clean"], want) for q, want in cases.items() if nz.clean(q)["clean"].lower() != want.lower()]
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
print("rules:", len(cases) - len(bad), "/", len(cases), "| job:", acc)
print("PASS" if not bad else "FAIL")
