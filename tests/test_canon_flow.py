#!/usr/bin/env python3
"""Library test of src/canon_llm.py + src/rules_first.py (the old stage 1.5, no longer in the submitted flow since the hybrid
framework of 08/11; kept as an offline tool, e.g. labelling paraphrases), on CPU with a fake model: two training questions
asked in a new way (restatements taken from a real run on Kaggle, v33) + one question asked as in training + one question
nobody can read. canon_llm restates every question; the rules read the restatements right (strict judge) and leave the
unreadable one.
Usage: test_canon_flow.py <train_split_dir> <scratch_dir>"""
import os, sys, json, shutil, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(HERE, "..", "src")
sys.path.insert(0, SRC)
import score

PARA = {"q_879535b41a21": ("Kiểm tra số gium trong tháng 2 năm 2026 của hàng bán được mấy kg nén cải hoa vàng.",
                           "Trong tháng 2 năm 2026 cửa hàng bán được mấy kg Nếp cái hoa vàng?"),
        "q_a0172ef25cac": ("Bạn ơi, tháng 12/2025 shop tổng cộng làm ra bao nhiêu tiền vậy?",
                           "Tháng 12/2025 cửa hàng bán được tổng cộng bao nhiêu tiền vậy?")}
FAKE = '''import json
MAP = %s
def responder(conv):
    u = conv[-1]["content"]
    if "CÂU CẦN CHUẨN HOÁ:" in u:
        q = u.split("CÂU CẦN CHUẨN HOÁ:\\n")[-1].strip()
        return MAP.get(q, q)
    return ""
'''


def main():
    split, scratch = sys.argv[1], sys.argv[2]
    shutil.rmtree(scratch, ignore_errors=True); os.makedirs(scratch)
    sp = os.path.join(scratch, "split"); os.makedirs(sp)
    os.symlink(os.path.abspath(os.path.join(split, "data")), os.path.join(sp, "data"))
    for f in ("calendar.csv", "rulebook.md"): shutil.copy(os.path.join(split, f), sp)
    qs = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(split, "questions.jsonl"), encoding="utf-8")}
    plain = next(q for q in qs.values() if q["business_id"] == "biz_101" and q["id"] not in PARA)
    rows = [dict(qs[i], question=p) for i, (p, _) in PARA.items()] + [plain, dict(plain, id="q_junk", question="xyz qwe asd zxc")]
    with open(os.path.join(sp, "questions.jsonl"), "w", encoding="utf-8") as f:
        for q in rows: f.write(json.dumps(q, ensure_ascii=False) + "\n")
    fake = os.path.join(scratch, "fake.py")
    open(fake, "w", encoding="utf-8").write(FAKE % json.dumps({p: c for p, c in PARA.values()}, ensure_ascii=False))
    work = os.path.join(scratch, "work"); cache = os.path.join(work, "cache"); csplit = os.path.join(work, "canon")
    r = subprocess.run([sys.executable, os.path.join(SRC, "canon_llm.py"), sp, csplit, "--model", "none", "--cache", cache, "--fake-responder", fake],
                       capture_output=True, text=True, timeout=1500)
    log = r.stdout + r.stderr
    assert r.returncode == 0, log[-3000:]
    assert "CANON 4 / 4 restated" in log or "CANON 3 / 3 restated" in log or "restated" in log, log[-2000:]
    rdir = os.path.join(csplit, "restated"); out = os.path.join(work, "rules")
    r = subprocess.run([sys.executable, os.path.join(SRC, "rules_first.py"), rdir, out, "--cache", cache, "--workers", "1"],
                       capture_output=True, text=True, timeout=1500)
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    sub = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(out, "submission.jsonl"), encoding="utf-8")}
    ref = score.load(os.path.join(split, "answers.jsonl"))
    for i in list(PARA) + [plain["id"]]:
        assert i in sub and score.judge(sub[i], ref[i]) == "correct", (i, sub.get(i), ref[i])
    assert "q_junk" not in sub, sub.get("q_junk")
    print("test_canon_flow PASS")


if __name__ == "__main__":
    main()
