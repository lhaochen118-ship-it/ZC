#!/usr/bin/env python3
"""Stage 1.5 of the submitted flow (src/predict_flow.py --canon 1), on CPU with a fake model: two training questions asked
in a new way (rules leave them; restatements taken from a real stage-1.5 run on Kaggle, v33) + one question asked as in
training + one question nobody can read. The restatements must be answered by the rules (right, strict judge), only the
unreadable question may reach stage 2, and the stage-1 answer is kept.
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
    out = os.path.join(scratch, "out")
    r = subprocess.run([sys.executable, os.path.join(SRC, "predict_flow.py"), sp, out, "--work", os.path.join(scratch, "work"),
                        "--runs", "1", "--budget", "900", "--rules-workers", "1", "--fake-responder", fake],
                       capture_output=True, text=True, timeout=1500)
    log = r.stdout + r.stderr
    print("\n".join(l for l in log.splitlines() if l.startswith(("stage", "CANON", "rules:", "after", "end"))))
    assert r.returncode == 0, log[-3000:]
    assert "CANON 3 / 3 restated" in log, "the fake model restates every pending question"
    assert "stage 1.5: rules answered 2 restated questions; 1 left for the model" in log, log[-3000:]
    sub = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(out, "submission.jsonl"), encoding="utf-8")}
    assert [json.loads(l)["id"] for l in open(os.path.join(out, "submission.jsonl"), encoding="utf-8")] == [q["id"] for q in rows]
    ref = score.load(os.path.join(split, "answers.jsonl"))
    for i in list(PARA) + [plain["id"]]:
        assert score.judge(sub[i], ref[i]) == "correct", (i, sub[i], ref[i])
    assert sub["q_junk"]["type"] == "abstain", sub["q_junk"]
    print("test_canon_flow PASS")


if __name__ == "__main__":
    main()
