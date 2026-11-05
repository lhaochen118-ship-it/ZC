#!/usr/bin/env python3
"""Rule-first stage + merge (docker/predict.sh stage 1), on CPU: a small split made of 3 training businesses (60 questions
+ 2 questions no rule reads) goes through src/rules_first.py; every question must end in submission.jsonl or
pending.jsonl (never both), the answered ones must match answers.jsonl (strict judge), the unreadable ones must be
pending; src/merge_sub.py must keep the rule answer first, take a second source only for the others, write abstain
for the rest and keep the split's order.
Usage: test_rules_first.py <train_split_dir> <scratch_dir>"""
import os, sys, json, shutil, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(HERE, "..", "src")
sys.path.insert(0, SRC)
import score


def main():
    split, scratch = sys.argv[1], sys.argv[2]
    shutil.rmtree(scratch, ignore_errors=True); os.makedirs(scratch)
    sp = os.path.join(scratch, "split"); os.makedirs(sp)
    os.symlink(os.path.abspath(os.path.join(split, "data")), os.path.join(sp, "data"))
    for f in ("calendar.csv", "rulebook.md"): shutil.copy(os.path.join(split, f), sp)
    qs = [json.loads(l) for l in open(os.path.join(split, "questions.jsonl"), encoding="utf-8")]
    bids = sorted({q["business_id"] for q in qs})[:3]
    sel = [q for q in qs if q["business_id"] in bids]
    sel = [q for b in bids for q in [x for x in sel if x["business_id"] == b][:20]]
    junk = [dict(sel[0], id="q_junk_1", question="Bạn có thể hát một bài không?"),
            dict(sel[1], id="q_junk_2", question="xyz qwe asd zxc")]
    with open(os.path.join(sp, "questions.jsonl"), "w", encoding="utf-8") as f:
        for q in sel + junk: f.write(json.dumps(q, ensure_ascii=False) + "\n")
    out = os.path.join(scratch, "out")
    r = subprocess.run([sys.executable, os.path.join(SRC, "rules_first.py"), sp, out, "--cache", os.path.join(scratch, "cache"),
                        "--workers", "3", "--log-dir", os.path.join(out, "logs")], capture_output=True, text=True, timeout=1500)
    print(r.stdout[-1500:], r.stderr[-1500:])
    assert r.returncode == 0, "rules_first failed"
    sub = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(out, "submission.jsonl"), encoding="utf-8")}
    pend = [json.loads(l)["id"] for l in open(os.path.join(out, "pending.jsonl"), encoding="utf-8")]
    ids = [q["id"] for q in sel + junk]
    assert set(sub) | set(pend) == set(ids) and not (set(sub) & set(pend)), "every question answered xor pending"
    assert {"q_junk_1", "q_junk_2"} <= set(pend), f"unreadable questions must be pending: {pend}"
    ref = score.load(os.path.join(split, "answers.jsonl"))
    verdicts = [score.judge(sub[q], ref[q]) for q in sub]
    ok = verdicts.count("correct"); bad = verdicts.count("wrong")
    print(f"rules: {len(sub)} answered, {ok} correct, {bad} wrong, {len(pend)} pending")
    assert len(sub) >= 55 and bad == 0, "the rule stage must answer the training questions right"
    assert len(os.listdir(os.path.join(out, "logs"))) == len(sub), "one log per answered question"
    # merge: rules first, the second source for the others only, abstain for the rest, split order
    second = os.path.join(scratch, "second.jsonl")
    with open(second, "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "q_junk_1", "type": "unanswerable", "answer": None}) + "\n")
        some = next(iter(sub)); f.write(json.dumps({"id": some, "type": "number", "answer": -12345}) + "\n")
        f.write('{"id": "q_junk_2", "type": "num')                         # a line being written
    final = os.path.join(scratch, "final.jsonl")
    r = subprocess.run([sys.executable, os.path.join(SRC, "merge_sub.py"), sp, final, os.path.join(out, "submission.jsonl"), second],
                       capture_output=True, text=True, timeout=120)
    print(r.stdout.strip())
    assert r.returncode == 0
    rows = [json.loads(l) for l in open(final, encoding="utf-8")]
    assert [x["id"] for x in rows] == ids, "split order"
    by = {x["id"]: x for x in rows}
    assert by[some] == sub[some], "a rule answer is never replaced"
    assert by["q_junk_1"]["type"] == "unanswerable" and by["q_junk_2"]["type"] == "abstain"
    # a term the split's rulebook adds (not in the train / public rulebooks): its questions go to the model
    sp2 = os.path.join(scratch, "split_newterm"); os.makedirs(sp2)
    os.symlink(os.path.abspath(os.path.join(split, "data")), os.path.join(sp2, "data"))
    shutil.copy(os.path.join(split, "calendar.csv"), sp2)
    rb = open(os.path.join(split, "rulebook.md"), encoding="utf-8").read() + "\n| Khách chịu chơi nhất | khách có nhiều đơn trên 1 triệu nhất |\n"
    open(os.path.join(sp2, "rulebook.md"), "w", encoding="utf-8").write(rb)
    q0 = sel[0]
    with open(os.path.join(sp2, "questions.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps(dict(q0, id="q_new_term", question="Tháng này khách chịu chơi nhất là ai?"), ensure_ascii=False) + "\n")
        f.write(json.dumps(q0, ensure_ascii=False) + "\n")
    out2 = os.path.join(scratch, "out_newterm")
    r = subprocess.run([sys.executable, os.path.join(SRC, "rules_first.py"), sp2, out2, "--cache", os.path.join(scratch, "cache"), "--workers", "1"],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-1500:]
    pend2 = [json.loads(l)["id"] for l in open(os.path.join(out2, "pending.jsonl"), encoding="utf-8")]
    assert pend2 == ["q_new_term"], f"a new rulebook term must send its question (only) to the model: {pend2}"
    print("test_rules_first PASS")


if __name__ == "__main__":
    main()
