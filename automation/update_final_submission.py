#!/usr/bin/env python3
"""Keep final_submission/submission.jsonl = the best-scoring public-test submission produced so far (user 10/11: "lưu file
submission.jsonl đạt score cao nhất hiện tại … mỗi khi đạt score cao hơn thì tự động update").

For each run directory given: its submission.jsonl is restricted to the 1000 original public questions (in the split's
order; a question without an answer = abstain, as the container writes it), scored with the organisers' rules against
data/raw/public_gt.jsonl (scoring only), and written to final_submission/ only when its PA is strictly higher than the one
recorded in final_submission/score.json. The answers are those the model's programs produced in that run; nothing is edited.
Usage: update_final_submission.py <run_dir>[,<run_dir>...] [--force]"""
import os, sys, json, time, shutil
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import score, merge_sub

SPLIT = os.path.join(ROOT, "data", "raw", "public_test", "public_test")
OUT = os.path.join(ROOT, "final_submission")


def pa_of(path, refs):
    sub = score.load(path); right = wrong = 0
    for i, ref in refs.items():
        a = sub.get(i)
        if a is None or a.get("type") == "abstain": continue
        try: ok = score.judge(a, ref) == "correct"
        except Exception: ok = False
        right += ok; wrong += not ok
    n = len(refs)
    return {"pa": round((right - wrong) / n, 4), "right": right, "wrong": wrong, "abstain": n - right - wrong, "n": n}


def main():
    runs = [x for x in sys.argv[1].split(",") if x]
    refs = score.load(os.path.join(ROOT, "data", "raw", "public_gt.jsonl"))
    os.makedirs(OUT, exist_ok=True)
    meta_p = os.path.join(OUT, "score.json")
    best = json.load(open(meta_p)) if os.path.isfile(meta_p) and "--force" not in sys.argv else {"pa": -1}
    for run in runs:
        src = run if run.endswith(".jsonl") else os.path.join(run, "submission.jsonl")
        if not os.path.isfile(src): print(f"{run}: no submission.jsonl"); continue
        tmp = os.path.join(OUT, ".candidate.jsonl")
        merge_sub.merge(SPLIT, tmp, [src], quiet=True)                   # the 1000 public questions, split order
        s = pa_of(tmp, refs)
        print(f"{run}: PA {s['pa']:.4f} ({s['right']} / {s['wrong']} / {s['abstain']})  current best {best.get('pa')}")
        if s["pa"] > best.get("pa", -1):
            shutil.move(tmp, os.path.join(OUT, "submission.jsonl"))
            best = dict(s, source=os.path.abspath(src), updated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            json.dump(best, open(meta_p, "w"), indent=1)
            print(f"  -> final_submission/submission.jsonl updated (PA {s['pa']:.4f})")
        elif os.path.isfile(tmp): os.remove(tmp)


if __name__ == "__main__":
    main()
