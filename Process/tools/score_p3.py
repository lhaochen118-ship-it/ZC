#!/usr/bin/env python3
"""Plan 3 scores of one system (docs/16 §5): PA on pub_full, pub_ho400c, tr_ho400c, pub_noind, tr_newterm from one or more
submission files (shards of the same system in different kernels), the held-out combined = question-weighted PA over the
held-out sets present (pub_ho400c + tr_ho400c + pub_noind + tr_newterm), appended to Process/scores_p3.csv.
Usage: score_p3.py <name> <submission.jsonl or run dir>[,...] [--no-csv]"""
import os, sys, json, glob, csv, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
import score

SETS = ("pub_full", "pub_ho400c", "tr_ho400c", "pub_noind", "tr_newterm")
HELD = ("pub_ho400c", "tr_ho400c", "pub_noind", "tr_newterm")


def subs_of(paths):
    out = {}
    for p in paths:
        fs = [p] if p.endswith(".jsonl") else sorted(glob.glob(os.path.join(p, "**", "submission.jsonl"), recursive=True))
        for f in fs:
            for l in open(f, encoding="utf-8"):
                if l.strip():
                    r = json.loads(l)
                    if r.get("type") != "abstain" or r["id"] not in out: out[r["id"]] = r
    return out


def main():
    name, paths = sys.argv[1], sys.argv[2].split(",")
    S = subs_of(paths); R = {}
    for f in ("data/raw/public_gt.jsonl", "data/raw/train/train/answers.jsonl"): R.update(score.load(os.path.join(ROOT, f)))
    row = {"run": name, "time": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())}; held_n = held_s = 0
    for s in SETS:
        ids = [l.strip() for l in open(os.path.join(ROOT, "tests", "splits", s + ".txt")) if l.strip()]
        have = [i for i in ids if i in S]
        if len(have) < len(ids) * 0.95: continue                       # set not (fully) in this run
        c = {"correct": 0, "wrong": 0, "abstain": 0}
        for i in ids: c[score.judge(S.get(i), R[i.split("~")[0]])] += 1
        pa = (c["correct"] - c["wrong"]) / len(ids)
        row[s] = round(pa, 4); row[s + "_n"] = len(ids); row[s + "_cwa"] = f"{c['correct']}/{c['wrong']}/{c['abstain']}"
        print(f"{s:11s} PA {pa:.4f}  n {len(ids):4d}  right/wrong/abstain {row[s + '_cwa']}")
        if s in HELD: held_n += len(ids); held_s += pa * len(ids)
    if held_n:
        row["held_combined"] = round(held_s / held_n, 4)
        print(f"held-out combined ({', '.join(s for s in HELD if s in row)}): {row['held_combined']}")
    if "--no-csv" not in sys.argv:
        f = os.path.join(ROOT, "Process", "scores_p3.csv")
        cols = ["run", "time"] + [x for s in SETS for x in (s, s + "_n", s + "_cwa")] + ["held_combined"]
        new = not os.path.isfile(f)
        with open(f, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            if new: w.writeheader()
            w.writerow({k: row.get(k, "") for k in cols})


if __name__ == "__main__":
    main()
