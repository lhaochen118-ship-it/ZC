"""PA per evaluation set of a run's submission.jsonl (same numbers as Process/scores.csv): pub_full, pub_ho, tr_ho, pub_ho400,
tr_ho400 (+ held-out combined: full sets (pub_ho*1373 + tr_ho*1826)/3199, subsets mean of the two).
Usage: score_sets.py <run_dir_or_submission> [<name>] [--csv]"""
import os, sys, json, glob
ROOT = "/home/user/ZC"; sys.path.insert(0, os.path.join(ROOT, "src"))
import score
p = sys.argv[1]
if os.path.isdir(p): p = sorted(glob.glob(os.path.join(p, "**", "submission.jsonl"), recursive=True))[0]
subs = {json.loads(l)["id"]: json.loads(l) for l in open(p) if l.strip()}
refs = {}
for f in ("data/raw/public_gt.jsonl", "data/raw/train/train/answers.jsonl"): refs.update(score.load(os.path.join(ROOT, f)))
out = {}
for s in ("pub_full", "pub_ho", "tr_ho", "pub_ho400", "tr_ho400"):
    ids = [l.strip() for l in open(os.path.join(ROOT, "tests", "splits", s + ".txt")) if l.strip()]
    have = [i for i in ids if i in subs]
    if not have: continue
    tot = 0; c = {"right": 0, "wrong": 0, "abstain": 0}
    for i in have:
        a = subs[i]; ref = refs[i.split("~")[0]]
        if a.get("type") == "abstain": c["abstain"] += 1; continue
        try: j = score.judge(a, ref)
        except Exception: j = "wrong"
        if j == "correct": c["right"] += 1; tot += 1
        else: c["wrong"] += 1; tot -= 1
    out[s] = (round(tot / len(ids), 4), len(have), len(ids), c)
for s, v in out.items(): print(f"{s}: PA {v[0]} ({v[1]}/{v[2]} answered lines) {v[3]}")
if "pub_ho" in out and "tr_ho" in out and out["pub_ho"][1] == 1373 * 1:
    print("held-out combined (full):", round((out["pub_ho"][0] * 1373 + out["tr_ho"][0] * 1826) / 3199, 4))
if "pub_ho400" in out and "tr_ho400" in out:
    print("held-out subsets mean:", round((out["pub_ho400"][0] + out["tr_ho400"][0]) / 2, 4))
