#!/usr/bin/env python3
"""score judge outputs of a judge-only run (every variant) + paired vs the mode-H baselines H0a / H0a2 (container files)."""
import sys, json, gzip, glob, os
sys.path.insert(0, "/home/user/ZC/src"); import score
gt = score.load("/home/user/ZC/data/raw/public_gt.jsonl"); S = {"correct": 1, "abstain": 0, "wrong": -1}
def load(p):
    op = gzip.open if p.endswith(".gz") else open
    return {j["id"]: j for j in map(json.loads, op(p, "rt")) if j}
base = {k: load(f"/home/user/ZC/Process/runs/v2/{k}/{k}/flow_p5_final_h/submission.jsonl.gz") for k in ("h0a", "h0a2")}
vb = {k: score.score(gt, v)[1] for k, v in base.items()}
for p in sorted(glob.glob(os.path.join(sys.argv[1], "**", "submission.jsonl"), recursive=True)):
    sub = load(p); r, v = score.score(gt, sub); out = [os.path.relpath(p, sys.argv[1]), r["PA"], f"{r.get('correct',0)}/{r.get('wrong',0)}/{r.get('abstain',0)}"]
    for k in vb:
        d = [S[v[q]] - S[vb[k][q]] for q in gt]; u = sum(x > 0 for x in d); w = sum(x < 0 for x in d)
        out.append(f"vs {k}: +{u}/-{w} z {round(sum(d) / (u + w) ** .5, 2) if u + w else 0}")
    print(" | ".join(map(str, out)))
