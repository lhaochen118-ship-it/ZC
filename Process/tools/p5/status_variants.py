#!/usr/bin/env python3
"""16/11: parts done per variant, score of the parts (vs D1 on the same questions) and of the merged submission when complete."""
import sys, json, glob, os
sys.path.insert(0, "/home/user/ZC/src"); import score
R = "/home/user/ZC/Process/runs/p5"
V = {"C4 (mốc, còn industry.py)": ["p5c4h0", "p5c4h1"], "C5 (bỏ industry.py)": ["p5c5h0", "p5c5h1"], "C6 (industry tổng quát)": ["p5c6h0", "p5c6h1"],
     "PG (0.994 tổng quát)": ["pgq0", "pgq1", "pgq2", "pgq3"], "C6-noR": ["c6rq0", "c6rq1", "c6rq2", "c6rq3"], "C6-noT": ["c6tq0", "c6tq1", "c6tq2", "c6tq3"]}
gt = score.load("/home/user/ZC/data/raw/public_gt.jsonl")
d1 = score.load("/home/user/ZC/final_submission/p5_container_d1/submission.jsonl")
S = {"correct": 1, "abstain": 0, "wrong": -1}
def cmp(sub):
    g = {k: gt[k] for k in sub}; r, v = score.score(g, sub); r1, v1 = score.score(g, {k: d1[k] for k in sub})
    d = [S[v[q]] - S[v1[q]] for q in g]; u = sum(x > 0 for x in d); w = sum(x < 0 for x in d)
    return r, r1, u, w, (sum(d) / (u + w) ** .5 if u + w else 0)
for name, parts in V.items():
    subs, done = {}, []
    for p in parts:
        f = glob.glob(f"{R}/{p}/*_out/results/*/*/submission.jsonl")
        st = "".join(open(x).read() for x in glob.glob(f"{R}/{p}/*.log"))
        nogpu = any("no GPU" in open(x).read() for x in glob.glob(f"{R}/{p}/*_out/results/*/*/logs/run.log"))
        if f and "STATUS=DONE" in st and not nogpu:
            subs.update(score.load(f[0])); done.append(p)
    if not subs: print(f"{name}: 0/{len(parts)} parts"); continue
    r, r1, u, w, z = cmp(subs)
    tag = "FULL" if len(done) == len(parts) else f"{len(done)}/{len(parts)} parts"
    print(f"{name} [{tag}, {len(subs)} q]: PA {r['PA']:.3f} ({r['correct']}/{r['wrong']}/{r.get('abstain',0)}) | D1 same q {r1['PA']:.3f} | {r['PA']-r1['PA']:+.3f} (+{u}/-{w}, z {z:.2f})")
    if len(done) == len(parts):
        out = f"/home/user/ZC/Process/runs/p5_results/merged/{parts[0][:-1]}.jsonl"
        with open(out, "w", encoding="utf-8") as fo:
            for q in gt: fo.write(json.dumps(subs.get(q, {"id": q, "type": "abstain"}), ensure_ascii=False) + "\n")
