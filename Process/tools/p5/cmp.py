#!/usr/bin/env python3
"""score a run (one or several submission.jsonl[.gz] files = shop halves, concatenated) + paired comparison with references.
Usage: cmp.py <out.jsonl> <sub>[,<sub>...] [--ref name=path ...]"""
import sys, json, gzip
sys.path.insert(0, "/home/user/ZC/src"); import score
S = {"correct": 1, "abstain": 0, "wrong": -1}
def load(p):
    op = gzip.open if p.endswith(".gz") else open
    return {j["id"]: j for j in map(json.loads, op(p, "rt")) if j}
gt = score.load("/home/user/ZC/data/raw/public_gt.jsonl")
sub = {}
for p in sys.argv[2].split(","): sub.update(load(p))
with open(sys.argv[1], "w", encoding="utf-8") as f:
    for q in gt:
        f.write(json.dumps(sub.get(q, {"id": q, "type": "abstain"}), ensure_ascii=False) + "\n")
r, v = score.score(gt, score.load(sys.argv[1]))
print(f"{sys.argv[1]}: n_sub {len(sub)} PA {r['PA']} ({r.get('correct',0)}/{r.get('wrong',0)}/{r.get('abstain',0)})")
for a in [x for x in sys.argv[3:] if "=" in x]:
    name, path = a.split("=", 1); rr, rv = score.score(gt, load(path))
    d = [S[v[q]] - S[rv[q]] for q in gt]; u = sum(x > 0 for x in d); w = sum(x < 0 for x in d)
    print(f"   vs {name} {rr['PA']}: {r['PA'] - rr['PA']:+.3f}  (+{u} / -{w}, z {round(sum(d) / (u + w) ** .5, 2) if u + w else 0})")
