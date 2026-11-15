import json, sys, os
sys.path.insert(0, "src"); import score
from collections import Counter, defaultdict
R = score.load("data/raw/public_gt.jsonl")
def load(f):
    out = {}
    for l in open(f):
        r = json.loads(l); out[r["id"]] = r
    return out
def ans(s):
    try: a = eval(s["answer"]) if isinstance(s["answer"], str) else s["answer"]
    except Exception: a = None
    return a
def key(a):
    if not a or a.get("type") == "abstain": return None
    return json.dumps(a, sort_keys=True, ensure_ascii=False)
def vote(samples, theta):
    c = Counter(); rep = {}
    for s in samples:
        a = ans(s); k = key(a)
        if k is None: continue
        c[k] += 1; rep[k] = a
    n = len(samples)
    if not c: return None
    k, v = c.most_common(1)[0]
    return rep[k] if v / n >= theta else None
runs = [load(f) for f in sys.argv[1:]]
for theta in (0.4, 0.5, 0.6):
    for name, sel in [("single0", [0]), ("single1", [1]), ("pool", list(range(len(runs))))]:
        if max(sel) >= len(runs): continue
        tot = 0
        for qid in runs[0]:
            ss = sum((runs[i][qid]["samples"]["samples"] for i in sel if qid in runs[i]), [])
            a = vote(ss, theta)
            if a is None: continue
            try: tot += 1 if score.judge(a, R[qid]) == "correct" else -1
            except Exception: tot -= 1
        print(theta, name, tot / 1000)
