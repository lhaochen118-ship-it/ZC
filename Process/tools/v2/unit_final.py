import json, sys, os, re
sys.path.insert(0, "src"); import score
from collections import Counter, defaultdict
R = score.load("data/raw/public_gt.jsonl")
for d in sys.argv[1:]:
    names = defaultdict(set)
    for l in open(os.path.join(d, "cards_unit.jsonl")):
        if l.strip(): u = json.loads(l); names[u["bid"]].add(u["name"])
    sub = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(d, "submission.jsonl"))}
    c = Counter()
    for l in open(os.path.join(d, "solver_logs.jsonl")):
        r = json.loads(l); ns = names.get(r["business_id"], set())
        ss = r["samples"]["samples"]
        calls = sum(any(re.search(r"\b%s\s*\(" % re.escape(n), " ".join((t.get("code") or "") for t in s["turns"])) for n in ns) for s in ss)
        if not calls: continue
        frac = calls / len(ss)
        a = sub.get(r["id"]); ref = R[r["id"]]
        res = "abstain" if (not a or a.get("type") == "abstain") else ("right" if score.judge(a, ref) == "correct" else "wrong")
        c[("maj" if frac >= 0.5 else "min", res)] += 1
    print(os.path.basename(d), sorted(c.items()))
