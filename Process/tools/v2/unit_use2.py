import json, sys, os, re
sys.path.insert(0, "src"); import score
from collections import Counter, defaultdict
R = score.load("data/raw/public_gt.jsonl")
d = sys.argv[1]
units = defaultdict(list)
for l in open(os.path.join(d, "cards_unit.jsonl")):
    if l.strip(): u = json.loads(l); units[u["bid"]].append(u)
per = defaultdict(Counter)
for l in open(os.path.join(d, "solver_logs.jsonl")):
    r = json.loads(l)
    for s in r["samples"]["samples"]:
        code = " ".join((t.get("code") or "") for t in s["turns"])
        try: a = eval(s["answer"]) if isinstance(s["answer"], str) else s["answer"]
        except Exception: a = None
        ok = False
        if a and a.get("type") != "abstain":
            try: ok = score.judge(a, R[r["id"]]) == "correct"
            except Exception: pass
        for u in units.get(r["business_id"], []):
            if re.search(r"\b%s\s*\(" % re.escape(u["name"]), code):
                per[(u["bid"], u["name"])]["n"] += 1; per[(u["bid"], u["name"])]["ok"] += ok; per[(u["bid"], u["name"])]["q_" + r["id"]] += 1
for (b, n), c in sorted(per.items()):
    qs = len([k for k in c if k.startswith("q_")])
    print(f"{b} {n:35s} calls {c['n']:3d} ok {c['ok']:3d} questions {qs}")
print("units total", sum(len(v) for v in units.values()), "used", len(per))
