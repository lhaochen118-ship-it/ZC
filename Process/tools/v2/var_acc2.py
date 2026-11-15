import json, sys, os
sys.path.insert(0, "src"); import score
from collections import Counter, defaultdict
R = score.load("data/raw/public_gt.jsonl")
ind = set(l.strip() for l in open("tests/splits/pub_ind.txt") if l.strip())
for f in sys.argv[1:]:
    st = defaultdict(Counter); qany = defaultdict(Counter)
    for l in open(f):
        r = json.loads(l); g = "ind" if r["id"] in ind else "non"
        ss = r["samples"]["samples"]
        if len(ss) <= 4: continue
        anyok = Counter()
        for i, s in enumerate(ss):
            try: a = eval(s["answer"]) if isinstance(s["answer"], str) else s["answer"]
            except Exception: a = None
            ok = False
            if a and a.get("type") != "abstain":
                try: ok = score.judge(a, R[r["id"]]) == "correct"
                except Exception: pass
            k = (g, s["variant"], "r1" if i < 4 else "r2")
            st[k]["n"] += 1; st[k]["ok"] += ok; anyok[s["variant"]] |= ok
        for v in ("direct", "think"): qany[g][v] += anyok[v]
        qany[g]["n"] += 1
    print(f.split("/")[-3])
    for k in sorted(st): print("  ", k, st[k]["n"], round(st[k]["ok"] / st[k]["n"], 3))
    print("  any-right by variant (hard qs):", {g: dict(c) for g, c in qany.items()})
