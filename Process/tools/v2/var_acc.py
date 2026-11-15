import json, sys, os
sys.path.insert(0, "src"); import score
from collections import Counter, defaultdict
R = score.load("data/raw/public_gt.jsonl")
ind = set(l.strip() for l in open("tests/splits/pub_ind.txt") if l.strip())
for f in sys.argv[1:]:
    st = defaultdict(Counter); nsamp = Counter(); stop = Counter()
    for l in open(f):
        r = json.loads(l); g = "ind" if r["id"] in ind else "non"
        ss = r["samples"]["samples"]; nsamp[len(ss)] += 1; stop[r["samples"].get("stop")] += 1
        for s in ss:
            try: a = eval(s["answer"]) if isinstance(s["answer"], str) else s["answer"]
            except Exception: a = None
            ok = False
            if a and a.get("type") != "abstain":
                try: ok = score.judge(a, R[r["id"]]) == "correct"
                except Exception: pass
            k = (g, s["variant"], float(s["temp"]) > 0)
            st[k]["n"] += 1; st[k]["ok"] += ok
    print(f.split("/")[-3])
    for k in sorted(st): print("  ", k, st[k]["n"], round(st[k]["ok"] / st[k]["n"], 3))
    print("  nsamples", sorted(nsamp.items())[:12], "stop", dict(stop))
