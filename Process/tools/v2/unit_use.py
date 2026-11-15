import json, sys, os, re
sys.path.insert(0, "src"); import score
from collections import Counter, defaultdict
R = score.load("data/raw/public_gt.jsonl")
ind = set(l.strip() for l in open("tests/splits/pub_ind.txt") if l.strip())
for d in sys.argv[1:]:
    names = defaultdict(set)
    for l in open(os.path.join(d, "cards_unit.jsonl")):
        if l.strip(): u = json.loads(l); names[u["bid"]].add(u["name"])
    st = Counter(); qcall = Counter()
    for l in open(os.path.join(d, "solver_logs.jsonl")):
        r = json.loads(l)
        if r["id"] not in ind: continue
        ns = names.get(r["business_id"], set())
        st["q"] += 1; st["q_has_units"] += bool(ns)
        called_any = False
        for s in r["samples"]["samples"]:
            code = " ".join((t.get("code") or "") for t in s["turns"])
            call = any(re.search(r"\b%s\s*\(" % re.escape(n), code) for n in ns)
            try: a = eval(s["answer"]) if isinstance(s["answer"], str) else s["answer"]
            except Exception: a = None
            ok = False
            if a and a.get("type") != "abstain":
                try: ok = score.judge(a, R[r["id"]]) == "correct"
                except Exception: ok = False
            k = "call" if call else ("nocall_u" if ns else "nounits")
            st[k] += 1; st[k + "_ok"] += ok
            called_any |= call
        qcall[called_any] += 1
    print(os.path.basename(os.path.dirname(d)), dict(st), "q_with_call", qcall[True])
    for k in ("call", "nocall_u", "nounits"):
        if st[k]: print("  %s acc %.3f (%d)" % (k, st[k + "_ok"] / st[k], st[k]))
