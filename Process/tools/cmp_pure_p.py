"""compare wrong / abstained public answers of a pure run with the P run: which program part differs (period, entity, family)."""
import json, sys, re, collections
sys.path.insert(0, "/home/user/ZC/src"); import score
ref = score.load("/home/user/ZC/data/raw/public_gt.jsonl")
def logs(p):
    out = {}
    for l in open(p):
        r = json.loads(l)
        if "~" in r["id"]: continue
        best = None
        for s in (r.get("samples") or {}).get("samples") or []:
            if s.get("ok") and s.get("answer") == r.get("final"):
                best = s["turns"][-1].get("code"); break
        out[r["id"]] = (r, best)
    return out
pure, p = logs(sys.argv[1]), logs(sys.argv[2])
PER = re.compile(r"\b(month|quarter|year|week|day|days|range|ytd|mtd|qtd|yesterday|today|last_\w+|this_\w+|between|period|tet\w*)\(([^()]*(\([^()]*\))?[^()]*)\)")
STR = re.compile(r"\"([^\"]{2,})\"|'([^']{2,})'")
FN = re.compile(r"\b([a-z_][a-z0-9_]*)\(")
cat = collections.Counter(); ex = collections.defaultdict(list)
for i, (r, code) in pure.items():
    if i not in ref: continue
    v = score.judge(r.get("final"), ref[i])
    if v == "correct": continue
    pr, pcode = p.get(i, (None, None))
    pv = score.judge(pr.get("final"), ref[i]) if pr else "?"
    if code is None: k = "no_program(abstain/none)"
    elif pv != "correct" or not pcode: k = "P_also_wrong"
    else:
        a, b = code.replace(" ", ""), pcode.replace(" ", "")
        per = set(m.group(0) for m in PER.finditer(a)) != set(m.group(0) for m in PER.finditer(b))
        ent = set(x for t in STR.findall(a) for x in t if x) - {"type", "answer", "number", "percent", "list", "string", "table", "boolean", "date"} != set(x for t in STR.findall(b) for x in t if x) - {"type", "answer", "number", "percent", "list", "string", "table", "boolean", "date"}
        fam = [f for f in FN.findall(a) if f not in ("ANSWER",)][:1] != [f for f in FN.findall(b)][:1]
        k = ("period " if per else "") + ("entity " if ent else "") + ("function " if fam else "")
        k = k.strip() or "other(args/filters/format)"
    cat[(v, k)] += 1; ex[k].append((r["question"], code, pcode))
for (v, k), n in sorted(cat.items(), key=lambda x: -x[1]): print(f"{v:8s} {k:35s} {n}")
import random
for k in sys.argv[3:]:
    print("=====", k)
    for q, a, b in ex.get(k, [])[:7]:
        print("Q:", q); print("  pure:", (a or "")[:230].replace("\n", " ")); print("  P   :", (b or "")[:230].replace("\n", " "))
