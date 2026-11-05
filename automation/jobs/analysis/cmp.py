import json, sys, collections
sys.path.insert(0, "src"); import agent
Q = {json.loads(l)["id"]: json.loads(l)["business_id"] for l in open("data/raw/public_test/public_test/questions.jsonl")}
def ld(p): return {r["id"]: r for r in map(json.loads, open(p))}
a, b = ld(sys.argv[1]), ld(sys.argv[2])
c = collections.Counter(); per = collections.defaultdict(collections.Counter)
for q in Q:
    x, y = a.get(q, {"type": "abstain"}), b.get(q, {"type": "abstain"})
    ab = lambda r: r.get("type") == "abstain"
    k = "same" if (ab(x) and ab(y)) or (not ab(x) and not ab(y) and agent.canon_of(x) == agent.canon_of(y)) else \
        "new_abstain" if ab(y) else "new_answer" if ab(x) else "changed"
    c[k] += 1; per[Q[q]][k] += 1
print(dict(c))
for bz in sorted(per): print(" ", bz, dict(per[bz]))
