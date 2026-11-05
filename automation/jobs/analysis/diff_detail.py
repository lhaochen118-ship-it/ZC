import json, sys
sys.path.insert(0, "src"); import agent
Q = {json.loads(l)["id"]: json.loads(l) for l in open("data/raw/public_test/public_test/questions.jsonl")}
def ld(p): return {r["id"]: r for r in map(json.loads, open(p))}
a, b = ld(sys.argv[1]), ld(sys.argv[2])
bf = sys.argv[3] if len(sys.argv) > 3 else ""
for q in Q:
    x, y = a.get(q, {"type": "abstain"}), b.get(q, {"type": "abstain"})
    if agent.canon_of(x) == agent.canon_of(y) or (x.get("type")=="abstain" and y.get("type")=="abstain"): continue
    if bf and Q[q]["business_id"] != bf: continue
    sx = json.dumps({k: v for k, v in x.items() if k != "id"}, ensure_ascii=False)[:110]
    sy = json.dumps({k: v for k, v in y.items() if k != "id"}, ensure_ascii=False)[:110]
    print(f"{q} [{Q[q]['business_id']}] {Q[q]['question'][:120]}\n   A: {sx}\n   B: {sy}")
