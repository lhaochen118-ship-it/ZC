"""v2 15/11: the judge's abstain decision must survive the final merge (merge_sub first_decides)."""
import os, sys, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import merge_sub
d = tempfile.mkdtemp()
def w(name, rows):
    p = os.path.join(d, name)
    with open(p, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    return p
w("questions.jsonl", [{"id": i} for i in ("a", "b", "c")])
j = w("judge.jsonl", [{"id": "a", "type": "number", "answer": 1}, {"id": "b", "type": "abstain"}])
s = w("solver.jsonl", [{"id": "a", "type": "number", "answer": 9}, {"id": "b", "type": "number", "answer": 2}, {"id": "c", "type": "number", "answer": 3}])
out = os.path.join(d, "out.jsonl")
merge_sub.merge(d, out, [j, s], quiet=True, first_decides=True)
r = {x["id"]: x for x in map(json.loads, open(out))}
assert r["a"]["answer"] == 1 and r["b"]["type"] == "abstain" and r["c"]["answer"] == 3, r
merge_sub.merge(d, out, [j, s], quiet=True)                         # default unchanged: abstain lines skipped
r = {x["id"]: x for x in map(json.loads, open(out))}
assert r["b"]["answer"] == 2, r
print("merge first_decides PASS")
