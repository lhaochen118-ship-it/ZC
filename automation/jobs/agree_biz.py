"""agree_biz.py <run_dir>...: per business, how many questions the runs answer unanimously / by majority / split."""
import sys, json, collections
sys.path.insert(0, "/home/user/ZC/src")
import vote, agent
Q = {json.loads(l)["id"]: json.loads(l)["business_id"] for l in open("/home/user/ZC/data/raw/public_test/public_test/questions.jsonl")}
runs = [vote.load(r) for r in sys.argv[1:]]
st = collections.defaultdict(collections.Counter)
for q, b in Q.items():
    fin = [L[q]["final"] for L in runs if q in L]
    c = collections.Counter(agent.canon_of(f) for f in fin if f.get("type") != "abstain")
    top = c.most_common(1)[0][1] if c else 0
    k = "all%d" % len(fin) if top == len(fin) else "maj" if top * 2 > len(fin) else "split"
    st[b][k] += 1; st[b]["n"] += 1
    if top == len(fin) and fin and fin[0].get("type") == "unanswerable": st[b]["unans_all"] += 1
T = collections.Counter()
print("biz      n unanimous majority split")
for b in sorted(st):
    s = st[b]; T.update(s); print(b, f"{s['n']:4d} {s['all3']:4d} {s['maj']:4d} {s['split']:4d}")
print("TOTAL", f"{T['n']:4d} {T['all3']:4d} {T['maj']:4d} {T['split']:4d}")
