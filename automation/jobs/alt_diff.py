"""alt_diff.py <results_dir>: per business, rule answers on the chosen mapping vs on the runner-up mapping (alt)."""
import sys, json, glob, ast, collections
sys.path.insert(0, "/home/user/ZC/src"); import agent
st = collections.defaultdict(collections.Counter); ex = collections.defaultdict(list)
for f in glob.glob(sys.argv[1] + "/shard*/logs.jsonl"):
    for l in open(f):
        r = json.loads(l); b = r["business_id"]; ru = r.get("samples", {}).get("rule")
        if isinstance(ru, str):
            try: ru = ast.literal_eval(ru)
            except Exception: ru = None
        st[b]["n"] += 1
        if not ru or not ru.get("ok") or not ru.get("alt"): continue
        a0 = agent.canon_of(ru["answer"]); a1 = ru["alt"][0]; a1 = agent.canon_of(a1) if isinstance(a1, dict) else None
        st[b]["with_alt"] += 1
        if a1 is None: st[b]["alt_fail"] += 1
        elif a0 != a1: st[b]["differ"] += 1; ex[b].append((r["question"][:90], str(a0)[:50], str(a1)[:50]))
print("biz n rule_with_alt alt_failed differ")
for b in sorted(st): s = st[b]; print(b, s["n"], s["with_alt"], s["alt_fail"], s["differ"])
if len(sys.argv) > 2:
    for b in sys.argv[2:]:
        for e in ex[b][:12]: print(b, "|", *e, sep=" ")
