"""scan_maps.py <results_dir>...: rebuild every ok candidate mapping of the runs (LLM attempts + reference transfers),
apply canon.secondary_checks, report new-warning hits and whether the ranking (coverage - penalties) would change."""
import sys, os, json, glob, pickle, textwrap, collections, multiprocessing as mp
sys.path.insert(0, "/home/user/ZC/src"); import canon, agent
S = "/tmp/claude-0/-home-user-ZC/5e8e7c6d-abe3-5b50-8cec-e8284f659410/scratchpad"
REF = "/home/user/ZC/src/refs/compile"
NEW = ("orders.revenue differs from the sum", "orders.order_time has no time", "lines.item_name holds the short", "orders.staff holds given")
def job(t):
    run, b, cand, code = t
    biz = pickle.load(open(f"{S}/cache_pub/{b}.pkl", "rb"))
    try: r = canon.run_mapping(textwrap.dedent(code), biz, llm=cand.startswith("llm"))
    except Exception as e: return run, b, cand, None, str(e)[:100]
    if not r.get("ok"): return run, b, cand, None, (r.get("error") or "")[:100]
    w = canon.secondary_checks(r["C"], biz)
    return run, b, cand, [x[:90] for x in w if x.startswith(NEW)], None
tasks = []; logged = {}
BIZ = set(os.environ.get("BIZ", "").split(",")) - {""}
for d in sys.argv[1:]:
    run = d.split("/")[-3]
    for f in glob.glob(d + "/shard*/compile_logs.jsonl"):
        for l in open(f):
            r = json.loads(l); b = r["business_id"]
            if BIZ and b not in BIZ: continue
            if "chosen" in r: logged[(run, b)] = (r["chosen"], json.loads(r["candidates"]) if isinstance(r["candidates"], str) else r["candidates"])
            for s, atts in (r.get("llm_attempts") or {}).items():
                prev = None; code = None
                for x in atts:
                    if x.get("code"): code = (agent.splice(prev, x["code"]) if (prev and x.get("round", 0) > 0) else None) or x["code"]; prev = code
                    if x.get("ok"): okc = code
                okc = None; prev = None
                for x in atts:
                    if x.get("code"): prev = (agent.splice(prev, x["code"]) if (prev and x.get("round", 0) > 0) else None) or x["code"]
                    if x.get("ok"): okc = prev
                if okc: tasks.append((run, b, f"llm:{s}", okc))
for (run, b), (ch, cands) in logged.items():
    for c in cands:
        if str(c[0]).startswith("ref:"):
            p = os.path.join(REF, c[0][4:] + ".py")
            if os.path.isfile(p): tasks.append((run, b, c[0], open(p).read()))
with mp.Pool(2) as pool: res = pool.map(job, tasks, chunksize=1)
W = {(r, b, c): (w, e) for r, b, c, w, e in res}
for (run, b), (ch, cands) in sorted(logged.items()):
    rows = []
    for c in cands:
        w, e = W.get((run, b, c[0]), (None, "not rebuilt"))
        pen = sum(canon.warning_penalty(x) for x in (w or []))
        rows.append((c[0], c[1], c[2], pen, w, e))
    old = ch
    def sc(x): return x[1] - x[3]
    hit = [x for x in rows if x[4]]
    if hit: print(run, b, "chosen", ch, "| hits:", [(x[0], x[3], [y[:40] for y in x[4]]) for x in hit], "| clean:", [x[0] for x in rows if x[4] == [] ])
print("rebuilt", len(res), "failed", sum(1 for x in res if x[3] is None))
