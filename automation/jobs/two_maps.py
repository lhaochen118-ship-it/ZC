"""two_maps.py <compile_logs dir> <biz> <cand1> <cand2>: run two logged LLM mappings, save their C as pickles, compare."""
import sys, json, glob, pickle, textwrap
sys.path.insert(0, "/home/user/ZC/src"); import canon, agent, dm, pandas as pd
S = "/tmp/claude-0/-home-user-ZC/5e8e7c6d-abe3-5b50-8cec-e8284f659410/scratchpad"
d, b, c1, c2 = sys.argv[1:5]
att = {}
for f in glob.glob(d + "/shard*/compile_logs.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if r["business_id"] == b and r.get("llm_attempts"): att = r["llm_attempts"]
biz = pickle.load(open(f"{S}/cache_pub/{b}.pkl", "rb"))
out = {}
for c in (c1, c2):
    a = att[c.split(":")[1]]; prev = None; code = None
    for x in a:
        if x.get("code"): code = (agent.splice(prev, x["code"]) if (prev and x.get("round", 0) > 0) else None) or x["code"]; prev = code
        if x.get("ok") and c != "llm:0": break
    open(f"{S}/map_{b}_{c.replace(':', '')}.py", "w").write(code)
    r = canon.run_mapping(textwrap.dedent(code), biz, llm=True)
    print(c, "ok", r.get("ok"), (r.get("error") or "")[:200]); out[c] = r.get("C") if r.get("ok") else None
pickle.dump(out, open(f"{S}/maps_{b}.pkl", "wb"))
for c, C in out.items():
    if C is None: continue
    print("==", c, {k: (v.shape, list(v.columns)) for k, v in C.items() if hasattr(v, "shape")})
