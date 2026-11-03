#!/usr/bin/env python3
"""map_penalties.py <ingest_cache_dir> <results_dir>...: rebuild each business's CHOSEN mapping of a finished run with the current src/ and write
<results_dir>/map_penalties.json {business: sum of canon.warning_penalty} (vote.py's split tie-break). LLM samples and
reference programs; other kinds (adapt / fmt) are left out (no tie-break for them). Runs from before the pipeline logged
chosen_penalty need this; newer runs carry it in compile_logs.jsonl."""
import sys, os, json, glob, pickle, textwrap, multiprocessing as mp
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); import canon, agent
CACHE = sys.argv[1]; REF = os.path.join(ROOT, "src", "refs", "compile")   # CACHE: src/ingest.py output of the split
def job(t):
    d, b, ch, code = t
    biz = pickle.load(open(os.path.join(CACHE, b + ".pkl"), "rb"))
    try: r = canon.run_mapping(textwrap.dedent(code), biz, llm=ch.startswith("llm"))
    except Exception as e: return d, b, ch, None
    if not r.get("ok"): return d, b, ch, None
    return d, b, ch, round(sum(canon.warning_penalty(w) for w in canon.secondary_checks(r["C"], biz) + (r.get("warnings") or [])
                               ), 1)
tasks = []
for d in sys.argv[2:]:
    att, chosen = {}, {}
    for f in glob.glob(d + "/shard*/compile_logs.jsonl"):
        for l in open(f):
            r = json.loads(l); b = r["business_id"]
            if r.get("llm_attempts"): att[b] = r["llm_attempts"]
            if "chosen" in r: chosen[b] = r["chosen"]
    for b, ch in chosen.items():
        code = None
        if ch.startswith("llm:"):
            prev = None
            for x in att.get(b, {}).get(ch[4:], []):
                if x.get("code"): prev = (agent.splice(prev, x["code"]) if (prev and x.get("round", 0) > 0) else None) or x["code"]
                if x.get("ok"): code = prev
        elif ch.startswith("ref:") and os.path.isfile(os.path.join(REF, ch[4:] + ".py")): code = open(os.path.join(REF, ch[4:] + ".py")).read()
        if code: tasks.append((d, b, ch, code))
with mp.Pool(3) as pool: res = pool.map(job, tasks, chunksize=1)
out = {}
for d, b, ch, p in res:
    if p is not None: out.setdefault(d, {})[b] = p
for d, m in out.items():
    json.dump(m, open(os.path.join(d, "map_penalties.json"), "w"), indent=0); print(d.split("/")[-3], m)
