import os, sys, json, collections
ROOT = "/home/user/ZC"; sys.path.insert(0, os.path.join(ROOT, "src"))
import score, api_spec, precheck as pre, industry
spec = api_spec.build_api_spec(); doc = industry.INDUSTRY_API["sửa xe"]
ids = set()
for s in ("pub_full", "tr_dev_all", "d3sel", "tr_dev_para"):
    ids |= {l.strip() for l in open(os.path.join(ROOT, "tests", "splits", s + ".txt")) if l.strip()}
refs = {}
for p in ("data/raw/public_gt.jsonl", "data/raw/train/train/answers.jsonl", "tests/splits/d3sel.answers.jsonl"):
    refs.update(score.load(os.path.join(ROOT, p)))
REPAIR = set(sys.argv[2].split(","))
hit = collections.Counter(); Q = collections.defaultdict(collections.Counter)
for run in sys.argv[1].split(","):
    f = os.path.join(run, "logs.jsonl")
    if not os.path.isfile(f): continue
    for l in open(f):
        r = json.loads(l)
        if r["id"] not in ids or r["business_id"] not in REPAIR: continue
        ref = refs.get(r["id"].split("~")[0]); S = r.get("samples") or {}
        for s in S.get("samples") or []:
            t = [t for t in s.get("turns") or [] if t.get("code")]
            if not t or s.get("answer") is None: continue
            iss = pre.precheck(t[-1]["code"], spec, S.get("hints") or {}, None, t[-1].get("escape"), r["as_of_date"], None, False, False, False, False, doc)
            if not any(i["rule"] == "ticket_avg" for i in iss): continue
            ok = score.judge(s["answer"], ref) == "correct"; hit[ok] += 1; Q[(r["id"], r["question"][:70])][ok] += 1
print("hit right", hit[True], "wrong", hit[False]); [print(k, dict(v)) for k, v in Q.items()]
