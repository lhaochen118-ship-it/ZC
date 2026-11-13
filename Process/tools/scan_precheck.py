"""Re-run src/precheck.py (one rule) on the last program of every sample in run logs: right / wrong of the hit samples,
dev ids only (pub_full, tr_dev_all, d3sel, tr_dev_para), and the questions hit."""
import os, sys, json, collections
ROOT = "/home/user/ZC"; sys.path.insert(0, os.path.join(ROOT, "src"))
import score, api_spec, precheck as pre
rule = sys.argv[2] if len(sys.argv) > 2 else "count_by"
spec = api_spec.build_api_spec()
ids = set()
for s in ("pub_full", "tr_dev_all", "d3sel", "tr_dev_para"):
    ids |= {l.strip() for l in open(os.path.join(ROOT, "tests", "splits", s + ".txt")) if l.strip()}
refs = {}
for p in ("data/raw/public_gt.jsonl", "data/raw/train/train/answers.jsonl", "tests/splits/d3sel.answers.jsonl"):
    refs.update(score.load(os.path.join(ROOT, p)))
hit = collections.Counter(); qs = collections.defaultdict(collections.Counter); n_s = 0
for run in sys.argv[1].split(","):
    f = os.path.join(run, "logs.jsonl")
    if not os.path.isfile(f): continue
    for l in open(f):
        r = json.loads(l)
        if r["id"] not in ids: continue
        S = r.get("samples") or {}; h = S.get("hints") or {}
        ref = refs.get(r["id"].split("~")[0])
        if ref is None: continue
        for s in S.get("samples") or []:
            turns = [t for t in s.get("turns") or [] if t.get("code")]
            if not turns or s.get("answer") is None: continue
            n_s += 1
            iss = pre.precheck(turns[-1]["code"], spec, h, None, turns[-1].get("escape"), r["as_of_date"], r["question"], False, False, True, True)
            if not any(i["rule"] == rule for i in iss): continue
            try: ok = score.judge(s["answer"], ref) == "correct"
            except Exception: ok = False
            hit[ok] += 1; qs[(r["id"], r["question"][:80])][ok] += 1
print("samples scanned", n_s, "hit right", hit[True], "hit wrong", hit[False])
for k, v in qs.items(): print(k, dict(v))
