"""Signal scan: samples whose last program calls a given function (regex), right / wrong per question, dev ids only."""
import os, sys, json, collections, re
ROOT = "/home/user/ZC"; sys.path.insert(0, os.path.join(ROOT, "src"))
import score
rx = re.compile(sys.argv[2])
ids = set()
for s in ("pub_full", "tr_dev_all", "d3sel", "tr_dev_para"):
    ids |= {l.strip() for l in open(os.path.join(ROOT, "tests", "splits", s + ".txt")) if l.strip()}
refs = {}
for p in ("data/raw/public_gt.jsonl", "data/raw/train/train/answers.jsonl", "tests/splits/d3sel.answers.jsonl"):
    refs.update(score.load(os.path.join(ROOT, p)))
byb = collections.defaultdict(collections.Counter); Q = collections.defaultdict(collections.Counter)
for run in sys.argv[1].split(","):
    f = os.path.join(run, "logs.jsonl")
    if not os.path.isfile(f): continue
    for l in open(f):
        r = json.loads(l)
        if r["id"] not in ids: continue
        ref = refs.get(r["id"].split("~")[0])
        if ref is None: continue
        for s in (r.get("samples") or {}).get("samples") or []:
            turns = [t for t in s.get("turns") or [] if t.get("code")]
            if not turns or s.get("answer") is None or not rx.search(turns[-1]["code"]): continue
            try: ok = score.judge(s["answer"], ref) == "correct"
            except Exception: ok = False
            byb[r["business_id"]][ok] += 1; Q[(r["business_id"], r["id"], r["question"][:70])][ok] += 1
for b, c in sorted(byb.items()): print(b, dict(c))
if "--q" in sys.argv:
    for k, v in sorted(Q.items()): print("  ", k, dict(v))
