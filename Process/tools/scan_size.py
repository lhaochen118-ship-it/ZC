"""Signal scan: hints name a catalogue size (entity kind 'size') but the last program of a sample neither calls size_qty
nor mentions the size value. Also: base unit asked ('mấy bịch') + size_qty without a unit. Dev ids only."""
import os, sys, json, collections, re
ROOT = "/home/user/ZC"; sys.path.insert(0, os.path.join(ROOT, "src"))
import score, dm
ids = set()
for s in ("pub_full", "tr_dev_all", "d3sel", "tr_dev_para"):
    ids |= {l.strip() for l in open(os.path.join(ROOT, "tests", "splits", s + ".txt")) if l.strip()}
refs = {}
for p in ("data/raw/public_gt.jsonl", "data/raw/train/train/answers.jsonl", "tests/splits/d3sel.answers.jsonl"):
    refs.update(score.load(os.path.join(ROOT, p)))
A = collections.Counter(); QA = collections.defaultdict(collections.Counter)
B = collections.Counter(); QB = collections.defaultdict(collections.Counter)
for run in sys.argv[1].split(","):
    f = os.path.join(run, "logs.jsonl")
    if not os.path.isfile(f): continue
    for l in open(f):
        r = json.loads(l)
        if r["id"] not in ids: continue
        S = r.get("samples") or {}; h = S.get("hints") or {}
        ents = h.get("entities") or []
        sizes = [e["data_value"] for e in ents if e.get("kind") == "size" and e.get("data_value")]
        units = [e["data_value"] for e in ents if e.get("kind") == "base_unit" and e.get("data_value")]
        if not sizes: continue
        ref = refs.get(r["id"].split("~")[0])
        if ref is None: continue
        for s in S.get("samples") or []:
            turns = [t for t in s.get("turns") or [] if t.get("code")]
            if not turns or s.get("answer") is None: continue
            code = turns[-1]["code"]
            try: ok = score.judge(s["answer"], ref) == "correct"
            except Exception: ok = False
            ck = dm.key(code)
            used = "size_qty" in code or any(re.sub(r"^(?:size|so|co)\s+", "", dm.key(z)) in ck for z in sizes)
            if not used: A[ok] += 1; QA[(r["id"], r["question"][:70])][ok] += 1
            if units and "size_qty" in code: B[ok] += 1; QB[(r["id"], r["question"][:70])][ok] += 1
print("size named, program ignores it: right", A[True], "wrong", A[False])
for k, v in sorted(QA.items(), key=lambda x: -sum(x[1].values()))[:25]: print("  ", k, dict(v))
print("size + base unit, size_qty: right", B[True], "wrong", B[False])
for k, v in sorted(QB.items(), key=lambda x: -sum(x[1].values()))[:25]: print("  ", k, dict(v))
