#!/usr/bin/env python3
"""Regenerate the cleaned training questions, statistics and the module copy (CPU, a few minutes).

Usage (repo root):
  python3 question_cleaning/make.py [data/raw/train/train] [--cache <ingest cache dir>] [--llm <qnorm_llm questions_clean.jsonl>]

1. rules (src/qnorm.py): vocabulary = rulebook + accented training questions of the OTHER businesses + the business's
   own names (so a question never helps clean itself) + general phrasing (src/qnorm_phrases.txt);
2. optional LLM pass (output of src/qnorm_llm.py on Kaggle): accepted only when its hard check passes (accents only added,
   user's accents / words / numbers unchanged); merged word by word (codes and mid-sentence names untouched);
3. writes question_cleaning/train_questions_clean.jsonl, stats.json, and copies the module (src/qnorm*.py + data files)
   into question_cleaning/module/ (tests/test_qnorm.py checks that the copy is identical to src/)."""
import os, sys, json, shutil, pickle, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import qnorm, qnorm_llm, ingest

MODULE_FILES = ["qnorm.py", "qnorm_llm.py", "qnorm_train_questions.txt", "qnorm_phrases.txt"]
args = [a for a in sys.argv[1:]]
def opt(name):
    if name in args: i = args.index(name); v = args[i + 1]; del args[i:i + 2]; return v
    return ""
cache, llm_file = opt("--cache"), opt("--llm")
split = args[0] if args else os.path.join(ROOT, "data", "raw", "train", "train")
HERE = os.path.join(ROOT, "question_cleaning")


def business(b):
    p = os.path.join(cache, b + ".pkl") if cache else ""
    return pickle.load(open(p, "rb")) if p and os.path.isfile(p) else ingest.load_business(os.path.join(split, "data", b))


llm = {}
if llm_file:
    for l in open(llm_file, encoding="utf-8"):
        r = json.loads(l); llm[r["id"]] = r.get("clean_llm") or ""
qs = [json.loads(l) for l in open(os.path.join(split, "questions.jsonl"), encoding="utf-8")]
by_b = collections.defaultdict(list)
for q in qs: by_b[q["business_id"]].append(q)
out = {}; rules = collections.Counter(); st = collections.Counter()
for b, bq in sorted(by_b.items()):
    nz = qnorm.Normalizer.for_split(split, exclude_business=b, extra_texts=qnorm.Normalizer.entity_texts(business(b)))
    for q in bq:
        r = nz.clean(q["question"], q["as_of_date"]); rec = {"id": q["id"], "business_id": b, "as_of_date": q["as_of_date"],
                                                            "question": q["question"], "clean": r["clean"], "clean_rules": r["clean"]}
        src = "rules"
        if q["id"] in llm:
            t = llm[q["id"]].strip(); ok = qnorm_llm.check(r["clean"], t); rec["clean_llm"] = t; rec["llm_check"] = ok
            if ok:
                m = qnorm_llm.merge(r["clean"], t, q["question"])
                if m != r["clean"]: rec["clean"] = m; src = "rules+llm"; r["changes"].append([r["clean"], m, "llm"])
            st["llm_ok" if ok else "llm_rejected"] += 1
        rec.update(source=src, key=qnorm.strip_accents(rec["clean"]).lower(), period=r["period"], changes=r["changes"])
        out[q["id"]] = rec
        st["changed"] += rec["clean"] != q["question"].strip(); st["period_detected"] += r["period"] is not None; st[src] += 1
        for rule in {c[2] for c in r["changes"]}: rules[rule] += 1
with open(os.path.join(HERE, "train_questions_clean.jsonl"), "w", encoding="utf-8") as f:
    for q in qs: f.write(json.dumps(out[q["id"]], ensure_ascii=False) + "\n")
stats = {"questions": len(qs), **dict(st), "questions_per_rule": dict(rules.most_common())}
json.dump(stats, open(os.path.join(HERE, "stats.json"), "w"), indent=1, ensure_ascii=False)
os.makedirs(os.path.join(HERE, "module"), exist_ok=True)
for f in MODULE_FILES:                                                 # the module itself, next to its output
    if os.path.isfile(os.path.join(ROOT, "src", f)): shutil.copy(os.path.join(ROOT, "src", f), os.path.join(HERE, "module", f))
print(json.dumps(stats, ensure_ascii=False))
