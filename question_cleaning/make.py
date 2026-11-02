#!/usr/bin/env python3
"""Regenerate the cleaned training questions + statistics (CPU, a few minutes: each business is ingested once).
Usage (repo root): python3 question_cleaning/make.py [data/raw/train/train] [--cache <ingest cache dir>]
Writes question_cleaning/train_questions_clean.jsonl and question_cleaning/stats.json."""
import os, sys, json, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import qnorm, ingest, pickle

args = [a for a in sys.argv[1:] if not a.startswith("--")]
cache = sys.argv[sys.argv.index("--cache") + 1] if "--cache" in sys.argv else ""
if cache in args: args.remove(cache)
split = args[0] if args else os.path.join(ROOT, "data", "raw", "train", "train")


def business(b):
    p = os.path.join(cache, b + ".pkl") if cache else ""
    return pickle.load(open(p, "rb")) if p and os.path.isfile(p) else ingest.load_business(os.path.join(split, "data", b))

qs = [json.loads(l) for l in open(os.path.join(split, "questions.jsonl"), encoding="utf-8")]
by_b = collections.defaultdict(list)
for q in qs: by_b[q["business_id"]].append(q)
out = {}; rules = collections.Counter(); n_changed = n_period = 0
for b, bq in sorted(by_b.items()):
    nz = qnorm.Normalizer.for_split(split, exclude_business=b,           # vocabulary: never the business's own questions,
                                    extra_texts=qnorm.Normalizer.entity_texts(business(b)))   # plus the names in its data
    for q in bq:
        r = nz.clean(q["question"], q["as_of_date"])
        out[q["id"]] = {"id": q["id"], "business_id": b, "as_of_date": q["as_of_date"], "question": q["question"], **r}
        n_changed += r["clean"] != q["question"].strip(); n_period += r["period"] is not None
        for rule in {c[2] for c in r["changes"]}: rules[rule] += 1
with open(os.path.join(ROOT, "question_cleaning", "train_questions_clean.jsonl"), "w", encoding="utf-8") as f:
    for q in qs: f.write(json.dumps(out[q["id"]], ensure_ascii=False) + "\n")
stats = {"questions": len(qs), "changed": n_changed, "period_detected": n_period, "questions_per_rule": dict(rules.most_common())}
json.dump(stats, open(os.path.join(ROOT, "question_cleaning", "stats.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(stats, ensure_ascii=False))
