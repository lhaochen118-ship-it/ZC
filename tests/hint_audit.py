#!/usr/bin/env python3
"""CPU audit of the hint bundle (src/hints.py through src/shopctx.py, exactly as at inference) against the reference answers,
so a change to the general hint rules is measured without a GPU run (docs/12_PLAN2.md WP-P2-02):
  - answer_type: share of questions with a type hint, and how often it equals the reference type (number / percent / list ...);
  - missing_data: precision / recall of the flag against 'unanswerable' references;
  - rule_terms: share of questions with at least one rulebook row shown;
  - the hints of every question saved (--out), --diff <old.jsonl> lists the questions whose hints changed.
Sets: 'public' (the original public test, references from data/raw/public_gt.jsonl: scoring only) and 'train' (training shops
minus the 6 held-out ones). Held-out questions are never read.
Usage: hint_audit.py public|train [--cache DIR] [--canon DIR] [--out hints.jsonl] [--diff old.jsonl] [--limit N] [--show-type]"""
import os, sys, json, time
from collections import Counter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "training", "data"))
os.environ.setdefault("ZC_ADJ_ROWS", "move")
import score, common

SPLITS = {"public": (os.path.join(ROOT, "data", "raw", "public_test", "public_test"), os.path.join(ROOT, "data", "raw", "public_gt.jsonl")),
          "train": (os.path.join(ROOT, "data", "raw", "train", "train"), os.path.join(ROOT, "data", "raw", "train", "train", "answers.jsonl"))}


def main():
    which = sys.argv[1]
    arg = lambda k, d="": sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    scratch = os.environ.get("ZC_SCRATCH", "/tmp")
    cache = arg("--cache", os.path.join(scratch, "cache_pub" if which == "public" else "cache_train"))
    canon = arg("--canon", os.path.join(scratch, "canon"))
    split, refp = SPLITS[which]
    R = score.load(refp)
    qs = common.jl(os.path.join(split, "questions.jsonl"))
    if which == "train": qs = [q for q in qs if q["business_id"] not in common.TR_HO]
    if arg("--limit"): qs = qs[:int(arg("--limit"))]
    out = []; t0 = time.time()
    c = Counter(); conf = Counter()
    for q in qs:
        ctx = common.ctx(split, cache, canon, q["business_id"], "full")
        try: h = ctx.hints(q["question"], q["as_of_date"])
        except Exception as e: h = {"error": f"{type(e).__name__}: {e}"}; c["hint_error"] += 1
        ref = R.get(q["id"]) or {}
        rt = ref.get("type"); ht = h.get("answer_type")
        c["n"] += 1
        if ht: c["typed"] += 1; c["type_ok"] += int(ht == rt or (rt == "unanswerable")); conf[(ht, rt)] += 1
        if rt == "unanswerable": c["unans"] += 1
        if h.get("missing_data"):
            c["flag"] += 1; c["flag_ok"] += int(rt == "unanswerable")
        if rt == "unanswerable" and h.get("missing_data"): c["unans_flagged"] += 1
        if h.get("rule_terms"): c["with_terms"] += 1
        if h.get("clean"): c["cleaned"] += 1
        out.append({"id": q["id"], "business_id": q["business_id"], "question": q["question"], "ref_type": rt,
                    "hints": {k: v for k, v in h.items() if k != "rule_terms"}, "terms": [t["term"] for t in h.get("rule_terms") or []]})
    n = max(1, c["n"])
    print(f"{which}: {c['n']} questions in {time.time() - t0:.0f}s")
    print(f"  answer_type: hinted {c['typed'] / n:.3f}; equals the reference type (unanswerable counted ok) {c['type_ok'] / max(1, c['typed']):.4f}")
    wrong = Counter({k: v for k, v in conf.items() if k[0] != k[1] and k[1] != "unanswerable"})
    print(f"  type confusions (hint, reference): {wrong.most_common(10)}")
    print(f"  missing_data: flagged {c['flag']}, precision {c['flag_ok'] / max(1, c['flag']):.3f}, recall {c['unans_flagged'] / max(1, c['unans']):.3f} ({c['unans']} unanswerable)")
    print(f"  rule terms shown: {c['with_terms'] / n:.3f}; cleaned: {c['cleaned'] / n:.3f}; errors {c['hint_error']}")
    if arg("--out"):
        with open(arg("--out"), "w", encoding="utf-8") as f:
            for r in out: f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    if arg("--diff") and os.path.isfile(arg("--diff")):
        old = {r["id"]: r for r in map(json.loads, open(arg("--diff"), encoding="utf-8"))}
        ch = [r for r in out if r["id"] in old and (old[r["id"]]["hints"] != r["hints"] or old[r["id"]]["terms"] != r["terms"])]
        print(f"  changed vs {arg('--diff')}: {len(ch)}")
        for r in ch[:int(arg("--show", "40"))]:
            o = old[r["id"]]
            d = {k: (o["hints"].get(k), r["hints"].get(k)) for k in set(o["hints"]) | set(r["hints"]) if o["hints"].get(k) != r["hints"].get(k)}
            if o["terms"] != r["terms"]: d["terms"] = (o["terms"], r["terms"])
            print(f"   {r['id']} [{r['ref_type']}] {r['question'][:90]}\n      {json.dumps(d, ensure_ascii=False, default=str)[:400]}")
    if "--show-type" in sys.argv:
        for r in out:
            ht = r["hints"].get("answer_type")
            if ht and ht != r["ref_type"] and r["ref_type"] != "unanswerable": print(f"   {r['id']} hint {ht} ref {r['ref_type']}: {r['question'][:110]}")


if __name__ == "__main__":
    main()
