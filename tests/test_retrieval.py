#!/usr/bin/env python3
"""Offline retrieval quality (docs/08 WP8, QĐ11): for each question, do the k retrieved checked examples (src/retrieval.py,
never the question's own shop or itself) share the function family of its right program? hit@k = at least one does;
share@k = mean fraction. Questions: training questions with a checked program (held-out shops excluded) and the public
questions with the rule reader's program (analysis only: --rules-log of src/rules_first.py on public; paraphrases k=0 with
--para-pub). Prints the numbers; PASS when hit@4 >= --min-hit (a regression floor).
Usage: test_retrieval.py <train_split> <scratch> [--rules-log F] [--para-pub F] [--mode ngram|embed] [--min-hit 0.8]"""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "training", "data"))
TR_HO = ("biz_105", "biz_109", "biz_114", "biz_123", "biz_127", "biz_130")


def evaluate(bank, items, k=4):
    hit = share = n = 0
    for q, fam, bid, qid, hints in items:
        ex = bank.retrieve(q, hints, k=k, exclude_bids={bid}, exclude_ids={qid})
        if not ex: n += 1; continue
        same = [e["family"] == fam for e in ex]
        hit += any(same); share += sum(same) / len(same); n += 1
    return hit / max(1, n), share / max(1, n), n


def main():
    split, S = sys.argv[1], sys.argv[2]
    arg = lambda k, d="": sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    min_hit = float(arg("--min-hit", "0.8"))
    import retrieval
    bank = retrieval.Bank.load(mode=arg("--mode", "ngram"), embed_model=arg("--embed-model"))
    tr = [json.loads(l) for l in open(os.path.join(ROOT, "src", "refs", "bank", "examples.jsonl"), encoding="utf-8")]
    items = [(r["question"], r["family"], r["business_id"], r["id"], None) for r in tr if r["business_id"] not in TR_HO]
    h1, s1, n1 = evaluate(bank, items)
    print(f"train (no held-out shops): hit@4 {h1:.3f} share@4 {s1:.3f} on {n1}")
    pub = []
    if arg("--rules-log"):
        rl = {json.loads(l)["id"]: json.loads(l) for l in open(arg("--rules-log"), encoding="utf-8")}
        qs = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(ROOT, "data", "raw", "public_test", "public_test", "questions.jsonl"), encoding="utf-8")}
        for i, r in rl.items():
            if r.get("code"): pub.append((qs[i]["question"], retrieval.family(r["code"]), qs[i]["business_id"], i, None))
        h2, s2, n2 = evaluate(bank, pub)
        print(f"public (rule programs): hit@4 {h2:.3f} share@4 {s2:.3f} on {n2}")
        if arg("--para-pub"):
            pp = [json.loads(l) for l in open(arg("--para-pub"), encoding="utf-8")]
            k0 = [(p["paraphrase"], retrieval.family(rl[p["id"]]["code"]), p["business_id"], p["id"], None) for p in pp
                  if p.get("ok") and p["k"] == 0 and rl.get(p["id"], {}).get("code")]
            h3, s3, n3 = evaluate(bank, k0)
            print(f"public k=0 paraphrases: hit@4 {h3:.3f} share@4 {s3:.3f} on {n3}")
    ok = h1 >= min_hit
    print("test_retrieval " + ("PASS" if ok else "FAIL")); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
