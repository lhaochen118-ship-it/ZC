#!/usr/bin/env python3
"""D2 (docs/08 §7): paraphrases of the training questions (src/paraphrase_llm.py output, kept 'ok' = numbers / dates / codes /
names preserved), labelled with the original question's checked program. Held-out shops removed. A paraphrase whose period
(qparse.parse_period of the paraphrase vs the original, as-of day of the question) differs is dropped: a changed period
changes the question (entry 26: such paraphrases were almost always another question).
--all-shops (final model only, plan 3 WP3-04): paraphrases of the 6 held-out shops too, examples from the all-shop bank (--bank).
Usage: d2_paraphrase.py <train_split> <ingest_cache> <canon_dir> <para_train.jsonl> <out.jsonl> [--mode compact] [--k 3] [--all-shops --bank F]"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("cache"); ap.add_argument("canon_dir"); ap.add_argument("para"); ap.add_argument("out")
    ap.add_argument("--mode", default="compact"); ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--all-shops", action="store_true"); ap.add_argument("--bank", default=None)
    a = ap.parse_args()
    import retrieval, qparse
    t0 = time.time()
    ex = {r["id"]: r for r in common.jl(os.path.join(common.SRC, "refs", "answer_examples.jsonl")) if a.all_shops or r["business_id"] not in common.TR_HO}
    ans = {r["id"]: r for r in common.jl(os.path.join(a.split, "answers.jsonl"))}
    bank = retrieval.Bank.load(a.bank, include_holdout=a.all_shops)
    rows, drop = [], {"no_program": 0, "period_changed": 0}
    for p in common.jl(a.para):
        if not p.get("ok"): continue
        r = ex.get(p["id"])
        if r is None: drop["no_program"] += 1; continue
        try:
            p0, _ = qparse.parse_period(r["question"], r["as_of_date"]); p1, _ = qparse.parse_period(p["paraphrase"], r["as_of_date"])
        except Exception: p0 = p1 = None
        if (p0 is None) != (p1 is None) or (p0 is not None and (p0.start, p0.end) != (p1.start, p1.end)): drop["period_changed"] += 1; continue
        c = common.ctx(a.split, a.cache, a.canon_dir, r["business_id"], a.mode)
        q = dict(r, id=f"{r['id']}~p{p['k']}", question=p["paraphrase"])
        h = c.hints(q["question"], q["as_of_date"])
        exs = bank.retrieve(q["question"], h, k=a.k, exclude_bids={r["business_id"]}, exclude_ids={r["id"]})
        a_ = ans.get(r["id"], {}); rows.append(common.sample(c, q, r["code"], exs, h, "D2", {"type": a_.get("type"), "answer": a_.get("answer")}))
    common.write_jl(a.out, rows)
    print(f"D2: {len(rows)} records, dropped {drop} -> {a.out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
