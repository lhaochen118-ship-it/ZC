#!/usr/bin/env python3
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("cache"); ap.add_argument("canon_dir"); ap.add_argument("out")
    ap.add_argument("--mode", default="compact"); ap.add_argument("--k", type=int, default=3); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--all-shops", action="store_true")
    ap.add_argument("--bank-out", default=os.path.join(common.SRC, "refs", "bank", "examples.jsonl"))
    a = ap.parse_args()
    import retrieval, grammar
    t0 = time.time()
    ex = [r for r in common.jl(os.path.join(common.SRC, "refs", "answer_examples.jsonl")) if a.all_shops or r["business_id"] not in common.TR_HO]
    ans = {r["id"]: r for r in common.jl(os.path.join(a.split, "answers.jsonl"))}
    H = {}
    for r in ex:
        c = common.ctx(a.split, a.cache, a.canon_dir, r["business_id"], a.mode)
        H[r["id"]] = c.hints(r["question"], r["as_of_date"])
    bank_rows = [{"id": r["id"], "business_id": r["business_id"], "question": r["question"], "as_of_date": r["as_of_date"], "code": r["code"],
                  "family": retrieval.family(r["code"]), "skeleton": retrieval.skeleton(r["question"], H[r["id"]])} for r in ex]
    common.write_jl(a.bank_out, bank_rows)
    print(f"pass 1: hints of {len(ex)} examples, bank -> {a.bank_out} ({time.time() - t0:.0f}s)", flush=True)
    bank = retrieval.Bank.load(a.bank_out, include_holdout=a.all_shops)
    rows = []
    for r in (ex[:a.limit] if a.limit else ex):
        c = common.ctx(a.split, a.cache, a.canon_dir, r["business_id"], a.mode)
        exs = bank.retrieve(r["question"], H[r["id"]], k=a.k, exclude_bids={r["business_id"]}, exclude_ids={r["id"]})
        a_ = ans.get(r["id"], {}); a_ = {"type": a_.get("type"), "answer": a_.get("answer")}
        rows.append(common.sample(c, r, r["code"], exs, H[r["id"]], "D1", a_))
    common.write_jl(a.out, rows)
    print(f"D1: {len(rows)} records -> {a.out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
