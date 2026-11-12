#!/usr/bin/env python3
"""Pure-model training records (user 12/11: the model writes the program from the question, the shop context and retrieved
checked examples only — no rule-read hints). Re-renders existing SFT records (D1 / D2 / D2b ...: same question, same checked
program, same label) in the prompt of the pure flow (src/hybrid.py --pure 1): no # HINTS block, examples retrieved by the question
text alone (retrieval.Bank.retrieve(question, {}): never the same shop, never the question itself or its paraphrases).
Usage: pure_records.py <train_split> <ingest_cache> <canon_dir> <in.jsonl>[,<in.jsonl>...] <out_dir> [--mode full] [--k 3]
       writes <out_dir>/<name of each input>p.jsonl (d1_full.jsonl -> d1_fullp.jsonl)"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("cache"); ap.add_argument("canon_dir"); ap.add_argument("inputs"); ap.add_argument("out_dir")
    ap.add_argument("--mode", default="full"); ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    import retrieval
    bank = retrieval.Bank.load()
    os.makedirs(a.out_dir, exist_ok=True)
    for f in a.inputs.split(","):
        t0 = time.time(); rows = []
        for r in common.jl(f):
            q = {"id": r["id"], "business_id": r["business_id"], "question": r["question"], "as_of_date": r["as_of_date"]}
            c = common.ctx(a.split, a.cache, a.canon_dir, r["business_id"], a.mode)
            exs = bank.retrieve(q["question"], {}, k=a.k, exclude_bids={r["business_id"]}, exclude_ids={r["id"].split("~")[0]})
            s = common.sample(c, q, r["program"], exs, None, r.get("source", "D?") + "p", r.get("answer"))
            rows.append(s)
        out = os.path.join(a.out_dir, os.path.basename(f).replace(".jsonl", "p.jsonl"))
        common.write_jl(out, rows)
        print(f"{f}: {len(rows)} pure records -> {out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
