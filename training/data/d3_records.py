#!/usr/bin/env python3
"""D3 step 3: SFT records from the teacher questions kept by the round-trip filter (training/d3_job.py output: d3_<tag>.jsonl,
kept = True). Each kept question gets the inference prompt of its shop (src/shopctx.py: hints read from the question,
examples retrieved from the checked bank, never the same shop) and the program as the label. Parallel by shop.
Usage: d3_records.py <train_split> <ingest_cache> <canon_dir> <out.jsonl> <d3_tag.jsonl>[,...] [--workers 4] [--max N] [--seed 20261108]"""
import os, sys, json, time, random, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common


def shop_records(args):
    split, cache, canon, bid, rows = args
    import retrieval
    bank = retrieval.Bank.load(); out = []
    c = common.ctx(split, cache, canon, bid, "compact")
    for r in rows:
        q = {"id": r["id"], "business_id": bid, "question": r["question"], "as_of_date": r["as_of_date"]}
        h = c.hints(q["question"], q["as_of_date"])
        exs = bank.retrieve(q["question"], h, k=3, exclude_bids={bid}, exclude_ids={r["id"]})
        out.append(common.sample(c, q, r["program"], exs, h, "D3:" + r.get("teacher", "?"), r.get("answer")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("cache"); ap.add_argument("canon_dir"); ap.add_argument("out"); ap.add_argument("inputs")
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--max", type=int, default=0); ap.add_argument("--seed", type=int, default=20261108)
    a = ap.parse_args()
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time(); rows = []
    for p in a.inputs.split(","): rows += [r for r in common.jl(p) if r.get("kept") and r["business_id"] not in common.TR_HO]
    rng = random.Random(a.seed); rng.shuffle(rows)
    if a.max: rows = rows[:a.max]
    by = {}
    for r in rows: by.setdefault(r["business_id"], []).append(r)
    recs = []
    with ProcessPoolExecutor(a.workers) as ex:
        for out in ex.map(shop_records, [(a.split, a.cache, a.canon_dir, b, rs) for b, rs in sorted(by.items(), key=lambda x: -len(x[1]))]):
            recs += out
    rng.shuffle(recs)
    common.write_jl(a.out, recs)
    print(f"D3 records: {len(recs)} from {len(by)} shops -> {a.out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
