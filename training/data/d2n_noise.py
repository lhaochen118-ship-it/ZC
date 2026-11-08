#!/usr/bin/env python3
"""D2N (docs/08 §7, H2): noise augmentation of D1 + D2 questions as owners type them: accent-free, chat noise (abbreviations,
Telex, a word without accents, s/x tr/ch swaps, two words glued: src/perturb.py) and regional words of the data's names
(dm.REGIONAL, the reverse direction: 'cá lóc' written 'cá tràu'). The label stays the original program; the prompt is rebuilt
(hints and examples are read from the noisy question, as at inference).
Usage: d2n_noise.py <train_split> <ingest_cache> <canon_dir> <out.jsonl> <in.jsonl>[,<in2.jsonl>] [--per 1] [--seed 20261108]"""
import os, sys, json, time, random, argparse, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common


def regional(q, rng):
    import dm
    pairs = [(b, a) for a, b in dm.REGIONAL]                           # canonical -> regional
    hits = [(b, a) for b, a in pairs if re.search(r"(?<!\w)" + re.escape(b) + r"(?!\w)", q, re.I)]
    if not hits: return None
    b, a = rng.choice(hits)
    return re.sub(r"(?<!\w)" + re.escape(b) + r"(?!\w)", a, q, count=1, flags=re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("cache"); ap.add_argument("canon_dir"); ap.add_argument("out"); ap.add_argument("inputs")
    ap.add_argument("--per", type=int, default=1); ap.add_argument("--seed", type=int, default=20261108); ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--mode", default="compact")
    a = ap.parse_args()
    import retrieval
    from perturb import perturb_question
    rng = random.Random(a.seed); t0 = time.time()
    bank = retrieval.Bank.load(); rows = []; kinds = {}
    src = []
    for p in a.inputs.split(","): src += common.jl(p)
    for r in src:
        for i in range(a.per):
            kind = rng.choice(["accentfree", "noise", "noise", "regional"])
            q0 = r["question"]
            nq = regional(q0, rng) if kind == "regional" else perturb_question(q0, kind, rng)
            if not nq or nq == q0:
                kind = "noise"; nq = perturb_question(q0, "noise", rng)
            if not nq or nq == q0: continue
            kinds[kind] = kinds.get(kind, 0) + 1
            c = common.ctx(a.split, a.cache, a.canon_dir, r["business_id"], a.mode)
            q = {"id": f"{r['id']}~n{i}", "business_id": r["business_id"], "question": nq, "as_of_date": r["as_of_date"]}
            h = c.hints(nq, r["as_of_date"])
            exs = bank.retrieve(nq, h, k=a.k, exclude_bids={r["business_id"]}, exclude_ids={r["id"].split("~")[0]})
            rows.append(common.sample(c, q, r["program"], exs, h, "D2N", r.get("answer")))
    common.write_jl(a.out, rows)
    print(f"D2N: {len(rows)} records {kinds} -> {a.out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
