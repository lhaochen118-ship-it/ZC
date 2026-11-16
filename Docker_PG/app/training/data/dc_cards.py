#!/usr/bin/env python3
import os, sys, json, time, random, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("cache"); ap.add_argument("canon_dir"); ap.add_argument("inputs"); ap.add_argument("out")
    ap.add_argument("--cards", required=True); ap.add_argument("--distract", type=float, default=0.3); ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--mode", default="full"); ap.add_argument("--k", type=int, default=3); ap.add_argument("--only-matched", action="store_true")
    ap.add_argument("--max-df", type=float, default=0.02)
    a = ap.parse_args()
    import retrieval, lrl
    rng = random.Random(a.seed); t0 = time.time()
    lib = lrl.Library.load(a.cards); bank = retrieval.Bank.load()
    tq = [json.loads(l)["question"] for l in open(os.path.join(a.split, "questions.jsonl"), encoding="utf-8") if l.strip()]
    lib.prune(tq, a.max_df); print(f"cards: {len(lib)} after pruning generic triggers", flush=True)
    pool = [c for c in lib.cards if not (c.get("trigger") or {}).get("bid")]
    rows, n_m, n_d = [], 0, 0
    for f in a.inputs.split(","):
        for r in common.jl(f):
            if r.get("business_id") in common.TR_HO: continue
            c = common.ctx(a.split, a.cache, a.canon_dir, r["business_id"], a.mode)
            cards = [x for x in lib.match(r["question"], bid=r["business_id"], industry=c.ind[2] if c.ind else None, rulebook=lrl.rb_key(c.rulebook))
                     if (x.get("case") or {}).get("business_id") != r["business_id"]]
            if not cards and a.only_matched: continue
            n_m += bool(cards)
            if pool and rng.random() < a.distract:
                d = rng.choice(pool)
                if d not in cards: cards = cards + [d]; n_d += 1
            base = r["id"].split("~")[0]
            q = {"id": r["id"], "business_id": r["business_id"], "question": r["question"], "as_of_date": r["as_of_date"]}
            exs = bank.retrieve(r["question"], {}, k=a.k, exclude_bids={r["business_id"]}, exclude_ids={base})
            s = common.sample(c, q, r["program"], exs, None, (r.get("source") or "D?") + "C", r.get("answer"), rules=lrl.render(cards))
            s["messages"][-1] = r["messages"][-1]
            s["rules"] = [x.get("id") for x in cards]; rows.append(s)
    common.write_jl(a.out, rows)
    print(f"DC: {len(rows)} records ({n_m} with matched cards, {n_d} with a distractor) from {a.inputs} -> {a.out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
