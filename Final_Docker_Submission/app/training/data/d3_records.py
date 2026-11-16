#!/usr/bin/env python3
import os, sys, re, json, time, random, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common


def degenerate(program):
    import ast
    try: tree = ast.parse(program)
    except SyntaxError: return True
    for n in ast.walk(tree):
        if isinstance(n, (ast.List, ast.Tuple, ast.Set)):
            d = [ast.dump(e) for e in n.elts]
            if len(d) != len(set(d)): return True
    return False


_CODEY = re.compile(r"[A-Za-z_]\w*\(|\bANSWER\b|\bunanswerable\b|\bquestions?\b|[{}]|[a-z]_[a-z]", re.I)
_SCRIPT = re.compile(r"[\u0400-\u04ff\u0e00-\u0e7f\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")


def junk(question):
    q = str(question or "").strip()
    return len(q.split()) < 4 or bool(_CODEY.search(q)) or bool(_SCRIPT.search(q))


def cap_same(rows, n=3):
    seen, out = {}, []
    for r in rows:
        k = " ".join(re.sub(r"[^\w\s]", " ", str(r["question"]).lower()).split())
        if seen.get(k, 0) >= n: continue
        seen[k] = seen.get(k, 0) + 1; out.append(r)
    return out


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
    for p in a.inputs.split(","): rows += [r for r in common.jl(p) if r.get("kept") and r["business_id"] not in common.TR_HO and not degenerate(r["program"])]
    n0 = len(rows); rows = [r for r in rows if not junk(r["question"])]
    rng = random.Random(a.seed); rng.shuffle(rows)
    rows = cap_same(rows); print(f"question filter: {n0} -> {len(rows)} (junk, > 3 copies)", flush=True)
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
