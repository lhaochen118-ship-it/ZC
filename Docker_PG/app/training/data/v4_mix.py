#!/usr/bin/env python3
import os, sys, json, random, argparse
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out")
    ap.add_argument("--ud", required=True); ap.add_argument("--rft", default=""); ap.add_argument("--v3a", required=True); ap.add_argument("--v3b", default="")
    ap.add_argument("--n-rep", type=int, default=800); ap.add_argument("--seed", type=int, default=20261113)
    a = ap.parse_args(); rnd = random.Random(a.seed)
    rows = common.jl(a.ud)
    if a.rft and os.path.isfile(a.rft): rows += common.jl(a.rft)
    if a.v3b and os.path.isfile(a.v3b): rows += [r for r in common.jl(a.v3b) if str(r.get("source", "")).startswith("DR")]
    rep = common.jl(a.v3a); rnd.shuffle(rep); rows += [dict(r, source="REP_" + str(r.get("source", ""))) for r in rep[:a.n_rep]]
    bad = [r for r in rows if r.get("business_id") in common.TR_HO or str(r.get("business_id", "")).startswith("dev_")]
    rows = [r for r in rows if r not in bad]
    rnd.shuffle(rows)
    common.write_jl(a.out, rows)
    c = Counter(str(r.get("source", "?")).split("_")[0] for r in rows)
    ntok = sum(len(m["content"]) for r in rows for m in r["messages"]) / 3.6
    print(f"v4 mix: {len(rows)} records (~{ntok / 1e6:.1f}M tokens), dropped {len(bad)} held-out / public; {dict(c)}", flush=True)


if __name__ == "__main__":
    main()
