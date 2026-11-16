#!/usr/bin/env python3
import os, sys, json, random, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("parts"); ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--rft-first-uncovered", default="")
    a = ap.parse_args(); rng = random.Random(a.seed)
    covered = {r["id"].split("~")[0] for r in common.jl(a.rft_first_uncovered)} if a.rft_first_uncovered else set()
    rows, report = [], []
    for part in a.parts.split(","):
        f, n = part.rsplit(":", 1); n = int(n); src = common.jl(f)
        src = [r for r in src if r.get("business_id") not in common.TR_HO]
        if "rft" in os.path.basename(f) and covered:
            unc = [r for r in src if r["id"].split("~")[0] not in covered]
            hard = [r for r in src if r not in unc and "hard" in str(r.get("source", ""))]
            rest = [r for r in src if r not in unc and r not in hard]
            for g in (unc, hard, rest): rng.shuffle(g)
            src = unc + hard + rest
        else: rng.shuffle(src)
        pick = src if n <= 0 or n >= len(src) else src[:n]
        rows += pick; report.append(f"{os.path.basename(f)} {len(pick)}/{len(src)}")
    rng.shuffle(rows)
    common.write_jl(a.out, rows)
    print(f"mix: {len(rows)} records ({'; '.join(report)}) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
