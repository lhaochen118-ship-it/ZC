#!/usr/bin/env python3
"""A fixed-size, seeded sample of one or more record files (the share of a data source in a branch does not follow the size of
its file: training/sft.py samples the token budget uniformly over the rows it is given).
Usage: mix.py <out.jsonl> <n> <in.jsonl>[,<in.jsonl>...] [--seed 20261108]
       n rows in total, split evenly over the input files (a file with fewer rows gives all of them; the rest is spread)."""
import sys, json, random


def main():
    out, n, files = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(",")
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 20261108
    rows = []
    for f in files:
        rs = [l for l in open(f, encoding="utf-8") if l.strip()]
        random.Random(seed).shuffle(rs); rows.append(rs)
    take = [0] * len(rows); left = n
    while left > 0:                                                      # even split, capped by each file's size
        open_ = [i for i, r in enumerate(rows) if take[i] < len(r)]
        if not open_: break
        share = max(1, left // len(open_))
        for i in open_:
            k = min(share, len(rows[i]) - take[i], left); take[i] += k; left -= k
            if left <= 0: break
    sel = [l for i, r in enumerate(rows) for l in r[:take[i]]]
    random.Random(seed + 1).shuffle(sel)
    with open(out, "w", encoding="utf-8") as f: f.writelines(sel)
    print(f"{out}: {len(sel)} rows " + ", ".join(f"{fn.split('/')[-1]} {t}" for fn, t in zip(files, take)))


if __name__ == "__main__":
    main()
