#!/usr/bin/env python3
"""automap canonical tables of every shop of a split, pickled once into <canon_dir>/<shop>.pkl ({"C", "notes", "data_end"}),
for the CPU tests and the training-data factory (training/data/common.canon reads the same files).
Usage: build_canon.py <split> <ingest_cache> <canon_dir> [workers]"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "training", "data"))
from concurrent.futures import ProcessPoolExecutor


def one(args):
    import common
    split, cache, canon, b = args
    common.canon(split, cache, canon, b); return b


if __name__ == "__main__":
    split, cache, canon = sys.argv[1:4]; w = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    bids = sorted(d for d in os.listdir(os.path.join(split, "data")) if os.path.isdir(os.path.join(split, "data", d)))
    with ProcessPoolExecutor(w) as ex: done = list(ex.map(one, [(split, cache, canon, b) for b in bids]))
    print(f"canon: {len(done)} shops -> {canon}")
