#!/usr/bin/env python3
"""Merge submissions in priority order: for every question of the split, the first file (in the order given) with a
non-abstain answer for it wins; questions without one are written as abstain. Written atomically (tmp + rename), in the
split's question order, so it can run repeatedly while the model is still answering.

Usage: merge_sub.py <split_dir> <out.jsonl> <first.jsonl> [<second.jsonl> ...]   (missing / partial files are fine)"""
import os, sys, json


def read(path):
    out = {}
    if not os.path.isfile(path): return out
    for l in open(path, encoding="utf-8"):
        try: r = json.loads(l)
        except ValueError: continue                                    # a line being written
        if isinstance(r, dict) and r.get("id") and r.get("type") and r.get("type") != "abstain": out[r["id"]] = r
    return out


def merge(split, out, srcs, quiet=False):
    """write out from srcs (priority order); -> [answers taken from each source]."""
    ids = [json.loads(l)["id"] for l in open(os.path.join(split, "questions.jsonl"), encoding="utf-8") if l.strip()]
    subs = [read(p) for p in srcs]
    n = [0] * len(srcs); tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for q in ids:
            for i, s in enumerate(subs):
                if q in s: f.write(json.dumps(s[q], ensure_ascii=False) + "\n"); n[i] += 1; break
            else: f.write(json.dumps({"id": q, "type": "abstain"}, ensure_ascii=False) + "\n")
    os.replace(tmp, out)
    if not quiet:
        print(f"merge: {len(ids)} questions | " + " | ".join(f"{os.path.basename(os.path.dirname(p)) or p}: {k}" for p, k in zip(srcs, n)) +
              f" | abstain {len(ids) - sum(n)}", flush=True)
    return n


def main():
    merge(sys.argv[1], sys.argv[2], sys.argv[3:])


if __name__ == "__main__":
    main()
