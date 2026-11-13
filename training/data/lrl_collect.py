#!/usr/bin/env python3
"""Collect the card writer's outputs (src/lrl_gen.py: <results>/LRL*/<part>/cards_*.jsonl) into the card files the flow loads
(src/refs/lrl/: compile_train, compile_public, alias_train, alias_public, translit; induced candidates go to --induced-out for
tests/lrl_accept.py). Compiled cards get the key of the rulebook they were compiled from (src/lrl.rb_key) when missing; duplicates
(same id) are dropped. Usage: lrl_collect.py <results_dir>[,<results_dir>...] [--out src/refs/lrl] [--induced-out F]"""
import os, sys, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import lrl

PARTS = {"pub_compile": ("compile_public", "public"), "tr_compile": ("compile_train", "train"), "pub_alias": ("alias_public", None),
         "tr_alias": ("alias_train", None), "translit": ("translit", None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results"); ap.add_argument("--out", default=os.path.join(common.SRC, "refs", "lrl")); ap.add_argument("--induced-out", default="")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rk = {"public": lrl.rb_key(open(os.path.join(common.ROOT, "data", "raw", "public_test", "public_test", "rulebook.md"), encoding="utf-8").read()),
          "train": lrl.rb_key(open(os.path.join(common.ROOT, "data", "raw", "train", "train", "rulebook.md"), encoding="utf-8").read())}
    got = {}; induced = {}
    for root in a.results.split(","):
        for f in sorted(glob.glob(os.path.join(root, "**", "cards_*.jsonl"), recursive=True)):
            part = os.path.basename(os.path.dirname(f))
            rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
            if part.startswith("induce"):
                for c in rows: induced.setdefault(c["id"], c)
                continue
            if part not in PARTS: continue
            name, split = PARTS[part]
            for c in rows:
                if split and not (c.get("trigger") or {}).get("rulebook"): c.setdefault("trigger", {})["rulebook"] = rk[split]
                got.setdefault(name, {}).setdefault(c["id"], c)
    for name, cards in got.items():
        with open(os.path.join(a.out, name + ".jsonl"), "w", encoding="utf-8") as f:
            for c in cards.values(): f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"{name}: {len(cards)} cards", flush=True)
    if a.induced_out and induced:
        common.write_jl(a.induced_out, list(induced.values()))
        print(f"induced candidates: {len(induced)} -> {a.induced_out}", flush=True)


if __name__ == "__main__":
    main()
