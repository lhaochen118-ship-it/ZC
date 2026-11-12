#!/usr/bin/env python3
"""Self-hint labels for existing pure-prompt records (e.g. RFT records of the pure flow, training/data/rft_from_logs.py --pure):
the label (last message) is prefixed with the <hints> block the rule hint module reads from the record's question
(training/data/pure_records.self_hints), so these records teach the same output format as pure_records.py --self-hints.
Prompts are kept as they are (already without hints).
Usage: add_self_hints.py <train_split> <ingest_cache> <canon_dir> <in.jsonl> <out.jsonl> [--mode full]"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from pure_records import self_hints


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("cache"); ap.add_argument("canon_dir"); ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--mode", default="full")
    a = ap.parse_args(); t0 = time.time(); rows = []; skipped = 0
    for r in common.jl(a.inp):
        if not r.get("messages") or r["messages"][-1].get("role") != "assistant" or "<hints>" in r["messages"][-1]["content"]:
            skipped += 1; rows.append(r); continue
        c = common.ctx(a.split, a.cache, a.canon_dir, r["business_id"], a.mode)
        c.hint_opts = {"counts_hints": True}
        r = dict(r, messages=r["messages"][:-1] + [dict(r["messages"][-1], content=self_hints(c.hints(r["question"], r["as_of_date"])) + r["messages"][-1]["content"])])
        r["source"] = (r.get("source") or "RFT") + "h"; rows.append(r)
    common.write_jl(a.out, rows)
    print(f"{a.inp}: {len(rows)} records ({skipped} unchanged) -> {a.out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
