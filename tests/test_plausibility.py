#!/usr/bin/env python3
"""No false positives: every reference mapping (training businesses + generic format adapters that resolve) passes
canon.plausibility on its own business, and the own reference gives no soft warning (canon.secondary_checks). Usage: test_plausibility.py <train_split_dir> <ingest_cache_dir>"""
import os, sys, glob, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import pandas as pd, canon, ingest, dm, agent
split, cache = sys.argv[1], sys.argv[2]
agent._init_worker(cache, split)
progs, meta = canon.ref_programs(), canon.load_meta(); bad = []; adapter_warn = []; n = 0
for f in sorted(glob.glob(os.path.join(cache, "biz_*.pkl"))):
    b = os.path.basename(f)[:-4]; biz = pickle.load(open(f, "rb"))
    for rid, code in progs.items():
        if rid != b and not (rid.startswith("fmt:") and canon.resolvable(biz, "generic")): continue
        r = canon.run_mapping(code, biz, meta.get(rid, {}).get("tables") if not rid.startswith("fmt:") else None)
        if not r.get("ok"): continue                                   # a format that does not fit: not a plausibility question
        n += 1; errs = canon.plausibility(r["C"], biz)
        if errs: bad.append((b, rid, errs))
        if r.get("warnings"): (bad if rid == b else adapter_warn).append((b, rid, r["warnings"]))   # soft checks too
for x in bad: print("FALSE POSITIVE", x)
for x in adapter_warn: print("adapter warning (allowed: an adapter may miss a table)", x[0], x[1], [w[:90] for w in x[2]])
print(f"{n} reference mappings checked, {len(bad)} flagged")
print("PASS" if not bad else "FAIL")
