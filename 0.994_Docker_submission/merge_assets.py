#!/usr/bin/env python3
"""Docker build stage (docker/Dockerfile): write the served weights /w/model from the base checkpoint /w/base and /w/adapter,
which holds either one LoRA adapter (adapter_config.json) or a soup (soup.json: {"members": [dir, ...], "weights": [w, ...]},
each member a LoRA adapter folder; WP7 / QĐ9 linear soup), or NONE (base weights served as they are). training/merge.py
writes the deltas into the original tensors (same names: vLLM serves it like the base, --quantization fp8)."""
import os, sys, json, shutil
W = os.environ.get("MERGE_ROOT", "/w")                                 # tests: another root
sys.path.insert(0, W)
import merge

A, BASE, OUT = os.path.join(W, "adapter"), os.path.join(W, "base"), os.path.join(W, "model")
if os.path.isfile(os.path.join(A, "adapter_config.json")):
    merge.merge(BASE, OUT, [A])
elif os.path.isfile(os.path.join(A, "soup.json")):
    s = json.load(open(os.path.join(A, "soup.json")))
    members = [os.path.join(A, m) for m in s["members"]]
    w = [float(x) for x in s.get("weights") or [1.0 / len(members)] * len(members)]
    assert len(w) == len(members) and all(os.path.isfile(os.path.join(m, "adapter_config.json")) for m in members), s
    merge.merge(BASE, OUT, members, w)
else:
    os.rename(BASE, OUT); print("no adapter: base weights"); sys.exit(0)
shutil.rmtree(BASE); print("merged ->", OUT)
