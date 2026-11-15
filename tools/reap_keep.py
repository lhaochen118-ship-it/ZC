#!/usr/bin/env python3
"""P5 plan P01: the expert keep-list of IAmTheRealGumbo/Qwen3.6-35B-A3B-REAP-64-experts, found by matching its router rows
(mlp.gate.weight, 64 x 2048 per layer) to the 256 rows of the official Qwen/Qwen3.6-35B-A3B-FP8 router (kept in bf16 there).
Output: JSON {layer: [kept original expert ids in REAP order]} -> configs/moe/keep_reap64.json"""
import sys, json, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hf_tensors as ht
REAP, FULL = "IAmTheRealGumbo/Qwen3.6-35B-A3B-REAP-64-experts", "Qwen/Qwen3.6-35B-A3B-FP8"
wr, wf = ht.weight_map(REAP), ht.weight_map(FULL)
keep = {}
for L in range(40):
    n = f"model.language_model.layers.{L}.mlp.gate.weight"
    r = ht.get(REAP, n, wr[n]); f = ht.get(FULL, n, wf[n])
    ids = []
    for row in r:
        d = np.abs(f - row[None, :]).max(1); j = int(d.argmin())
        if d[j] > 0: print(f"layer {L}: nearest row differs by {d[j]:.3g}", flush=True)
        ids.append(j)
    assert len(set(ids)) == len(ids), f"layer {L}: duplicate match"
    keep[str(L)] = ids; print(L, len(ids), ids[:8], flush=True)
os.makedirs(sys.argv[1] if len(sys.argv) > 1 else "configs/moe", exist_ok=True)
out = os.path.join(sys.argv[1] if len(sys.argv) > 1 else "configs/moe", "keep_reap64.json")
json.dump({"source": REAP, "full": FULL, "keep": keep}, open(out, "w")); print("->", out)
