#!/usr/bin/env python3
"""Adapters -> weights vLLM serves (docs/08_MASTER_PLAN.md §5.9, QĐ3 / QĐ9).

merge  : W += scale * B @ A for every LoRA module, written into a copy of the ORIGINAL checkpoint (same tensor names, shards,
         config: vLLM loads it exactly like the base model, e.g. Qwen3_5ForConditionalGeneration with --quantization fp8). The
         adapter may come from the text-only class (module 'model.layers.N...') or the VL class ('model.language_model.layers.N...'):
         each module is matched to the checkpoint tensor that exists.
         Several adapters with weights (--weights 0.5,0.5) = linear merge of their deltas (model soup of LoRA branches).
vllm   : a copy of one adapter whose module names follow the checkpoint's naming and whose target_modules is a plain list (what
         vLLM's LoRA loader expects), for --lora serving.
Usage: merge.py merge <base_dir> <out_dir> <adapter_dir>[,<adapter_dir>...] [--weights w1,w2]
       merge.py vllm <base_dir> <adapter_dir> <out_dir>"""
import os, sys, json, shutil, time, re


def _adapter(path):
    from safetensors.torch import load_file
    cfg = json.load(open(os.path.join(path, "adapter_config.json")))
    f = os.path.join(path, "adapter_model.safetensors")
    if os.path.isfile(f): W = load_file(f)
    else:
        import torch; W = torch.load(os.path.join(path, "adapter_model.bin"), map_location="cpu")
    mods = {}
    for k, v in W.items():
        m = re.match(r"^(?:base_model\.model\.)?(.*)\.lora_([AB])(?:\.default)?\.weight$", k)
        if m: mods.setdefault(m.group(1), {})[m.group(2)] = v
    r = cfg.get("r", 8); alpha = cfg.get("lora_alpha", r)
    scale = alpha / (r ** 0.5) if cfg.get("use_rslora") else alpha / r
    return mods, scale, cfg


def _target_key(mod, keys):
    """the checkpoint tensor name of a LoRA module path."""
    cands = [mod + ".weight", mod.replace("model.", "model.language_model.", 1) + ".weight",
             mod.replace("model.language_model.", "model.", 1) + ".weight", "model." + mod + ".weight"]
    return next((c for c in cands if c in keys), None)


def merge(base, out, adapters, weights=None):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    t0 = time.time()
    idx_p = os.path.join(base, "model.safetensors.index.json")
    wmap = json.load(open(idx_p))["weight_map"] if os.path.isfile(idx_p) else {k: "model.safetensors" for k in safe_open(os.path.join(base, "model.safetensors"), "pt").keys()}
    keys = set(wmap)
    ads = [_adapter(p) for p in adapters]
    weights = weights or [1.0] * len(ads)
    delta_of = {}
    miss = []
    for (mods, scale, _), w in zip(ads, weights):
        for mod, ab in mods.items():
            k = _target_key(mod, keys)
            if k is None or "A" not in ab or "B" not in ab: miss.append(mod); continue
            delta_of.setdefault(k, []).append((ab["A"], ab["B"], scale * w))
    if miss: print(f"[merge] {len(miss)} modules without a checkpoint tensor, e.g. {miss[:3]}", flush=True)
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(base):
        if not f.endswith(".safetensors") and os.path.isfile(os.path.join(base, f)): shutil.copy(os.path.join(base, f), os.path.join(out, f))
    n = 0
    for shard in sorted(set(wmap.values())):
        T = {}
        with safe_open(os.path.join(base, shard), "pt") as fh:
            meta = fh.metadata()
            for k in fh.keys():
                t = fh.get_tensor(k)
                if k in delta_of:
                    acc = t.to(torch.float32)
                    for A, B, s in delta_of[k]: acc += s * (B.to(torch.float32) @ A.to(torch.float32))
                    t = acc.to(t.dtype); n += 1
                T[k] = t
        save_file(T, os.path.join(out, shard), metadata=meta or {"format": "pt"})
    print(f"[merge] {n} tensors updated from {len(ads)} adapter(s) (weights {weights}) -> {out} in {time.time() - t0:.0f}s", flush=True)
    return n


def vllm_adapter(base, adapter, out):
    """a copy of the adapter with checkpoint-style module names and a list of target leaf names."""
    from safetensors.torch import load_file, save_file
    idx_p = os.path.join(base, "model.safetensors.index.json")
    keys = set(json.load(open(idx_p))["weight_map"]) if os.path.isfile(idx_p) else set()
    W = load_file(os.path.join(adapter, "adapter_model.safetensors")); cfg = json.load(open(os.path.join(adapter, "adapter_config.json")))
    out_w, leaves = {}, set()
    for k, v in W.items():
        m = re.match(r"^(?:base_model\.model\.)?(.*)\.(lora_[AB])(?:\.default)?\.weight$", k)
        if not m: continue
        tk = _target_key(m.group(1), keys) if keys else m.group(1) + ".weight"
        mod = (tk or m.group(1) + ".weight")[:-len(".weight")]
        out_w[f"base_model.model.{mod}.{m.group(2)}.weight"] = v.contiguous(); leaves.add(mod.split(".")[-1])
    os.makedirs(out, exist_ok=True); save_file(out_w, os.path.join(out, "adapter_model.safetensors"))
    cfg["target_modules"] = sorted(leaves); cfg.pop("layers_to_transform", None)
    json.dump(cfg, open(os.path.join(out, "adapter_config.json"), "w"), indent=1)
    print(f"[merge] vLLM adapter: {len(out_w)} tensors, target_modules {cfg['target_modules']} -> {out}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "merge":
        w = [float(x) for x in sys.argv[sys.argv.index("--weights") + 1].split(",")] if "--weights" in sys.argv else None
        merge(sys.argv[2], sys.argv[3], sys.argv[4].split(","), w)
    elif cmd == "vllm": vllm_adapter(sys.argv[2], sys.argv[3], sys.argv[4])
    else: sys.exit(__doc__)
