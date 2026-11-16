#!/usr/bin/env python3
import os, sys, json, glob, time, argparse, subprocess, re, struct, shutil, math

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
SRC = ROOT if os.path.isfile(os.path.join(ROOT, "stage_hf.py")) else os.path.join(ROOT, "src")
sys.path.insert(0, SRC)


def find_adapter(name):
    if os.path.isfile(os.path.join(name, "adapter_config.json")): return name
    c = sorted(glob.glob(f"/kaggle/input/**/{name}/adapter/adapter_config.json", recursive=True))
    if not c: sys.exit(f"[merge_fp8] adapter {name} not found")
    return os.path.dirname(c[0])


def block_quant(W, bs=128):
    import torch
    o, i = W.shape; po, pi = (-o) % bs, (-i) % bs
    Wp = torch.nn.functional.pad(W, (0, pi, 0, po))
    B = Wp.reshape((o + po) // bs, bs, (i + pi) // bs, bs)
    amax = B.abs().amax(dim=(1, 3)).clamp(min=1e-12)
    s = amax / 448.0
    q = (B / s[:, None, :, None]).clamp(-448, 448).reshape(o + po, i + pi)[:o, :i].to(torch.float8_e4m3fn)
    return q, s


def dequant(q, s, bs=128):
    import torch
    o, i = q.shape
    return q.float() * s.float().repeat_interleave(bs, 0)[:o].repeat_interleave(bs, 1)[:, :i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("out"); ap.add_argument("--name", required=True); ap.add_argument("--adapter", required=True)
    ap.add_argument("--weight", type=float, default=1.0); ap.add_argument("--keep", default="configs/moe/keep_e88.json")
    ap.add_argument("--fp8-dir", default=""); ap.add_argument("--repo", default="Qwen/Qwen3.6-35B-A3B-FP8")
    a = ap.parse_args(); t0 = time.time()
    import torch, stage_hf as st
    from safetensors.torch import load_file
    dst = os.path.join(a.out, "models", a.name)
    if a.fp8_dir: shutil.copytree(a.fp8_dir, dst, dirs_exist_ok=True)
    else:
        rc = subprocess.run([sys.executable, os.path.join(SRC, "stage_hf.py"), a.split, "/tmp/fp8slice", "--repo", a.repo, "--name", a.name,
                             "--keep", a.keep]).returncode
        if rc: sys.exit(rc)
        shutil.move(os.path.join("/tmp/fp8slice", "models", a.name), dst)
    ad = find_adapter(a.adapter); cfg = json.load(open(os.path.join(ad, "adapter_config.json")))
    r = float(cfg.get("r", 8))
    scale = a.weight * float(cfg.get("lora_alpha", 16)) / (math.sqrt(r) if cfg.get("use_rslora") else r)
    if cfg.get("use_dora") or cfg.get("modules_to_save"): sys.exit("[merge_fp8] DoRA / modules_to_save adapters are not supported")
    sd = load_file(os.path.join(ad, "adapter_model.safetensors"))
    pairs, xp = {}, {}
    for k, v in sd.items():
        if "visual" in k or "mtp" in k: continue
        m = re.search(r"layers\.(\d+)\.mlp\.experts(\.base_layer)?\.lora_([AB])(?:\.default)?\.weight$", k)
        if m:
            xp.setdefault((int(m.group(1)), m.group(2) or ""), {})[m.group(3)] = v.float(); continue
        m = re.search(r"layers\.(\d+)\.(.+)\.lora_([AB])(?:\.default)?\.weight$", k)
        if not m: continue
        pairs.setdefault(f"model.language_model.layers.{m.group(1)}.{m.group(2)}.weight", {})[m.group(3)] = v.float()
    deltas = {n: (lambda p=p: scale * (p["B"] @ p["A"])) for n, p in pairs.items()}
    if xp:
        tc = json.load(open(os.path.join(dst, "config.json"))); tc = tc.get("text_config", tc)
        I, E, r = int(tc["moe_intermediate_size"]), int(tc["num_experts"]), int(cfg.get("r", 8))
        for (L, _), ab in xp.items():
            A, B = ab["A"], ab["B"]
            if A.shape[0] != E * r: sys.exit(f"[merge_fp8] layer {L}: expert LoRA A {tuple(A.shape)} != {E} experts x r {r}")
            fused = B.shape[0] == 2 * I
            for e in range(E):
                pre = f"model.language_model.layers.{L}.mlp.experts.{e}."
                d = lambda A=A, B=B, e=e: scale * (B[:, e::E] @ A[e * r:(e + 1) * r])
                if fused:
                    deltas[pre + "gate_proj.weight"] = (lambda d=d: d()[:I]); deltas[pre + "up_proj.weight"] = (lambda d=d: d()[I:])
                else: deltas[pre + "down_proj.weight"] = d
        print(f"[merge_fp8] expert LoRA on {len(xp)} fused expert tensors ({E} experts, I {I}, r {r})", flush=True)
    print(f"[merge_fp8] {len(pairs)} LoRA modules + {len(deltas) - len(pairs)} expert matrices from {ad} (scale {scale:.3f})", flush=True)
    idx = json.load(open(os.path.join(dst, "model.safetensors.index.json")))["weight_map"]
    by_file = {}
    for name in deltas:
        if name not in idx: sys.exit(f"[merge_fp8] {name} not in the FP8 checkpoint")
        by_file.setdefault(idx[name], []).append(name)
    done, maxerr = 0, 0.0
    for f, names in by_file.items():
        p = os.path.join(dst, f); h, base, meta = st.read_st(p)
        raw = open(p, "rb").read()
        out = []
        for n, m in sorted(h.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            b0, b1 = m["data_offsets"]; out.append([n, m["dtype"], m["shape"], raw[base + b0: base + b1]])
        pos = {n: i for i, (n, _, _, _) in enumerate(out)}
        for n in names:
            qn, sn = out[pos[n]], out[pos[n + "_scale_inv"]]
            if qn[1] != "F8_E4M3": sys.exit(f"[merge_fp8] {n} is {qn[1]}, not block FP8")
            q = torch.frombuffer(bytearray(qn[3]), dtype=torch.float8_e4m3fn).reshape(qn[2])
            sdt = {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}[sn[1]]
            s = torch.frombuffer(bytearray(sn[3]), dtype=sdt).reshape(sn[2])
            W = dequant(q, s) + deltas[n]()
            q2, s2 = block_quant(W)
            maxerr = max(maxerr, float((dequant(q2, s2) - W).abs().max() / (W.abs().max() + 1e-12)))
            qn[3] = q2.contiguous().view(torch.uint8).numpy().tobytes(); sn[3] = s2.to(sdt).contiguous().view(torch.uint8).numpy().tobytes() \
                if sdt != torch.float32 else s2.to(torch.float32).numpy().tobytes()
            done += 1
        st.write_st(p, [tuple(x) for x in out], meta)
    c = json.load(open(os.path.join(dst, "config.json"))); c["_lora_merged"] = {"adapter": a.adapter, "weight": a.weight, "modules": done}
    json.dump(c, open(os.path.join(dst, "config.json"), "w"), indent=2)
    print(f"[merge_fp8] {done} modules merged into {dst}; max relative requantisation error {maxerr:.4f}; {time.time() - t0:.0f}s", flush=True)
    if done != len(deltas): sys.exit("[merge_fp8] not every LoRA module was merged")


if __name__ == "__main__":
    main()
