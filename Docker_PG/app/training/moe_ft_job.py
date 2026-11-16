#!/usr/bin/env python3
import os, sys, json, time, glob, argparse, subprocess

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
PY = sys.executable
SRC = ROOT if os.path.isfile(os.path.join(ROOT, "agent.py")) else os.path.join(ROOT, "src")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("out"); ap.add_argument("--name", required=True); ap.add_argument("--files", required=True)
    ap.add_argument("--data-dir", default="p5ft"); ap.add_argument("--keep", default="configs/moe/keep_e88.json")
    ap.add_argument("--src", default="qwen3-6-35b-a3b")
    ap.add_argument("--r", type=int, default=32); ap.add_argument("--alpha", type=int, default=64); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0); ap.add_argument("--mb", type=int, default=2); ap.add_argument("--eff-batch", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=4608); ap.add_argument("--max-tokens", type=float, default=0); ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--init-adapter", default="", help="16/11: continue from this adapter (name under training/data/out/<dir>/<NAME>/adapter or a path)")
    ap.add_argument("--lora-experts", type=int, default=0, help="16/11 MS3: LoRA on the routed experts too (PEFT target_parameters)")
    ap.add_argument("--seed", type=int, default=61); ap.add_argument("--fake", action="store_true", help="CPU test: no accelerate, 1 process")
    a = ap.parse_args(); t0 = time.time(); os.makedirs(a.out, exist_ok=True)
    base = "/tmp/e88b/models/E88"
    if not os.path.isfile(os.path.join(base, "config.json")):
        rc = subprocess.run([PY, os.path.join(SRC, "stage_hf.py"), a.split, "/tmp/e88b", "--repo", "Qwen/Qwen3.6-35B-A3B", "--name", "E88",
                             "--keep", a.keep, "--src-dir", a.src], stdout=open(os.path.join(a.out, "slice.log"), "w"), stderr=subprocess.STDOUT).returncode
        print(f"[moe_ft] bf16 slice rc {rc} in {time.time() - t0:.0f}s: " + open(os.path.join(a.out, "slice.log")).read()[-300:], flush=True)
        if rc: sys.exit(rc)
    data = []
    for f in a.files.split(","):
        c = [p for p in (os.path.join(ROOT, "training", "data", "out", a.data_dir, f), os.path.join(HERE, "data", "out", a.data_dir, f)) if os.path.isfile(p)]
        c = c or glob.glob(f"/kaggle/input/**/{f}", recursive=True)
        if not c: sys.exit(f"[moe_ft] data {f} not found")
        data.append(c[0])
    cmd = ([PY] if a.fake else [PY, "-m", "accelerate.commands.launch", "--config_file", os.path.join(HERE, "fsdp4_moe.yaml"),
                                "--num_processes", "4", "--main_process_port", "29533"]) + \
          [os.path.join(HERE, "sft.py"), "--model", base, "--data", ",".join(data), "--out", os.path.join(a.out, a.name, "adapter"),
           "--r", str(a.r), "--alpha", str(a.alpha), "--lr", str(a.lr), "--epochs", str(a.epochs), "--mb", str(a.mb),
           "--eff-batch", str(a.eff_batch), "--max-len", str(a.max_len), "--seed", str(a.seed), "--max-steps", str(a.max_steps),
           "--metrics", os.path.join(a.out, a.name, "train_metrics.json"), "--ckpt-dir", os.path.join(a.out, a.name, "ckpt")] + \
          (["--max-tokens", str(a.max_tokens)] if a.max_tokens else [])
    os.makedirs(os.path.join(a.out, a.name), exist_ok=True)
    ia = ""
    if a.init_adapter:
        c = [a.init_adapter] if os.path.isfile(os.path.join(a.init_adapter, "adapter_config.json")) else sorted(glob.glob(os.path.join(ROOT, "training", "data", "out", "*", a.init_adapter, "adapter", "adapter_config.json"))) + sorted(glob.glob(f"/kaggle/input/**/{a.init_adapter}/adapter/adapter_config.json", recursive=True))
        if not c: sys.exit(f"[moe_ft] init adapter {a.init_adapter} not found")
        ia = c[0] if os.path.isdir(c[0]) else os.path.dirname(c[0]); print(f"[moe_ft] init adapter {ia}", flush=True)
    tries = [(ei, mb) for ei, mb in (("grouped_mm", 8), ("grouped_mm", 4), ("batched_mm", 4), ("grouped_mm", 2), ("batched_mm", 2), ("eager", 2))
             if mb <= max(a.mb, 8)] if not a.fake else [("eager", a.mb)]
    t1 = time.time(); rc = 1; used = None
    for ei, mb in tries:
        c2 = list(cmd); c2[c2.index("--mb") + 1] = str(mb)
        env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", ZC_EXPERTS_IMPL=ei, ZC_LORA_EXPERTS=str(int(a.lora_experts)), ZC_INIT_ADAPTER=ia)
        ta = time.time(); log = os.path.join(a.out, a.name, f"train_{ei}_mb{mb}.log")
        with open(log, "w") as lf: rc = subprocess.run(c2, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        tail = open(log, errors="ignore").read()[-1500:]
        print(f"[moe_ft] experts {ei} mb {mb}: rc {rc} in {time.time() - ta:.0f}s\n{tail[-600:]}", flush=True)
        if rc == 0: used = (ei, mb); break
    res = {"name": a.name, "rc": rc, "lora_experts": a.lora_experts, "init_adapter": ia, "experts_impl": used, "train_s": round(time.time() - t1), "total_s": round(time.time() - t0), "files": a.files, "keep": a.keep}
    json.dump(res, open(os.path.join(a.out, a.name, "result.json"), "w"), indent=1); print("[moe_ft]", json.dumps(res), flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
