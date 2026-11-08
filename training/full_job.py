#!/usr/bin/env python3
"""WP7 full training on one 4xL4 session (docs/08 §8 WP7, §5.9 'bản cuối'): LoRA SFT of the winning data mix with FSDP over the
4 GPUs (training/sft.py + training/fsdp4.yaml: micro-batch 8, ~2k tok/s measured in WP1), the adapter merged into the base
checkpoint (training/merge.py), then the merged model evaluated through the hybrid flow (src/eval_flow.py, --dp 4).
OUT/: adapter/ (published for the Docker build: docker/fetch_assets.sh merges it), train metrics, evaluation submission + logs.

Usage (Kaggle job): run_kaggle.py public_test OUT/full --script training/full_job.py --model Qwen/Qwen3.5-9B --train
        --files d1.jsonl,d2.jsonl,... [--data sft] [--r 64 --alpha 128 --lr 1e-4 --epochs 1 --budget 0] [--mb 8 --eff-batch 64]
        [--sets quick400,pub_full] [--eval-config configs/e0.json] [--subset N] [--no-eval]
       (smoke: --budget 3e5 --subset 40 checks FSDP save -> merge -> evaluation end to end)"""
import os, sys, json, time, glob, shutil, argparse, subprocess

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
PY = sys.executable
SRC = ROOT if os.path.isfile(os.path.join(ROOT, "agent.py")) else os.path.join(ROOT, "src")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("out"); ap.add_argument("--model", required=True)
    ap.add_argument("--files", required=True); ap.add_argument("--data", default="sft")
    ap.add_argument("--r", type=int, default=64); ap.add_argument("--alpha", type=int, default=128); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0); ap.add_argument("--budget", type=float, default=0)
    ap.add_argument("--mb", type=int, default=8); ap.add_argument("--eff-batch", type=int, default=64); ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--sets", default="quick400,pub_full"); ap.add_argument("--eval-config", default="configs/e0.json")
    ap.add_argument("--max-len", type=int, default=3072); ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--no-eval", action="store_true"); ap.add_argument("--subset", type=int, default=0, help="evaluation: first N questions per shop")
    a = ap.parse_args()
    os.chdir(ROOT); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    D = a.data if os.path.isabs(a.data) else os.path.join(HERE, "data", "out", a.data)
    data = ",".join(os.path.join(D, f) for f in a.files.split(","))
    tmp = "/tmp/full_" + os.path.basename(os.path.normpath(a.out)); os.makedirs(tmp, exist_ok=True)   # one folder per job
    cfg = os.path.join(HERE, f"fsdp{a.gpus}.yaml")
    cmd = [PY, "-m", "accelerate.commands.launch", "--config_file", cfg, "--num_processes", str(a.gpus), "--main_process_port", "29531",
           os.path.join(HERE, "sft.py"), "--model", a.model, "--data", data, "--out", f"{tmp}/adapter", "--mb", str(a.mb), "--eff-batch", str(a.eff_batch),
           "--r", str(a.r), "--alpha", str(a.alpha), "--lr", str(a.lr), "--epochs", str(a.epochs), "--max-tokens", str(a.budget),
           "--metrics", os.path.join(a.out, "train_metrics.json"), "--log-every", "10", "--max-len", str(a.max_len), "--seed", str(a.seed)]
    rc = subprocess.run(cmd, stdout=open(os.path.join(a.out, "train.log"), "w"), stderr=subprocess.STDOUT).returncode
    print(f"[full] training rc {rc} in {time.time() - t0:.0f}s", flush=True)
    if rc != 0 or not os.path.isfile(f"{tmp}/adapter/adapter_config.json"):
        print("[full] training failed:\n" + open(os.path.join(a.out, "train.log"), errors="ignore").read()[-3000:], flush=True); sys.exit(1)
    shutil.copytree(f"{tmp}/adapter", os.path.join(a.out, "adapter"), dirs_exist_ok=True)
    if a.no_eval: return
    t1 = time.time()
    rc = subprocess.run([PY, os.path.join(HERE, "merge.py"), "merge", a.model, f"{tmp}/merged", f"{tmp}/adapter"],
                        stdout=open(os.path.join(a.out, "merge.log"), "w"), stderr=subprocess.STDOUT).returncode
    print(f"[full] merge rc {rc} in {time.time() - t1:.0f}s", flush=True)
    if rc != 0: sys.exit(2)
    t2 = time.time(); ev = f"{tmp}/eval"
    cmd = [PY, os.path.join(SRC, "eval_flow.py"), a.split, ev, "--sets", a.sets, "--model", f"{tmp}/merged", "--dp", str(a.gpus),
           "--config", a.eval_config, "--cache", "/tmp/zc_cache"] + (["--", "--subset", str(a.subset)] if a.subset else [])
    rc = subprocess.run(cmd, stdout=open(os.path.join(a.out, "eval.log"), "w"), stderr=subprocess.STDOUT).returncode
    for f in ("submission.jsonl", "logs.jsonl", "compile_logs.jsonl"):
        p = os.path.join(ev, "run", f)
        if os.path.isfile(p): shutil.copy(p, a.out)
    print(f"[full] eval rc {rc} in {time.time() - t2:.0f}s; total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
