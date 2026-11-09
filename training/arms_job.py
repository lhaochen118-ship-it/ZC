#!/usr/bin/env python3
"""WP6 fine-tuning branches on one 4xL4 session (docs/08 §8 WP6, QĐ1: one branch per GPU with Unsloth, 4 at once). Per branch,
on its own GPU: LoRA SFT (training/sft_unsloth.py, token budget) -> adapter merged into the base checkpoint (training/merge.py)
-> evaluation of the merged model through the hybrid flow on the evaluation sets (src/eval_flow.py, --dp 1) -> OUT/<branch>/.
A branch that fails leaves its log; the others go on.

Usage (Kaggle job): run_kaggle.py train OUT/arms --script training/arms_job.py --model Qwen/Qwen3.5-9B --train
        --arms "A1=d1.jsonl+A2=d1.jsonl,d2.jsonl,d2n.jsonl+..." [--data sft] [--budget 4e6] [--r 32 --alpha 64 --lr 2e-4 --epochs 1]
        [--mb 4 --eff-batch 64] [--sets quick400] [--eval-config configs/e0.json] [--hf]   (data files in training/data/out/<data>/)"""
import os, sys, json, time, glob, shutil, argparse, subprocess

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
PY = sys.executable
SRC = ROOT if os.path.isfile(os.path.join(ROOT, "agent.py")) else os.path.join(ROOT, "src")


def unsloth_env():
    env_dir = "/tmp/uns_env"
    if not os.path.isdir(env_dir):
        wh = next(iter(glob.glob("/kaggle/input/**/wheels_unsloth", recursive=True)), None)
        if wh:
            r = subprocess.run([PY, "-m", "pip", "install", "-q", "--no-deps", "--target", env_dir] + sorted(glob.glob(wh + "/*.whl")), capture_output=True, text=True)
            print("[arms] unsloth env:", r.returncode, (r.stdout + r.stderr)[-300:], flush=True)
    return {"PYTHONPATH": env_dir + os.pathsep + os.environ.get("PYTHONPATH", "")}


def branch(i, name, files, a, D, base_env):
    import copy
    files, _, over = files.partition("|")                              # per-branch overrides: "files|r=64,alpha=128,lr=1e-4"
    if over:
        a = copy.copy(a)
        for kv in over.split(","):
            k, v = kv.split("="); setattr(a, k.strip().replace("-", "_"), type(getattr(a, k.strip().replace("-", "_")))(v))
    out = os.path.join(a.out, name); os.makedirs(out, exist_ok=True); tmp = f"/tmp/arms/{name}"; os.makedirs(tmp, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    if a.fake_responder: env.pop("CUDA_VISIBLE_DEVICES")
    data = ",".join(os.path.join(D, f) for f in files.split(","))
    res = {"name": name, "data": files, "gpu": i, "r": a.r, "alpha": a.alpha, "lr": a.lr, "epochs": a.epochs, "budget": a.budget,
           "max_len": a.max_len, "seed": a.seed, "mb": a.mb, "eval_config": a.eval_config}
    t0 = time.time()
    if name.upper().startswith("BASE"):                                  # the base model, no adapter (reference of the same sets)
        res["train_s"] = 0; ad = None
    elif a.adapters_dir:                                                 # evaluation only: an adapter trained earlier
        ad = os.path.join(a.adapters_dir, name, "adapter")
        if not os.path.isfile(os.path.join(ad, "adapter_config.json")):
            res["error"] = f"no adapter {ad}"; json.dump(res, open(f"{out}/result.json", "w"), indent=1); return res
        res["train_s"] = 0
    else:
        if a.hf:
            cmd = [PY, os.path.join(HERE, "sft.py"), "--model", a.model, "--data", data, "--out", f"{tmp}/adapter", "--mb", str(a.mb), "--eff-batch", str(a.eff_batch),
                   "--max-tokens", str(a.budget), "--r", str(a.r), "--alpha", str(a.alpha), "--lr", str(a.lr), "--epochs", str(a.epochs), "--metrics", f"{out}/train_metrics.json",
                   "--max-len", str(a.max_len), "--seed", str(a.seed)]
        else:
            env.update(base_env)
            cmd = [PY, os.path.join(HERE, "sft_unsloth.py"), "--model", a.model, "--data", data, "--out", f"{tmp}/adapter", "--mb", str(a.mb), "--eff-batch", str(a.eff_batch),
                   "--max-tokens", str(a.budget), "--r", str(a.r), "--alpha", str(a.alpha), "--lr", str(a.lr), "--epochs", str(a.epochs),
                   "--max-len", str(a.max_len), "--seed", str(a.seed)]
        rc = subprocess.run(cmd, env=env, stdout=open(f"{out}/train.log", "w"), stderr=subprocess.STDOUT).returncode
        txt = open(f"{out}/train.log", errors="ignore").read()
        m = [l for l in txt.splitlines() if l.startswith("[sft] METRICS ")]
        res["train"] = json.loads(m[-1][14:]) if m else {"rc": rc, "tail": txt[-2000:]}
        res["train_s"] = round(time.time() - t0)
        if rc != 0 or not os.path.isfile(f"{tmp}/adapter/adapter_config.json"):
            res["error"] = "training failed"; json.dump(res, open(f"{out}/result.json", "w"), indent=1); return res
        shutil.copytree(f"{tmp}/adapter", f"{out}/adapter", dirs_exist_ok=True)
        ad = f"{tmp}/adapter"
        if a.no_eval:                                                    # soup members (WP7): adapters only, evaluated merged
            res["merge_s"] = res["eval_s"] = 0; json.dump(res, open(f"{out}/result.json", "w"), indent=1)
            print(f"[arms] {name}: train {res['train_s']}s (no evaluation)", flush=True); return res
    # evaluation with the adapter served by vLLM (--lora): no merged copy is written (4 merged 19 GB copies at once turned
    # /tmp read-only on Kaggle in round 1); merging is for the final model only (QĐ3)
    t1 = time.time()
    if ad is not None:
        rc = subprocess.run([PY, os.path.join(HERE, "merge.py"), "vllm", a.model, ad, f"{tmp}/adapter_vllm"],
                            stdout=open(f"{out}/merge.log", "w"), stderr=subprocess.STDOUT).returncode
        if rc != 0: res["error"] = "adapter conversion failed"; json.dump(res, open(f"{out}/result.json", "w"), indent=1); return res
    res["merge_s"] = round(time.time() - t1)
    t2 = time.time(); ev = f"{tmp}/eval"; dp = max(1, int(a.eval_dp))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(str(i * dp + g) for g in range(dp)), PYTHONUNBUFFERED="1")
    if a.fake_responder: env.pop("CUDA_VISIBLE_DEVICES")
    cmd = [PY, os.path.join(SRC, "eval_flow.py"), a.split, ev, "--sets", a.sets, "--model", a.model] + (["--lora", f"{tmp}/adapter_vllm"] if ad else []) + \
          ["--dp", str(dp), "--config", a.eval_config, "--cache", f"/tmp/zc_cache_{i}", "--workers", str(max(2, (os.cpu_count() or 8) // 4))]
    if a.fake_responder: cmd += ["--", "--fake-responder", a.fake_responder]
    if a.subset: cmd += ([] if a.fake_responder else ["--"]) + ["--subset", str(a.subset)]
    rc = subprocess.run(cmd, env=env, stdout=open(f"{out}/eval.log", "w"), stderr=subprocess.STDOUT).returncode
    res["eval_s"] = round(time.time() - t2); res["eval_rc"] = rc
    for f in ("submission.jsonl", "logs.jsonl", "compile_logs.jsonl"):
        p = os.path.join(ev, "run", f)
        if os.path.isfile(p): shutil.copy(p, out)
    json.dump(res, open(f"{out}/result.json", "w"), indent=1)
    print(f"[arms] {name}: train {res['train_s']}s merge {res['merge_s']}s eval {res['eval_s']}s rc {rc}", flush=True)
    return res


def adapters_root(given, names, search="/kaggle/input"):
    """the folder holding <name>/adapter/adapter_config.json for every trained branch: the given one, else found under the mounted
    datasets (Kaggle mounts a dataset at a path that differs from its slug: /kaggle/input/datasets/<owner>/<slug>)."""
    ok = lambda r: all(os.path.isfile(os.path.join(r, n, "adapter", "adapter_config.json")) for n in names)
    if os.path.isdir(given) and ok(given): return given
    cands = sorted({os.path.dirname(os.path.dirname(os.path.dirname(c))) for c in glob.glob(os.path.join(search, "**", "adapter", "adapter_config.json"), recursive=True)})
    good = [r for r in cands if ok(r)]
    if not good: print(f"[arms] no folder under {search} holds adapters of {names}: {cands[:5]}", flush=True); return given
    pref = [r for r in good if os.path.basename(r.rstrip("/")) == os.path.basename(given.rstrip("/"))]
    return (pref or good)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("out"); ap.add_argument("--model", required=True)
    ap.add_argument("--arms", required=True); ap.add_argument("--data", default="sft")
    ap.add_argument("--budget", type=float, default=4e6); ap.add_argument("--r", type=int, default=32); ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--mb", type=int, default=4); ap.add_argument("--eff-batch", type=int, default=64)
    ap.add_argument("--sets", default="quick400"); ap.add_argument("--eval-config", default="configs/e0.json"); ap.add_argument("--hf", action="store_true")
    ap.add_argument("--fake-responder", default="", help="CPU test: evaluation with a fake model (no vLLM)"); ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--adapters-dir", default="", help="evaluation only: <dir>/<branch>/adapter trained earlier (a mounted dataset)")
    ap.add_argument("--eval-dp", type=int, default=1, help="GPUs per branch evaluation (a single BASE branch: 4)")
    ap.add_argument("--max-len", type=int, default=3072, help="training sequence length (full-prompt records: 5120 with --mb 2)")
    ap.add_argument("--seed", type=int, default=20261108, help="training seed (record sample and order)")
    ap.add_argument("--no-eval", action="store_true", help="train and save the adapters only (soup members: training/soup_job.py)")
    a = ap.parse_args()
    os.chdir(ROOT)
    D = a.data if os.path.isabs(a.data) else os.path.join(HERE, "data", "out", a.data)
    arms = [x.split("=", 1) for x in a.arms.replace(";", "+").split("+") if x.strip()]   # '+' between branches (';' separates jobs)
    trained = [n for n, _ in arms if not n.upper().startswith("BASE")]
    if a.adapters_dir: a.adapters_dir = adapters_root(a.adapters_dir, trained)
    ngpu = int(os.environ.get("ZC_NGPU", "4"))
    if len(arms) * max(1, a.eval_dp) > ngpu and not a.fake_responder: sys.exit(f"{len(arms)} branches x --eval-dp {a.eval_dp} > {ngpu} GPUs")
    base_env = {} if (a.hf or a.adapters_dir or not trained) else unsloth_env()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(len(arms)) as ex:
        res = list(ex.map(lambda ia: branch(ia[0], ia[1][0], ia[1][1], a, D, base_env), enumerate(arms)))
    json.dump(res, open(os.path.join(a.out, "arms.json"), "w"), indent=1, default=str)
    print("[arms] DONE " + json.dumps([{k: r.get(k) for k in ("name", "train_s", "merge_s", "eval_s", "error")} for r in res]), flush=True)


if __name__ == "__main__":
    main()
