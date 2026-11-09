#!/usr/bin/env python3
"""QĐ9 (docs/08 WP6 G19): a linear merge ('model soup') of several trained LoRA adapters written into the base weights
(training/merge.py merge, deltas weighted, default equal weights 1/n), then the merged model evaluated through the hybrid flow
(src/eval_flow.py, --dp 4) on the same sets as the single branches. Adapters are found by branch name under the mounted
datasets (<root>/<name>/adapter/adapter_config.json) or given as directories.

Usage (Kaggle job): run_kaggle.py public_test OUT/soup --script training/soup_job.py --model Qwen/Qwen3.5-9B
        --adapters A1,A2T64,B3MIX[,...] | NONE (the base) [--weights 0.4,0.3,0.3] (WiSE-FT: weights summing to alpha < 1) [--sets quick400,pub_ho] [--eval-config configs/e0.json]
        [--search /kaggle/input] [--fake-responder F --subset N] (CPU test)"""
import os, sys, glob, json, time, shutil, argparse, subprocess

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
PY = sys.executable
SRC = ROOT if os.path.isfile(os.path.join(ROOT, "agent.py")) else os.path.join(ROOT, "src")


def find_adapter(name, search):
    if os.path.isfile(os.path.join(name, "adapter_config.json")): return name
    c = sorted(glob.glob(os.path.join(search, "**", name, "adapter", "adapter_config.json"), recursive=True))
    return os.path.dirname(c[0]) if c else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("out"); ap.add_argument("--model", required=True); ap.add_argument("--adapters", required=True)
    ap.add_argument("--weights", default=""); ap.add_argument("--sets", default="quick400,pub_ho"); ap.add_argument("--eval-config", default="configs/e0.json")
    ap.add_argument("--search", default="/kaggle/input"); ap.add_argument("--dp", type=int, default=4); ap.add_argument("--tmp", default="")
    ap.add_argument("--fake-responder", default=""); ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--runner", default="eval", choices=["eval", "predict"],
                    help="predict: the submission path (src/predict_flow.py --config, whole split) instead of eval_flow's sets")
    a = ap.parse_args()
    os.chdir(ROOT); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    a.tmp = a.tmp or "/tmp/soup_" + os.path.basename(os.path.normpath(a.out))      # one folder per job (two soups in a kernel)
    names = [x for x in a.adapters.split(",") if x]
    if names == ["NONE"]:                                               # the base model on the same sets (no merge)
        os.makedirs(a.tmp, exist_ok=True); return evaluate(a, a.model, {"adapters": "NONE", "merge_rc": 0, "merge_s": 0}, t0)
    dirs = [find_adapter(n, a.search) for n in names]
    miss = [n for n, d in zip(names, dirs) if d is None]
    if miss: sys.exit(f"[soup] adapters not found under {a.search}: {miss}")
    w = [float(x) for x in a.weights.split(",")] if a.weights else [1.0 / len(dirs)] * len(dirs)
    if len(w) != len(dirs): sys.exit("[soup] one weight per adapter")
    shutil.rmtree(a.tmp, ignore_errors=True); os.makedirs(a.tmp)
    rc = subprocess.run([PY, os.path.join(HERE, "merge.py"), "merge", a.model, f"{a.tmp}/merged", ",".join(dirs), "--weights", ",".join(map(str, w))],
                        stdout=open(os.path.join(a.out, "merge.log"), "w"), stderr=subprocess.STDOUT).returncode
    res = {"adapters": dict(zip(names, dirs)), "weights": w, "merge_rc": rc, "merge_s": round(time.time() - t0)}
    print(f"[soup] merge rc {rc} in {res['merge_s']}s: {names} x {w}", flush=True)
    if rc != 0: json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1); sys.exit(2)
    evaluate(a, f"{a.tmp}/merged", res, t0)


def evaluate(a, model, res, t0):
    t1 = time.time(); ev = f"{a.tmp}/eval"
    if a.runner == "predict":                                          # the container's entry point on the whole split
        cmd = [PY, os.path.join(SRC, "predict_flow.py"), a.split, ev, "--model", model, "--dp", str(a.dp), "--config", a.eval_config,
               "--runs", "1", "--work", f"{a.tmp}/work"]
        if a.fake_responder: cmd += ["--fake-responder", a.fake_responder]   # predict_flow's own flag (tests)
        if a.subset: cmd += ["--", "--subset", str(a.subset)]
    else:
        cmd = [PY, os.path.join(SRC, "eval_flow.py"), a.split, ev, "--sets", a.sets, "--model", model, "--dp", str(a.dp),
               "--config", a.eval_config, "--cache", f"{a.tmp}/cache"]
        if a.fake_responder: cmd += ["--", "--fake-responder", a.fake_responder]
        if a.subset: cmd += ([] if a.fake_responder else ["--"]) + ["--subset", str(a.subset)]
    rc = subprocess.run(cmd, stdout=open(os.path.join(a.out, "eval.log"), "w"), stderr=subprocess.STDOUT).returncode
    for f in ("submission.jsonl", "logs.jsonl", "compile_logs.jsonl"):
        for p in (os.path.join(ev, "run", f), os.path.join(ev, f)):
            if os.path.isfile(p): shutil.copy(p, a.out); break
    if a.runner == "predict" and os.path.isdir(os.path.join(ev, "logs")):
        shutil.copytree(os.path.join(ev, "logs"), os.path.join(a.out, "logs"), dirs_exist_ok=True)
    res.update(eval_rc=rc, eval_s=round(time.time() - t1))
    if model != a.model: shutil.rmtree(model, ignore_errors=True)       # 19 GB merged copy: the next soup of the kernel needs the space
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    print(f"[soup] eval rc {rc} in {res['eval_s']}s; total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
