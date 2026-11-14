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
    ap.add_argument("--eval-configs", default="", help="several configs evaluated one after the other on the SAME merged model "
                    "(one merge for an A/B; results in <out>/<config name>/)")
    ap.add_argument("--parallel-gpus", type=int, default=0, help="with --eval-configs: run the configs at the same time, one GPU group "
                    "each (4 L4: 4 variants of 1 GPU; 2: 2 variants of 2 GPUs) instead of one after the other on every GPU (user 14/11)")
    ap.add_argument("--runner", default="eval", choices=["eval", "predict", "judge", "script"],
                    help="predict: the submission path (src/predict_flow.py --config, whole split) instead of eval_flow's sets; "
                         "judge: plan 5 W3 (src/judge.py) on a solver run's saved logs with the merged model")
    ap.add_argument("--judge-logs", default="", help="judge: the solver log file (name under training/data/out, shipped with --train-data)")
    ap.add_argument("--judge-units", default=""); ap.add_argument("--judge-variants", default="list")
    ap.add_argument("--judge-k", type=int, default=5); ap.add_argument("--judge-only-split", type=int, default=1)
    ap.add_argument("--judge-links", default="", help="judge: links.jsonl per --judge-logs entry (comma-separated, '-' = none)")
    ap.add_argument("--script", default="", help="runner script: a src/ script run as <script> <split> <out>/script --model <merged> "
                    "<--script-args> (e.g. lrl_gen.py --mode bind: the rule writer with the solver's merged weights)")
    ap.add_argument("--script-args", default="", help="runner script: extra arguments, space-separated (use ',' for spaces inside none)")
    ap.add_argument("--script-shards", type=int, default=0, help="runner script: N processes with --shard i/N, one GPU each")
    ap.add_argument("--judge-lora", default="", help="judge: a LoRA-W adapter (branch name under --search, or a directory) applied at "
                    "run time on the merged model (vLLM LoRA) instead of being merged (--adapters ...,W --weights ...,1.0)")
    a = ap.parse_args()
    os.chdir(ROOT); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    a.tmp = a.tmp or "/tmp/soup_" + os.path.basename(os.path.normpath(a.out))      # one folder per job (two soups in a kernel)
    names = [x for x in a.adapters.split(",") if x]
    if names == ["NONE"]:                                               # the base model on the same sets (no merge)
        os.makedirs(a.tmp, exist_ok=True); return evaluate_all(a, a.model, {"adapters": "NONE", "merge_rc": 0, "merge_s": 0}, t0)
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
    evaluate_all(a, f"{a.tmp}/merged", res, t0)


def evaluate_all(a, model, res, t0):
    """one evaluation (--eval-config) or several on the same merged model (--eval-configs), then the merged copy removed."""
    cfgs = [c for c in a.eval_configs.split(",") if c] if a.eval_configs else []
    if a.runner == "judge":
        judge(a, model, res, t0)
    elif a.runner == "script":
        script(a, model, res, t0)
    elif not cfgs:
        evaluate(a, model, res, t0, a.out, a.eval_config)
    elif a.parallel_gpus > 1 and len(cfgs) > 1:                        # one experiment variant per GPU group, all at once
        import queue, threading
        sys.path.insert(0, SRC); import predict_flow
        devs = predict_flow.visible_gpus() or [str(i) for i in range(a.parallel_gpus)]
        k = min(a.parallel_gpus, len(cfgs), max(1, len(devs)))
        per = max(1, len(devs) // k)
        groups = queue.Queue()
        for i in range(k): groups.put(devs[i * per:(i + 1) * per] or [devs[i % len(devs)]])
        nw = max(2, (os.cpu_count() or 8) // k)                         # CPU workers split between the variants (no oversubscription)
        print(f"[soup] {len(cfgs)} configs, {k} at a time on GPU groups of {per} ({nw} CPU workers each)", flush=True)
        def run(c):
            g = groups.get()
            try:
                name = os.path.splitext(os.path.basename(c))[0]; out = os.path.join(a.out, name); os.makedirs(out, exist_ok=True)
                evaluate(a, model, dict(res, config=c, gpus=g), t0, out, c, tag=name, gpus=g, workers=nw)
            finally: groups.put(g)
        sem = threading.Semaphore(k)
        def guarded(c):
            with sem: run(c)
        ts = [threading.Thread(target=guarded, args=(c,)) for c in cfgs]
        for t in ts: t.start()
        for t in ts: t.join()
    else:
        for c in cfgs:
            name = os.path.splitext(os.path.basename(c))[0]; out = os.path.join(a.out, name); os.makedirs(out, exist_ok=True)
            evaluate(a, model, dict(res, config=c), t0, out, c, tag=name)
    if model != a.model: shutil.rmtree(model, ignore_errors=True)       # 19 GB merged copy: the next soup of the kernel needs the space


def judge(a, model, res, t0):
    """plan 5 W3: src/judge.py on saved solver logs (one variant per GPU inside judge.py); outputs under <out>/judge/ (several
    --judge-logs, comma-separated: one after the other, under <out>/judge/<log name>/)."""
    t1 = time.time(); logs = [x for x in a.judge_logs.split(",") if x]
    units = [x for x in a.judge_units.split(",")] if a.judge_units else []
    lora = None
    if a.judge_lora:
        lora = find_adapter(a.judge_lora, a.search)
        if lora is None: sys.exit(f"[soup] judge adapter {a.judge_lora} not found under {a.search}")
        conv = os.path.join(a.tmp, "lora_vllm")                         # module names / target_modules as vLLM's loader expects
        rc = subprocess.run([PY, os.path.join(HERE, "merge.py"), "vllm", model, lora, conv], stdout=open(os.path.join(a.out, "lora_conv.log"), "w"),
                            stderr=subprocess.STDOUT).returncode     # (14/11 ejb1lora: the raw adapter loaded but had no effect)
        if rc == 0 and os.path.isfile(os.path.join(conv, "adapter_config.json")): lora = conv
        else: sys.exit(f"[soup] merge.py vllm failed rc {rc}")
    rcs = []
    for n, lg in enumerate(logs):
        out = os.path.join(a.out, "judge") if len(logs) == 1 else os.path.join(a.out, "judge", lg.split(".")[0])
        os.makedirs(out, exist_ok=True)
        cmd = [PY, os.path.join(SRC, "judge.py"), a.split, out, "--model", model, "--logs", lg, "--variants", a.judge_variants,
               "--k", str(a.judge_k), "--only-split", str(a.judge_only_split)]
        u = units[n] if n < len(units) else (units[0] if len(units) == 1 and len(logs) == 1 else "")
        if u: cmd += ["--units", u]
        if lora: cmd += ["--lora", lora]
        lk = [x for x in a.judge_links.split(",")] if a.judge_links else []
        if n < len(lk) and lk[n] not in ("", "-"): cmd += ["--links", lk[n]]
        if a.fake_responder: cmd += ["--fake-responder", a.fake_responder]
        if a.subset: cmd += ["--subset", str(a.subset)]
        rc = subprocess.run(cmd, stdout=open(os.path.join(a.out, f"judge{n}.log"), "w"), stderr=subprocess.STDOUT).returncode
        rcs.append(rc)
        print(f"[soup] judge {lg} rc {rc}\n" + open(os.path.join(a.out, f"judge{n}.log"), errors="ignore").read()[-800:], flush=True)
    res.update(judge_rc=rcs, judge_s=round(time.time() - t1), judge_logs=a.judge_logs, judge_variants=a.judge_variants, judge_lora=a.judge_lora)
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    print(f"[soup] judge done {rcs} in {res['judge_s']}s; total {time.time() - t0:.0f}s", flush=True)
    if not any(rc == 0 for rc in rcs): sys.exit(5)


def script(a, model, res, t0):
    """a src/ script on the merged model; --script-shards N: N processes (--shard i/N), one GPU each."""
    t1 = time.time(); out = os.path.join(a.out, "script"); os.makedirs(out, exist_ok=True)
    base = [PY, os.path.join(SRC, a.script), a.split, out, "--model", model] + [x for x in a.script_args.replace(",", " ").split() if x]
    if a.fake_responder: base += ["--fake-responder", a.fake_responder]
    n = max(1, a.script_shards)
    sys.path.insert(0, SRC); import predict_flow
    devs = predict_flow.visible_gpus() or [str(i) for i in range(n)]
    procs = []
    for i in range(n):
        env = dict(os.environ)
        if not a.fake_responder and n > 1: env["CUDA_VISIBLE_DEVICES"] = devs[i % len(devs)]
        procs.append(subprocess.Popen(base + (["--shard", f"{i}/{n}"] if n > 1 else []), env=env,
                                      stdout=open(os.path.join(a.out, f"script{i}.log"), "w"), stderr=subprocess.STDOUT))
    rcs = [p.wait() for p in procs]
    res.update(script=a.script, script_rc=rcs, script_s=round(time.time() - t1))
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    for i in range(n): print(f"[soup] script shard {i} rc {rcs[i]}:\n" + open(os.path.join(a.out, f"script{i}.log"), errors="ignore").read()[-800:], flush=True)
    if not any(rc == 0 for rc in rcs): sys.exit(6)


def evaluate(a, model, res, t0, out_dir, eval_config, tag="eval", gpus=None, workers=0):
    t1 = time.time(); ev = f"{a.tmp}/{tag}"
    dp = len(gpus) if gpus else a.dp
    cache = f"{a.tmp}/cache_{tag}" if gpus else f"{a.tmp}/cache"        # parallel variants: no shared cache writes
    if a.runner == "predict":                                          # the container's entry point on the whole split
        cmd = [PY, os.path.join(SRC, "predict_flow.py"), a.split, ev, "--model", model, "--dp", str(dp), "--config", eval_config,
               "--runs", "1", "--work", f"{a.tmp}/work_{tag}"]
        if a.fake_responder: cmd += ["--fake-responder", a.fake_responder]   # predict_flow's own flag (tests)
        if a.subset: cmd += ["--", "--subset", str(a.subset)]
    else:
        cmd = [PY, os.path.join(SRC, "eval_flow.py"), a.split, ev, "--sets", a.sets, "--model", model, "--dp", str(dp),
               "--config", eval_config, "--cache", cache] + (["--workers", str(workers)] if workers else [])
        if a.fake_responder: cmd += ["--", "--fake-responder", a.fake_responder]
        if a.subset: cmd += ([] if a.fake_responder else ["--"]) + ["--subset", str(a.subset)]
    env = dict(os.environ, ZC_BASE_MODEL=a.model)                     # "units_model": "base" studies only (W = the solver's weights)
    if gpus and not a.fake_responder: env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    rc = subprocess.run(cmd, stdout=open(os.path.join(out_dir, "eval.log"), "w"), stderr=subprocess.STDOUT, env=env).returncode
    for f in ("submission.jsonl", "logs.jsonl", "compile_logs.jsonl", os.path.join("units", "cards_unit.jsonl"), os.path.join("links", "links.jsonl")):
        for p in (os.path.join(ev, "run", f), os.path.join(ev, f)):
            if os.path.isfile(p): shutil.copy(p, out_dir); break
    for p in __import__("glob").glob(os.path.join(ev, "units", "raw_unit*.jsonl")): shutil.copy(p, out_dir)   # plan 4 W: what the writer wrote
    if a.runner == "predict" and os.path.isdir(os.path.join(ev, "logs")):
        shutil.copytree(os.path.join(ev, "logs"), os.path.join(out_dir, "logs"), dirs_exist_ok=True)
    res.update(eval_rc=rc, eval_s=round(time.time() - t1))
    json.dump(res, open(os.path.join(out_dir, "result.json"), "w"), indent=1)
    print(f"[soup] eval rc {rc} in {res['eval_s']}s; total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
