#!/usr/bin/env python3
import os, sys, json, time, glob, signal, shutil, subprocess, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "out", "smoke")
PY = sys.executable


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("out")
    ap.add_argument("--model", required=True); ap.add_argument("--part", default="train", choices=["train", "infer"])
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--steps", type=int, default=8)
    return ap.parse_args()


class Runner:
    def __init__(self, out, dry):
        self.out, self.dry = out, dry; self.R = {}; os.makedirs(out, exist_ok=True)
        self.rf = os.path.join(out, "results.json")

    def save(self): json.dump(self.R, open(self.rf, "w"), indent=1, default=str)

    def start(self, name, cmd, gpus, timeout, env=None):
        log = os.path.join(self.out, name + ".log")
        e = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", **(env or {}))
        print(f"[smoke] start {name} on GPUs {gpus}: {' '.join(cmd)[:300]}", flush=True)
        if self.dry: return (name, None, time.time(), timeout, log)
        p = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=e, start_new_session=True)
        return (name, p, time.time(), timeout, log)

    def wait(self, jobs):
        res = {}
        while jobs:
            for j in list(jobs):
                name, p, t0, lim, log = j
                if p is None: res[name] = (0, ""); jobs.remove(j); continue
                rc = p.poll()
                if rc is None and time.time() - t0 > lim:
                    try: os.killpg(p.pid, signal.SIGKILL)
                    except Exception: pass
                    rc = -9
                if rc is not None:
                    txt = open(log, errors="ignore").read() if os.path.isfile(log) else ""
                    res[name] = (rc, txt); jobs.remove(j)
                    print(f"[smoke] done {name} rc={rc} {time.time() - t0:.0f}s", flush=True)
            time.sleep(5)
        return res


def metrics_of(txt, tag):
    for line in reversed(txt.splitlines()):
        if line.startswith(tag):
            try: return json.loads(line[len(tag):].strip())
            except Exception: return None
    return None


def oom(txt): return "CUDA out of memory" in txt or "OutOfMemoryError" in txt or "out of memory" in txt.lower()


def sft_cmd(model, data, out, mb, steps, eff, launcher=None, arch="causal", extra=()):
    base = ["training/sft.py", "--model", model, "--data", data, "--out", out, "--mb", str(mb), "--max-steps", str(steps),
            "--eff-batch", str(eff), "--metrics", out + ".metrics.json", "--arch", arch] + list(extra)
    if launcher: return [PY, "-m", "accelerate.commands.launch", "--config_file", launcher[0], "--num_processes", str(launcher[1]), "--main_process_port", str(launcher[2])] + base
    return [PY] + base


def sweep(r, name, gpus, mbs, mk, timeout):
    tried = []
    for mb in mbs:
        jobs = [r.start(f"{name}_mb{mb}", mk(mb), gpus, timeout)]
        rc, txt = r.wait(jobs)[f"{name}_mb{mb}"]
        m = metrics_of(txt, "[sft] METRICS")
        tried.append({"mb": mb, "rc": rc, "oom": oom(txt), "metrics": m, "tail": txt[-1500:] if rc else ""})
        if rc == 0 and m: return {"best_mb": mb, "tried": tried, **m}
        if r.dry: return {"best_mb": mb, "tried": tried}
        if not oom(txt) and rc != -9: break
    return {"best_mb": None, "tried": tried}


def main():
    a = parse()
    os.chdir(os.path.dirname(HERE))
    out = os.path.join(a.out, "smoke") if not a.out.rstrip("/").endswith("smoke") else a.out
    r = Runner(out, a.dry_run); R = r.R
    tmp = "/tmp/smoke"; os.makedirs(tmp, exist_ok=True)
    tr, ev = os.path.join(DATA, "d1_smoke_train.jsonl"), os.path.join(DATA, "d1_smoke_eval.jsonl")
    R["env"] = env_info(); r.save()
    if a.part == "train":
        S = a.steps
        jA = []
        def one_gpu(name, gpu, extra=()):
            return lambda mb: sft_cmd(a.model, tr, f"{tmp}/{name}_mb{mb}", mb, S, mb * 4, None, "causal", ["--no-save"] + list(extra))
        import threading
        res = {}
        def th(key, fn): res[key] = fn()
        ts = [threading.Thread(target=th, args=("sft_1gpu", lambda: sweep(r, "sft_1gpu", "0", [4, 2, 1], one_gpu("sft_1gpu", 0), 1500))),
              threading.Thread(target=th, args=("unsloth", lambda: unsloth_sweep(r, a, tr, tmp, S))),
              threading.Thread(target=th, args=("sft_fsdp2", lambda: fsdp_sweep(r, a, tr, tmp, S, "2,3", 2, [8, 4, 2, 1], 29512)))]
        for t in ts: t.start()
        for t in ts: t.join()
        R.update(res); r.save()
        R["sft_fsdp4"] = fsdp_sweep(r, a, tr, tmp, S, "0,1,2,3", 4, [8, 4, 2], 29513); r.save()
        one = R.get("sft_1gpu", {}); two = R.get("sft_fsdp2", {}); uns = R.get("unsloth", {})
        if two.get("best_mb"):
            cfg = two.get("config") or "training/fsdp2.yaml"
            cmd = sft_cmd(a.model, tr, f"{tmp}/adapter", two["best_mb"], 30, 16, (cfg, 2, 29514), "causal")
            gpus = "0,1"
        elif one.get("best_mb"):
            cmd = sft_cmd(a.model, tr, f"{tmp}/adapter", one["best_mb"], 30, 16, None, "causal"); gpus = "0"
        elif uns.get("best_mb"):
            cmd = [PY, "training/sft_unsloth.py", "--model", a.model, "--data", tr, "--mb", str(uns["best_mb"]), "--max-steps", "30",
                   "--eff-batch", "16", "--out", f"{tmp}/adapter"]; gpus = "0"
        else: cmd = None
        if cmd:
            env = {"PYTHONPATH": "/tmp/uns_env" + os.pathsep + os.environ.get("PYTHONPATH", "")} if "sft_unsloth.py" in " ".join(cmd) else None
            rc, txt = r.wait([r.start("sft_main", cmd, gpus, 3600, env)])["sft_main"]
            R["sft_main"] = {"rc": rc, "metrics": metrics_of(txt, "[sft] METRICS"), "tail": txt[-2000:] if rc else ""}
            r.save()
            ok_ad = os.path.isfile(f"{tmp}/adapter/adapter_config.json")
            if ok_ad and not a.dry_run: shutil.copytree(f"{tmp}/adapter", os.path.join(out, "adapter"), dirs_exist_ok=True)
            jobs = [r.start("merge", [PY, "training/merge.py", "merge", a.model, f"{tmp}/merged", f"{tmp}/adapter"], "", 2400),
                    r.start("lora_copy", [PY, "training/merge.py", "vllm", a.model, f"{tmp}/adapter", f"{tmp}/adapter_vllm"], "", 600)]
            rr = r.wait(jobs); R["merge"] = {k: {"rc": v[0], "tail": v[1][-800:]} for k, v in rr.items()}; r.save()
            g = lambda name, model, gpu, extra: r.start(name, [PY, "training/vllm_gen.py", "--model", model, "--prompts", ev, "--out", f"{out}/{name}.jsonl",
                                                               "--quant", "fp8", "--max-tokens", "200"] + extra, gpu, 1800)
            jobs = [g("gen_merged", f"{tmp}/merged", "0", []), g("gen_lora", a.model, "1", ["--lora", f"{tmp}/adapter_vllm"]), g("gen_base", a.model, "2", [])]
            for k, (rc, txt) in r.wait(jobs).items():
                R[k] = {"rc": rc, "summary": metrics_of(txt, "SUMMARY"), "tail": txt[-1500:] if rc else "", **score_gen(f"{out}/{k}.jsonl", ev)}
            r.save()
    else:
        gd = os.path.join(DATA, "grammars")
        g = lambda name, model, gpu, extra, prompts=ev, tl=1800: r.start(name, [PY, "training/vllm_gen.py", "--model", model, "--prompts", prompts,
                                                                             "--out", f"{out}/{name}.jsonl"] + extra, gpu, tl)
        jobs = [g("gen_grammar", a.model, "0", ["--quant", "fp8", "--max-tokens", "200", "--grammar-dir", gd]),
                g("gen_free", a.model, "1", ["--quant", "fp8", "--max-tokens", "200"])]
        for k, (rc, txt) in r.wait(jobs).items():
            R[k] = {"rc": rc, "summary": metrics_of(txt, "SUMMARY"), "tail": txt[-1500:] if rc else "", **score_gen(f"{out}/{k}.jsonl", ev)}
        r.save()
        tp = os.path.join(DATA, "teacher_prompts.jsonl")
        q27, g31 = find_model(["qwen3-6-27b", "qwen3.6-27b"]), find_model(["gemma-4-31b-it"], exclude=["qat", "w4a16"])
        R["teacher_paths"] = {"qwen27": q27, "gemma31": g31}
        jobs = []
        if q27: jobs.append(g("teacher_qwen27", q27, "0,1", ["--tp", "2", "--max-tokens", "400", "--temp", "0.7", "--max-num-seqs", "64"], tp, 3000))
        if g31: jobs.append(g("teacher_gemma31", g31, "2,3", ["--tp", "2", "--quant", "fp8", "--max-tokens", "400", "--temp", "0.7", "--max-num-seqs", "64"], tp, 3000))
        for k, (rc, txt) in r.wait(jobs).items():
            R[k] = {"rc": rc, "summary": metrics_of(txt, "SUMMARY"), "tail": txt[-2500:] if rc else ""}
        r.save()
    print("[smoke] RESULTS " + json.dumps({k: (v.get("best_mb") if isinstance(v, dict) and "best_mb" in v else (v.get("summary") if isinstance(v, dict) else None))
                                           for k, v in R.items()}, default=str), flush=True)


def fsdp_sweep(r, a, tr, tmp, S, gpus, n, mbs, port):
    cfg2, cfg1 = f"training/fsdp{n}.yaml", f"training/fsdp{n}_v1.yaml"
    res = sweep(r, f"sft_fsdp{n}", gpus, mbs, lambda mb: sft_cmd(a.model, tr, f"{tmp}/fsdp{n}_mb{mb}", mb, S, mb * n * 2, (cfg2, n, port), "causal", ["--no-save"]), 1800)
    res["config"] = cfg2
    if res.get("best_mb") is None and not r.dry:
        res1 = sweep(r, f"sft_fsdp{n}v1", gpus, mbs, lambda mb: sft_cmd(a.model, tr, f"{tmp}/fsdp{n}v1_mb{mb}", mb, S, mb * n * 2, (cfg1, n, port + 10), "causal", ["--no-save"]), 1800)
        res1["config"] = cfg1; res1["fsdp2_failed"] = res["tried"]; return res1
    return res


def unsloth_sweep(r, a, tr, tmp, S):
    wh = next(iter(glob.glob("/kaggle/input/**/wheels_unsloth", recursive=True)), None)
    if r.dry: wh = wh or "/kaggle/input/x/wheels_unsloth"
    if not wh: return {"skipped": "no wheels_unsloth mounted"}
    env_dir = "/tmp/uns_env"
    if not r.dry:
        p = subprocess.run([PY, "-m", "pip", "install", "-q", "--no-deps", "--target", env_dir] + sorted(glob.glob(wh + "/*.whl")), capture_output=True, text=True)
        if p.returncode != 0: return {"skipped": "pip --target failed: " + (p.stdout + p.stderr)[-600:]}
    env = {"PYTHONPATH": env_dir + os.pathsep + os.environ.get("PYTHONPATH", "")}
    tried = []
    for mb in (4, 2):
        name = f"unsloth_mb{mb}"
        cmd = [PY, "training/sft_unsloth.py", "--model", a.model, "--data", tr, "--mb", str(mb), "--max-steps", str(S), "--eff-batch", str(mb * 4),
               "--out", f"{tmp}/unsloth_adapter"]
        j = r.start(name, cmd, "1", 1500, env)
        rc, txt = r.wait([j])[name]
        m = metrics_of(txt, "[sft] METRICS"); tried.append({"mb": mb, "rc": rc, "oom": oom(txt), "metrics": m, "tail": txt[-1200:] if rc else ""})
        if rc == 0 and m: return {"best_mb": mb, "tried": tried, **m}
        if r.dry: return {"best_mb": mb, "tried": tried}
        if not oom(txt): break
    return {"best_mb": None, "tried": tried}


def find_model(keys, exclude=()):
    for c in sorted(glob.glob("/kaggle/input/**/config.json", recursive=True)):
        d = os.path.dirname(c); low = d.lower()
        if any(k in low for k in keys) and not any(x in low for x in exclude) and glob.glob(d + "/*.safetensors"): return d
    return None


def score_gen(path, ev):
    import ast, re
    for d in ("src", "."):
        if os.path.isfile(os.path.join(d, "prompting.py")) and os.path.abspath(d) not in sys.path: sys.path.insert(0, os.path.abspath(d))
    try:
        import prompting
        labels = {json.loads(l)["id"]: json.loads(l)["messages"][2]["content"] for l in open(ev, encoding="utf-8")}
        outs = [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.isfile(path) else []
    except Exception as e: return {"score_error": str(e)[:200]}
    def norm(c): return re.sub(r"\s+", "", c or "")
    valid = exact = 0
    for o in outs:
        p = prompting.parse_output(o["text"]); lab = prompting.parse_output(labels.get(o["id"], ""))
        try:
            if p["code"] is not None: ast.parse(p["code"]); valid += 1
        except SyntaxError: pass
        if p["code"] is not None and norm(p["code"]) == norm(lab["code"]): exact += 1
    n = max(1, len(outs))
    return {"n_out": len(outs), "valid": round(valid / n, 3), "exact": round(exact / n, 3), "example": outs[0]["text"][:400] if outs else ""}


def env_info():
    info = {}
    try: info["gpus"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=60).stdout.strip().splitlines()
    except Exception as e: info["gpus"] = str(e)
    for m in ("torch", "vllm", "transformers", "peft", "trl", "accelerate", "datasets", "xgrammar", "fla", "unsloth"):
        try:
            mod = __import__(m); info[m] = getattr(mod, "__version__", "?")
        except Exception as e: info[m] = f"missing ({type(e).__name__})"
    try:
        from transformers.utils.import_utils import is_flash_linear_attention_available, is_causal_conv1d_available
        info["fla_fast_path"] = bool(is_flash_linear_attention_available()); info["causal_conv1d"] = bool(is_causal_conv1d_available())
    except Exception as e: info["fla_fast_path"] = str(e)[:100]
    try:
        import psutil; info["ram_gb"] = round(psutil.virtual_memory().total / 2**30); info["disk_tmp_gb"] = round(shutil.disk_usage("/tmp").free / 2**30)
    except Exception: pass
    return info


if __name__ == "__main__":
    main()
