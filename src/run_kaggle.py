#!/usr/bin/env python3
"""Kaggle job wrapper: install vLLM (from a mounted wheels dir if any, else PyPI), then run agent.py.
Usage (as a make_kernel job): run_kaggle.py <split: train|public_test> OUT <agent args...>"""
import os, sys, glob, subprocess, time

t = time.time()
# 1. vLLM from the mounted wheels (zc-assets/wheels, ~400-470 s of pip; proven on every L4 run). The prebuilt environment
#    (kernel output lelynhan/zc-env-vllm032, pip --target) is OPT-IN (ZC_USE_ENV=1): importing torch + vLLM straight from
#    the /kaggle/input mount (tens of thousands of small files) took > 900 s and killed all five v4 kernels (02/11).
#    Its probe can never raise: any failure or timeout falls back to the wheels.
env_dir = None
if os.environ.get("ZC_USE_ENV") == "1":
    env_dir = next((os.path.dirname(f) for f in glob.glob("/kaggle/input/*/*/*/ZC_ENV_MARKER") + glob.glob("/kaggle/input/*/ZC_ENV_MARKER")), None)
use_env = False
if env_dir:
    pe = dict(os.environ, PYTHONPATH=env_dir + os.pathsep + os.environ.get("PYTHONPATH", ""))
    try:
        r = subprocess.run([sys.executable, "-c", "import torch, vllm, python_calamine; print(vllm.__version__, torch.__version__, torch.cuda.is_available())"],
                           env=pe, capture_output=True, text=True, timeout=300)
        use_env = r.returncode == 0 and "True" in r.stdout; msg = r.stdout.strip() if use_env else (r.stdout + r.stderr)[-400:]
    except Exception as e: msg = repr(e)[:300]
    print("prebuilt env", env_dir, ("OK " if use_env else "FAILED: ") + msg, f"({time.time() - t:.0f}s)", flush=True)
if use_env:
    os.environ["PYTHONPATH"] = pe["PYTHONPATH"]; sys.path.insert(0, env_dir)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio", "torchcodec"], capture_output=True)
else:
    wh = next((d for d in glob.glob("/kaggle/input/**/wheels", recursive=True) if glob.glob(d + "/vllm-*.whl")), None)
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + (["--no-index", "--find-links", wh] if wh else []) + ["vllm"]
    r = subprocess.run(cmd, capture_output=True, text=True); print("pip vllm:", "wheels" if wh else "pypi", r.returncode, r.stderr[-300:], flush=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio", "torchcodec"], capture_output=True)
    cal = [sys.executable, "-m", "pip", "install", "-q"] + (["--no-index", "--find-links", wh] if wh else []) + ["python-calamine"]
    subprocess.run(cal, capture_output=True)                         # offline L4 sessions: calamine from the wheels too
print(f"setup {time.time() - t:.0f}s", flush=True)
data = os.environ["ZC_DATA"]
split = next(d for d in glob.glob(os.path.join(data, "**", sys.argv[1]), recursive=True) if os.path.isfile(os.path.join(d, "questions.jsonl")))
out = sys.argv[2]
args = sys.argv[3:]
if "--model" in args:                                                 # prefer weights mounted from nhatlong1103/zc-assets (offline sessions)
    i = args.index("--model") + 1
    local = next(iter(glob.glob(f"/kaggle/input/**/models/{os.path.basename(args[i])}/config.json", recursive=True)), None)
    if local:                                                          # copy once to local disk: 4 GPU processes reading the
        src = os.path.dirname(local); dst = "/tmp/models/" + os.path.basename(src)   # network mount took 18 min in E2b
        os.makedirs("/tmp/models", exist_ok=True)
        if not os.path.isfile(os.path.join(dst, "config.json")):
            t = time.time(); subprocess.run(["cp", "-r", src, dst]); print(f"model copied to {dst} in {time.time() - t:.0f}s", flush=True)
        args[i] = dst
os.makedirs(out, exist_ok=True)
try:
    import vllm; print("vllm", vllm.__version__, flush=True)
except Exception as e: print("vllm import:", e, flush=True)
# GPU monitor: one sample per GPU every 5 s (utilisation %, memory MiB) -> OUT/gpu_monitor.csv, summary -> gpu_summary.json
mon = subprocess.Popen(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits", "-l", "5"],
                       stdout=open(os.path.join(out, "gpu_monitor.csv"), "w"), stderr=subprocess.DEVNULL)
t0 = time.time()
script = "agent.py"                                                   # --script <file>: another job of src/ (qnorm_llm.py)
if "--script" in args: i = args.index("--script"); script = args[i + 1]; del args[i:i + 2]
rc = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), script), split, out] + args).returncode
mon.terminate()
import json, collections
g = collections.defaultdict(list)
for line in open(os.path.join(out, "gpu_monitor.csv")):
    try: i, u, m, tot = [float(x) for x in line.split(",")]; g[int(i)].append((u, m, tot))
    except ValueError: pass
summ = {f"gpu{i}": {"mean_util": round(sum(u for u, _, _ in v) / len(v), 1), "busy_share": round(sum(u >= 50 for u, _, _ in v) / len(v), 3),
                    "peak_mem_gb": round(max(m for _, m, _ in v) / 1024, 1), "total_gb": round(v[0][2] / 1024, 1), "samples": len(v)} for i, v in sorted(g.items())}
summ["wall_s"] = round(time.time() - t0)
json.dump(summ, open(os.path.join(out, "gpu_summary.json"), "w"), indent=1)
print("GPU", json.dumps(summ), flush=True)
sys.exit(rc)
