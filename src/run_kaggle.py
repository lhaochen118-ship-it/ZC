#!/usr/bin/env python3
"""Kaggle job wrapper: install vLLM (from a mounted wheels dir if any, else PyPI), then run agent.py.
Usage (as a make_kernel job): run_kaggle.py <split: train|public_test> OUT <agent args...>"""
import os, sys, glob, subprocess, time

t = time.time()
wh = next((d for d in glob.glob("/kaggle/input/**/wheels", recursive=True) if glob.glob(d + "/vllm-*.whl")), None)
cmd = [sys.executable, "-m", "pip", "install", "-q"] + (["--no-index", "--find-links", wh] if wh else []) + ["vllm"]
r = subprocess.run(cmd, capture_output=True, text=True); print("pip vllm:", "wheels" if wh else "pypi", r.returncode, r.stderr[-300:], flush=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio", "torchcodec"], capture_output=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "python-calamine"], capture_output=True)
print(f"setup {time.time() - t:.0f}s", flush=True)
data = os.environ["ZC_DATA"]
split = next(d for d in glob.glob(os.path.join(data, "**", sys.argv[1]), recursive=True) if os.path.isfile(os.path.join(d, "questions.jsonl")))
out = sys.argv[2]
args = sys.argv[3:]
if "--model" in args:                                                 # prefer weights mounted from nhatlong1103/zc-assets (offline sessions)
    i = args.index("--model") + 1
    local = next(iter(glob.glob(f"/kaggle/input/**/models/{os.path.basename(args[i])}/config.json", recursive=True)), None)
    if local: args[i] = os.path.dirname(local); print("model from mounted assets:", args[i], flush=True)
os.execv(sys.executable, [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent.py"), split, out] + args)
