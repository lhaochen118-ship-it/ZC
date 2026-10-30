#!/usr/bin/env python3
"""Pull the outputs of the Kaggle kernels listed in handoff/RUNNING_KERNELS.md (rows `| run | `acc/slug` | ...`) into
<out_dir>/<slug>/ (new chat: the local monitors of the previous chat are gone, the kernels kept running on Kaggle).
Kernels still RUNNING are reported and skipped. Usage: pull_kernels.py [out_dir] [--only run1,run2]"""
import sys, os, re, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
if only: args = [a for a in args if a != ",".join(only)]
out = args[0] if args else "runs_pulled"
rows = re.findall(r"^\| (\S+) \| `([\w-]+)/([\w-]+)` \|", open(os.path.join(ROOT, "handoff", "RUNNING_KERNELS.md"), encoding="utf-8").read(), re.M)
for run, acc, slug in rows:
    if only and run not in only: continue
    env = dict(os.environ, KAGGLE_API_TOKEN=open(os.path.expanduser(f"~/.kaggle/tok_{acc}")).read().strip())
    st = subprocess.run(["kaggle", "kernels", "status", f"{acc}/{slug}"], capture_output=True, text=True, env=env).stdout.strip()
    if "COMPLETE" not in st and "ERROR" not in st:
        print(f"{run:8s} {acc}/{slug}: not finished ({st[-40:]})"); continue
    d = os.path.join(out, slug); os.makedirs(d, exist_ok=True)
    subprocess.run(["kaggle", "kernels", "output", f"{acc}/{slug}", "-p", d, "-o"], capture_output=True, text=True, env=env)
    print(f"{run:8s} {acc}/{slug}: {st[-30:]} -> {d}", flush=True)
