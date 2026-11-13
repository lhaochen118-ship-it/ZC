#!/usr/bin/env python3
"""Archive the transferable part of scratch runs into Process/runs/<run>/<result>/ (handoff to a new chat / container):
logs.jsonl.gz, submission.jsonl.gz, result.json, train_metrics.json, gpu_summary.json, summary*.json, LRL card files (cards_*,
raw_*), clean split ids (*_clean.txt), the tail of a failed run's train.log. Skips runs already archived; never copies adapters /
checkpoints (they stay on Kaggle: Process/kaggle/kernels.csv).
Usage: archive_runs.py <scratch runs dir> [--dest Process/runs]"""
import os, sys, glob, gzip, shutil
src = sys.argv[1]
dest = sys.argv[sys.argv.index("--dest") + 1] if "--dest" in sys.argv else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Process", "runs")
KEEP_GZ = ("logs.jsonl", "submission.jsonl", "judge.jsonl")
KEEP = ("result.json", "train_metrics.json", "gpu_summary.json")
n = 0
for run in sorted(os.listdir(src)):
    rd = os.path.join(src, run)
    if not os.path.isdir(rd) or os.path.isdir(os.path.join(dest, run)): continue
    for res in glob.glob(os.path.join(rd, "*_out", "results", "*")) + glob.glob(os.path.join(rd, "*_out", "results", "*", "*")):
        if not os.path.isdir(res) or os.path.basename(res) in ("adapter", "ckpt") or "/ckpt/" in res: continue
        files = [f for f in os.listdir(res) if os.path.isfile(os.path.join(res, f))]
        pick = [f for f in files if f in KEEP_GZ or f in KEEP or f.startswith(("summary", "cards_", "raw_")) or f.endswith("_clean.txt")]
        if not pick and "train.log" not in files: continue
        rel = os.path.relpath(res, os.path.join(rd))
        out = os.path.join(dest, run, *[p for p in rel.split(os.sep) if not p.endswith("_out") and p != "results"])
        os.makedirs(out, exist_ok=True)
        for f in pick:
            if f in KEEP_GZ or f.startswith("raw_"):
                with open(os.path.join(res, f), "rb") as a, gzip.open(os.path.join(out, f + ".gz"), "wb", 9) as b: shutil.copyfileobj(a, b)
            else: shutil.copy(os.path.join(res, f), out)
        if "train.log" in files and not os.path.isfile(os.path.join(res, "train_metrics.json")):   # a failed training: keep the tail
            lines = open(os.path.join(res, "train.log"), encoding="utf-8", errors="replace").read().splitlines()[-80:]
            open(os.path.join(out, "train_log_tail.txt"), "w").write("\n".join(l[:400] for l in lines))
        n += 1
print(f"archived {n} result dirs -> {dest}")
