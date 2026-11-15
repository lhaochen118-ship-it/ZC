#!/usr/bin/env python3
"""Archive the transferable part of the v2 runs (15/11, mode H) into Process/runs/v2/<run>/ for the next chat / container.
Every result file under <run>/*_out/results/ that is small text is copied, the analysis files gzip'd:
  solver_logs.jsonl, submission.jsonl, links.jsonl, cards_*.jsonl, verdicts.jsonl, judge_verdicts.jsonl, raw_*.jsonl (gz);
  result.json, train_metrics.json, gpu_summary.json, summary*.json, *.txt (as is); eval.log / train.log: last 120 lines.
Never copies adapters, checkpoints, tokenizers, gpu_monitor.csv (adapters stay on Kaggle: Process/kaggle/kernels.csv).
Re-run is incremental (a file already archived is skipped).
Usage: archive_v2.py <scratch runs dir> <run>[,<run>...] [--dest Process/runs/v2]"""
import os, sys, glob, gzip, shutil
src, runs = sys.argv[1], sys.argv[2].split(",")
dest = sys.argv[sys.argv.index("--dest") + 1] if "--dest" in sys.argv else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Process", "runs", "v2")
GZ = ("solver_logs.jsonl", "submission.jsonl", "links.jsonl", "verdicts.jsonl", "judge_verdicts.jsonl")
KEEP = ("result.json", "train_metrics.json", "gpu_summary.json")
TAIL = ("eval.log", "train.log", "merge.log")
n = 0
for run in runs:
    for res_root in glob.glob(os.path.join(src, run, "*_out", "results")):
        for dp, dns, fns in os.walk(res_root):
            dns[:] = [d for d in dns if d not in ("adapter", "ckpt", "checkpoint") and not d.startswith(("model_", "checkpoint-"))]
            rel = os.path.relpath(dp, res_root)
            for f in fns:
                p = os.path.join(dp, f)
                if f in GZ or (f.startswith(("cards_", "raw_")) and f.endswith(".jsonl")) or f in ("units_teacher.jsonl", "rows.jsonl"):
                    out = os.path.join(dest, run, rel, f + ".gz")
                    if os.path.exists(out): continue
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    with open(p, "rb") as a, gzip.open(out, "wb", 9) as b: shutil.copyfileobj(a, b)
                elif f in KEEP or (f.startswith("summary") and f.endswith(".json")) or (f.endswith(".txt") and os.path.getsize(p) < 200000):
                    out = os.path.join(dest, run, rel, f)
                    if os.path.exists(out): continue
                    os.makedirs(os.path.dirname(out), exist_ok=True); shutil.copy(p, out)
                elif f in TAIL:
                    out = os.path.join(dest, run, rel, f + ".tail.txt")
                    if os.path.exists(out): continue
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    lines = open(p, encoding="utf-8", errors="replace").read().splitlines()
                    lines = [l for l in lines if "INFO" not in l and "WARNING" not in l][-120:]
                    open(out, "w", encoding="utf-8").write("\n".join(l[:400] for l in lines) + "\n")
                else: continue
                n += 1
print(f"archived {n} files of {len(runs)} runs into {dest}")
