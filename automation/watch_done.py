#!/usr/bin/env python3
"""Block until the next Kaggle kernel of the watched runs finishes, print one summary line per newly finished kernel
and exit, so Claude is notified the moment a kernel ends instead of polling (user rule: no active checking).

Run it with Bash `run_in_background` (one notification when it exits); after handling the result, start it again.
A kernel counts as finished when its runner log (<slug>.log or <slug>.attach.log, written by run_kernel.py) holds
STATUS=DONE / STATUS=PUSH_FAILED / STATUS=STALL_GIVEUP / STATUS=HARD_TIMEOUT. Already reported kernels are kept in
<runs>/.reported. Exits with "WATCH_TIMEOUT ..." after --max minutes (Claude background tasks die after ~2 h) and
"ALL_DONE" when nothing is left to wait for.
Usage: watch_done.py <runs_dir> [--max 100]"""
import sys, os, glob, time, re

runs = sys.argv[1]
MAX = float(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else 100
seen_f = os.path.join(runs, ".reported")
seen = set(open(seen_f).read().split()) if os.path.isfile(seen_f) else set()
END = re.compile(r"STATUS=(DONE:\w+|PUSH_FAILED|STALL_GIVEUP|HARD_TIMEOUT)")


def active():
    f = os.path.join(runs, "active_runs.txt")
    return [l.strip() for l in open(f) if l.strip()] if os.path.isfile(f) else []


t0 = time.time()
while True:
    new, pending = [], 0
    for r in active():
        for meta in glob.glob(os.path.join(runs, r, "*", "kernel-metadata.json")):
            kd = os.path.dirname(meta); slug = os.path.basename(kd)
            logs = [l for l in (kd + ".log", kd + ".attach.log") if os.path.isfile(l)]
            m = next(filter(None, (END.search(open(l, errors="ignore").read()) for l in logs)), None)
            if m is None: pending += 1; continue
            if f"{r}/{slug}" in seen: continue
            res = glob.glob(os.path.join(kd + "_out", "results", "*"))
            new.append(f"{r}/{slug}: {m.group(1)} ({len(res)} result files)"); seen.add(f"{r}/{slug}")
    if new:
        open(seen_f, "w").write("\n".join(sorted(seen)))
        print("\n".join(new) + f"\n[{pending} kernels still running]", flush=True); sys.exit(0)
    if pending == 0:
        print("ALL_DONE (no kernel left to wait for)", flush=True); sys.exit(0)
    if time.time() - t0 > MAX * 60:
        print(f"WATCH_TIMEOUT after {MAX:.0f} min, {pending} kernels still running -> re-arm", flush=True); sys.exit(0)
    time.sleep(20)
