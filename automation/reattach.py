#!/usr/bin/env python3
"""Re-attach monitors to Kaggle kernels that keep running after the local runners died (container restart, network
error), pull their outputs and copy results/*.json into results/.
Usage: reattach.py [--skip <regex>] <kernel_dir> [...]
  kernel_dir : <work>/<slug> as made by fanout.py (outputs go to <work>/<slug>_out, log to <work>/<slug>.attach.log)
  --skip     : result files whose name matches are NOT copied (e.g. results superseded by a later run)."""
import sys, os, re, glob, shutil, subprocess, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
args = sys.argv[1:]; skip = None
if "--skip" in args:
    i = args.index("--skip"); skip = re.compile(args[i + 1]); del args[i:i + 2]
procs = {}
for kd in args:
    kd = kd.rstrip("/")
    procs[kd] = subprocess.Popen([sys.executable, os.path.join(ROOT, "automation", "run_kernel.py"), kd, kd + "_out", "600", "4", "43200",
                                  "--attach"], stdout=open(kd + ".attach.log", "w"), stderr=subprocess.STDOUT)
    time.sleep(2)
print(f"attached {len(procs)} kernels", flush=True)
while procs:
    for kd, p in list(procs.items()):
        if p.poll() is None: continue
        del procs[kd]; log = open(kd + ".attach.log").read()
        res = [f for f in glob.glob(os.path.join(kd + "_out", "results", "*.json")) if not (skip and skip.search(os.path.basename(f)))]
        for f in res: shutil.copy(f, os.path.join(ROOT, "results"))
        print(f"{os.path.basename(kd)}: {'OK' if 'STATUS=DONE:complete' in log else 'FAILED'} {len(res)} results", flush=True)
    time.sleep(15)
