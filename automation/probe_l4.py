#!/usr/bin/env python3
"""Probe which Kaggle accounts can start a 4xL4 session (gemma-4-developer-agent grant accepted, GPU quota left).

For every account with a token (~/.kaggle/tok_<user>), push a one-line kernel `zc-probe-l4` (machine NvidiaL4, no
datasets), wait until it finishes, read the GPU names it printed, then delete it. Costs ~1-2 min of one session.
Usage: probe_l4.py <work_dir> [acc,acc,...]
Prints one line per account: OK <gpus> | PUSH_FAILED <msg> | ERROR <status>; writes <work_dir>/probe.json."""
import os, sys, json, glob, time, subprocess
from concurrent.futures import ThreadPoolExecutor

work = sys.argv[1]
accs = sys.argv[2].split(",") if len(sys.argv) > 2 else sorted(os.path.basename(f)[4:] for f in glob.glob(os.path.expanduser("~/.kaggle/tok_*")))
CODE = "import subprocess\nprint('GPUS', subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout.replace(chr(10), ' | '), flush=True)\n"


def kg(acc, *a, timeout=120):
    env = dict(os.environ, KAGGLE_API_TOKEN=open(os.path.expanduser(f"~/.kaggle/tok_{acc}")).read().strip())
    try: r = subprocess.run(["kaggle"] + list(a), capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired: return "TIMEOUT"
    return (r.stdout + r.stderr).strip()


def probe(acc):
    kd = os.path.join(work, acc); os.makedirs(kd, exist_ok=True)
    open(os.path.join(kd, "run.py"), "w").write(CODE)
    json.dump({"id": f"{acc}/zc-probe-l4", "title": "zc-probe-l4", "code_file": "run.py", "language": "python", "kernel_type": "script",
               "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "dataset_sources": [],
               "competition_sources": ["gemma-4-developer-agent"], "kernel_sources": [], "model_sources": [], "machine_shape": "NvidiaL4"},
              open(os.path.join(kd, "kernel-metadata.json"), "w"))
    o = kg(acc, "kernels", "push", "-p", kd)
    if "successfully pushed" not in o.lower() and "kernel version" not in o.lower(): return acc, "PUSH_FAILED", o.splitlines()[-1][:200] if o else ""
    t0 = time.time(); st = ""
    while time.time() - t0 < 900:
        time.sleep(20)
        s = kg(acc, "kernels", "status", f"{acc}/zc-probe-l4")
        if "KernelWorkerStatus." in s: st = s.split("KernelWorkerStatus.")[1].split('"')[0]
        if st in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"): break
    od = os.path.join(kd, "out"); kg(acc, "kernels", "output", f"{acc}/zc-probe-l4", "-p", od, "-o")
    gpus = ""
    for f in glob.glob(os.path.join(od, "*.log")):
        try: gpus = " ".join(r.get("data", "") for r in json.load(open(f, errors="ignore")) if "GPUS" in r.get("data", ""))
        except Exception: pass
    if st not in ("COMPLETE", "ERROR"): kg(acc, "kernels", "delete", "-y", f"{acc}/zc-probe-l4")
    return acc, ("OK" if "L4" in gpus else st or "TIMEOUT"), gpus.strip()[:200]


os.makedirs(work, exist_ok=True)
with ThreadPoolExecutor(12) as ex: res = list(ex.map(probe, accs))
for acc, st, msg in res: print(f"{acc:16s} {st:12s} {msg}", flush=True)
json.dump({a: {"status": s, "msg": m} for a, s, m in res}, open(os.path.join(work, "probe.json"), "w"), indent=1)
