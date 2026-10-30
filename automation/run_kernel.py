#!/usr/bin/env python3
"""Push a Kaggle kernel, wait for it, cancel+restart if it is stuck in the queue, pull + decode output.

Stall policy (user): a session with no output for ~10 min is queue-stuck -> cancel it for real
(`kaggle kernels delete -y`, the only verified cancel) and push a fresh session; at most MAX_RESTARTS times.
Kaggle's output API only exposes a RUNNING kernel's log near the end, so "no output" is judged by the kernel
STATUS: QUEUED for > STALL secs => stuck. A RUNNING kernel is NOT restarted for being silent by default
(silent_max=0): the output API hides logs until the end, and a silent-restart rule killed two healthy 60-min runs.

Usage: run_kernel.py <kernel_dir> <out_dir> [stall=600] [max_restarts=4] [hard=10800] [silent_max=0] [--attach]
  --attach : do not push; wait for an already-running kernel (e.g. after a local restart), then pull.
  --logs-only : pull only the kernel log, not the outputs (kernels whose outputs are large reusable assets).
Last line: STATUS=DONE:<complete|error|stall|timeout>
"""
import sys, subprocess, time, os, json, glob

kdir, outdir = sys.argv[1], sys.argv[2]
STALL = int(sys.argv[3]) if len(sys.argv) > 3 else 600
MAXR = int(sys.argv[4]) if len(sys.argv) > 4 else 4
HARD = int(sys.argv[5]) if len(sys.argv) > 5 else 10800
SILENT_MAX = int(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6].isdigit() else 0
ATTACH = "--attach" in sys.argv
LOGS_ONLY = "--logs-only" in sys.argv
kid = json.load(open(os.path.join(kdir, "kernel-metadata.json")))["id"]
_tok = os.path.expanduser(f"~/.kaggle/tok_{kid.split('/')[0]}")       # per-account token file (see ENVIRONMENT_SUMMARY)
if os.path.isfile(_tok): os.environ["KAGGLE_API_TOKEN"] = open(_tok).read().strip()


def kg(*a):
    r = subprocess.run(["kaggle"] + list(a), capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def ts():
    return time.strftime("%H:%M:%S")


def stdout_len(tmp):
    os.makedirs(tmp, exist_ok=True)
    subprocess.run(["kaggle", "kernels", "output", kid, "-p", tmp, "-o", "--file-pattern", r".*\.log$"],
                   capture_output=True, text=True)
    n = 0
    for f in glob.glob(tmp + "/*.log"):
        try:
            n += sum(len(r.get("data", "")) for r in json.load(open(f, errors="ignore")) if r.get("stream_name") == "stdout")
        except Exception:
            pass
    return n


def push():
    o = kg("kernels", "push", "-p", kdir)
    print(ts(), "PUSH:", (o.splitlines()[-1] if o else "")[:160], flush=True)
    return "successfully pushed" in o.lower() or "kernel version" in o.lower()


def cancel():
    o = kg("kernels", "delete", "-y", kid)
    print(ts(), "CANCEL(delete):", (o.splitlines()[-1] if o else "")[:120], flush=True)
    time.sleep(20)


t0 = time.time()
if not ATTACH and not push():
    print("STATUS=PUSH_FAILED", flush=True); sys.exit(1)
attempt_t0 = time.time(); restarts = 0; last = None; state = ""; running_since = None; saw_out = False
time.sleep(15)
while True:
    now = time.time()
    st = kg("kernels", "status", kid); low = st.lower()
    if low != last:
        print(ts(), f"[{int(now - t0)}s]", st[:140], flush=True); last = low
    if "complete" in low or "error" in low or "cancel" in low:
        state = "complete" if "complete" in low else ("error" if "error" in low else "cancel"); break
    queued = "queued" in low
    if "running" in low and running_since is None:
        running_since = now; print(ts(), "kernel RUNNING (GPU granted)", flush=True)
    stuck = False
    if queued and now - attempt_t0 > STALL:
        stuck = True; why = f"QUEUED {int(now - attempt_t0)}s"
    elif SILENT_MAX > 0 and running_since and not saw_out and now - running_since > SILENT_MAX:
        if stdout_len(outdir + "_partial") > 0: saw_out = True
        else: stuck = True; why = f"RUNNING silent {int(now - running_since)}s"
    if stuck:
        if restarts >= MAXR:
            print("STATUS=STALL_GIVEUP", why, flush=True); state = "stall"; break
        restarts += 1; print(ts(), f"STUCK ({why}) -> cancel + restart {restarts}/{MAXR}", flush=True)
        cancel(); push(); attempt_t0 = time.time(); running_since = None; saw_out = False; time.sleep(15); continue
    if now - t0 > HARD:
        print("STATUS=HARD_TIMEOUT", flush=True); state = "timeout"; break
    time.sleep(30)

os.makedirs(outdir, exist_ok=True)
print(f"=== PULL ({state}) -> {outdir} ===", flush=True)
print(kg("kernels", "output", kid, "-p", outdir, "-o", *(["--file-pattern", r".*\.log$"] if LOGS_ONLY else []))[-400:], flush=True)
for f in sorted(glob.glob(outdir + "/*.log")):
    try:
        recs = json.load(open(f, errors="ignore"))
        out = "".join(r.get("data", "") for r in recs if r.get("stream_name") == "stdout")
        err = "".join(r.get("data", "") for r in recs if r.get("stream_name") == "stderr")
        print("========== STDOUT ==========\n" + out[-20000:], flush=True)
        if state != "complete": print("========== STDERR (tail) ==========\n" + err[-4000:], flush=True)
    except Exception as e:
        print("decode failed:", e, flush=True)
print(f"elapsed {int(time.time() - t0)}s, restarts {restarts}", flush=True)
print(f"STATUS=DONE:{state}", flush=True)
