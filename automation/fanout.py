#!/usr/bin/env python3
"""Spread evaluation jobs over every Kaggle account's GPU slots (2 T4 sessions per account) and run them all.
Tokens: ~/.kaggle/tok_<username> (one file per account). Jobs are split round-robin into `n_kernels` kernels, each
kernel goes to the next free (account, slot); kernels beyond 2 x #accounts become CPU kernels (use only for light jobs).
Usage: fanout.py <work_dir> <slug_prefix> <jobs_file|"job;job;..."> [n_kernels=all GPU slots] [--accounts a,b,...] [--cpu|--l4]
Each kernel: <work_dir>/<slug>/ (kernel dir), <work_dir>/<slug>_out/ (outputs), <work_dir>/<slug>.log (runner log).
Prints one line per kernel; waits for all runners; then copies every results/*.json into results/."""
import sys, os, glob, subprocess, shutil, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACCOUNTS = sorted(os.path.basename(f)[4:] for f in glob.glob(os.path.expanduser("~/.kaggle/tok_*")))
ACCOUNTS = [a for a in ACCOUNTS if a != "nhatlong1103"]                  # nhatlong1103 only publishes datasets
DSARG = ["--ds", sys.argv[sys.argv.index("--ds") + 1]] if "--ds" in sys.argv else []   # extra dataset sources for every kernel
DSARG += (["--ks", sys.argv[sys.argv.index("--ks") + 1]] if "--ks" in sys.argv else []) + (["--internet"] if "--internet" in sys.argv else [])
DSARG += ["--env", sys.argv[sys.argv.index("--env") + 1]] if "--env" in sys.argv else []
DSARG += ["--ms", sys.argv[sys.argv.index("--ms") + 1]] if "--ms" in sys.argv else []
L4 = "--l4" in sys.argv                                                   # 4x L4 sessions (competition grant)
args = [a for a in sys.argv[1:] if not a.startswith("--") and a not in DSARG]
if "--accounts" in sys.argv: ACCOUNTS = sys.argv[sys.argv.index("--accounts") + 1].split(","); args = [a for a in args if a != ",".join(ACCOUNTS)]
work, prefix, jobs = args[0], args[1], args[2]
jobs = [j.strip() for j in (open(jobs).read().replace("\n", ";") if os.path.isfile(jobs) else jobs).split(";") if j.strip()]

# pre-run gate (user rule 2026-11-02): no launch before automation/preflight.py passes for this exact code + job string
import hashlib
h = hashlib.sha256(";".join(jobs).encode())
for f in sorted(f for f in glob.glob(os.path.join(ROOT, "src", "**", "*"), recursive=True) + glob.glob(os.path.join(ROOT, "automation", "*.py")) +
                glob.glob(os.path.join(ROOT, "tests", "*")) if os.path.isfile(f) and "__pycache__" not in f):
    h.update(f.encode()); h.update(open(f, "rb").read())
mark = os.path.expanduser(f"~/.cache/zc_preflight/{h.hexdigest()[:16]}")
if "--no-preflight" in sys.argv: print("preflight: SKIPPED (--no-preflight)", flush=True)
elif os.path.isfile(mark): print("preflight: already passed for this code + jobs", flush=True)
else:
    pf = [j for j in jobs if j.startswith("run_kaggle.py")]
    if pf:
        print("preflight: running (flags, kernel files, CPU tests, CPU smoke) ...", flush=True)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "automation", "preflight.py"), ";".join(pf)], capture_output=True, text=True)
        print(r.stdout[-1500:], flush=True)
        if r.returncode != 0: print("NOT LAUNCHED: preflight failed", flush=True); sys.exit(1)
    os.makedirs(os.path.dirname(mark), exist_ok=True); open(mark, "w").write("\n".join(jobs))
slots = [(a, s) for a in ACCOUNTS for s in (0, 1)]                       # fill BOTH GPU sessions of an account before the next (user rule: do not spend quota of extra accounts)
nk = min(int(args[3]) if len(args) > 3 else len(slots), len(jobs))
os.makedirs(work, exist_ok=True); procs = []
for k in range(nk):
    part = jobs[k::nk]; acc, _ = slots[k % len(slots)]; gpu = k < len(slots) and "--cpu" not in sys.argv
    slug = f"{prefix}-{k + 1}"; kd = os.path.join(work, slug)
    cmd = [sys.executable, os.path.join(ROOT, "automation", "make_kernel.py"), kd, slug, ";".join(part), "", "--owner", acc] + ((["--l4"] if L4 else ["--gpu"]) if gpu else []) + DSARG
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f"{slug}: {acc} {'GPU' if gpu else 'CPU'} {len(part)} jobs | {(r.stdout + r.stderr).strip().splitlines()[-1][:60]}", flush=True)
time.sleep(30)                                                             # let the code datasets finish versioning
for acc in sorted({slots[k % len(slots)][0] for k in range(nk)}):
    env = dict(os.environ, KAGGLE_API_TOKEN=open(os.path.expanduser(f"~/.kaggle/tok_{acc}")).read().strip())
    for _ in range(40):
        if subprocess.run(["kaggle", "datasets", "status", f"{acc}/zc-code"], capture_output=True, text=True, env=env).stdout.strip().endswith("ready"): break
        time.sleep(10)
def make(kd, slug, part, acc, gpu):
    cmd = [sys.executable, os.path.join(ROOT, "automation", "make_kernel.py"), kd, slug, ";".join(part), "", "--owner", acc] + ((["--l4"] if L4 else ["--gpu"]) if gpu else []) + DSARG
    return subprocess.run(cmd, capture_output=True, text=True)


def launch(k):
    slug = f"{prefix}-{k + 1}"
    return subprocess.Popen([sys.executable, os.path.join(ROOT, "automation", "run_kernel.py"), os.path.join(work, slug),
                             os.path.join(work, slug + "_out"), "600" if not L4 else "480", "5", "43200"], stdout=open(os.path.join(work, slug + ".log"), "w"), stderr=subprocess.STDOUT)


procs = {}
for k in range(nk):
    procs[k] = launch(k); time.sleep(2)
print(f"launched {nk} kernels", flush=True)
# failover: a kernel whose push hit the account's weekly GPU quota or its 2-session cap moves to the next account: first the
# --accounts list, then every other account with a token (03/11: phuongnehe and nguynduylm had used their 30 h; the old
# failover only tried the listed accounts). Accounts found out of quota are remembered for the ISO week (~/.cache/zc_quota.json).
import json, datetime
QF = os.path.expanduser("~/.cache/zc_quota.json"); WEEK = datetime.date.today().strftime("%G-W%V")
def quota_out():
    try: return {a for a, w in json.load(open(QF)).items() if w == WEEK}
    except Exception: return set()
def mark_quota(acc):
    try: q = json.load(open(QF))
    except Exception: q = {}
    q[acc] = WEEK; os.makedirs(os.path.dirname(QF), exist_ok=True); json.dump(q, open(QF, "w"))
ALL_ACC = [a for a in ACCOUNTS] + [a for a in sorted(os.path.basename(f)[4:] for f in glob.glob(os.path.expanduser("~/.kaggle/tok_*")))
                                   if a not in ACCOUNTS and a != "nhatlong1103"]
tried = {k: {slots[k % len(slots)][0]} for k in range(nk)}
busy = {}                                                                   # account -> kernels this fanout moved there
while procs:
    for k, p in list(procs.items()):
        if p.poll() is None: continue
        del procs[k]; slug = f"{prefix}-{k + 1}"; log = open(os.path.join(work, slug + ".log")).read()
        gpu = k < len(slots) and "--cpu" not in sys.argv
        if "weekly GPU quota" in log: mark_quota(json.load(open(os.path.join(work, slug, "kernel-metadata.json")))["id"].split("/")[0])   # the account it was pushed to
        cand = [a for a in ALL_ACC if a not in tried[k] and a not in quota_out() and busy.get(a, 0) < 2]
        if "STATUS=PUSH_FAILED" in log and cand:
            acc = cand[0]; tried[k].add(acc); busy[acc] = busy.get(acc, 0) + 1
            make(os.path.join(work, slug), slug, jobs[k::nk], acc, gpu); time.sleep(40)
            print(f"{slug}: push failed -> retry on {acc}", flush=True); procs[k] = launch(k); continue
        res = glob.glob(os.path.join(work, slug + "_out", "results", "**", "submission.jsonl"), recursive=True)   # outputs stay in <work>
        print(f"{slug}: {'OK' if 'STATUS=DONE:complete' in log else 'FAILED'} {len(res)} results, rc!=0: {log.count('rc=') - log.count('rc=0')}", flush=True)
    time.sleep(15)
