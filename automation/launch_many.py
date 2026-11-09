#!/usr/bin/env python3
"""Launch several fanout.py runs: runs on the same account one after the other (each waits for the previous one's 'launched'
line: two fanouts at once on one account can overwrite each other's code dataset version), different accounts in parallel.
Each run is registered in <runs>/active_runs.txt for automation/watch_done.py.

Usage: launch_many.py <runs_dir> <plan.json>    plan = [{"acc": "quynhtran35", "name": "e0", "jobs": "automation/jobs/v35_e0.jobs",
                                                         "args": ["--ms", "...", "--ks", "...", "--ds", "..."]}, ...]
fanout.py is run from this file's repository (its code is what the kernels get)."""
import os, sys, json, time, subprocess, threading

ROOT = os.environ.get("ZC_LAUNCH_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # a frozen worktree may be given


def run_acc(runs, items, log):
    for it in items:
        name = it["name"]; fl = os.path.join(runs, name + ".fanout.log")
        with open(os.path.join(runs, "active_runs.txt"), "a") as f: f.write(name + "\n")
        rep = os.path.join(runs, ".reported")                          # a run relaunched under an old name: forget the old kernel,
        if os.path.isfile(rep):                                        # else watch_done.py never reports the new one
            keep = [l for l in open(rep).read().split() if not l.startswith(name + "/")]
            open(rep, "w").write("\n".join(keep))
        cmd = [sys.executable, os.path.join(ROOT, "automation", "fanout.py"), os.path.join(runs, name), "zc-" + name.replace("_", "-"), it["jobs"],
               "1", "--accounts", it["acc"], "--l4"] + it.get("args", [])
        p = subprocess.Popen(cmd, stdout=open(fl, "w"), stderr=subprocess.STDOUT, cwd=ROOT, start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < 900:
            txt = open(fl, errors="ignore").read()
            if "launched" in txt or "NOT LAUNCHED" in txt or p.poll() is not None: break
            time.sleep(5)
        log(f"{it['acc']} {name}: " + (open(fl, errors="ignore").read().strip().splitlines() or ["?"])[-1][:160])


def main():
    runs, plan = sys.argv[1], json.load(open(sys.argv[2]))
    os.makedirs(runs, exist_ok=True)
    by = {}
    for it in plan: by.setdefault(it["acc"], []).append(it)
    lock = threading.Lock()
    def log(s):
        with lock: print(time.strftime("%H:%M:%S"), s, flush=True)
    ts = [threading.Thread(target=run_acc, args=(runs, items, log)) for items in by.values()]
    for t in ts: t.start()
    for t in ts: t.join()
    log("all launched")


if __name__ == "__main__":
    main()
