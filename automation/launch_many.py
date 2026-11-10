#!/usr/bin/env python3
"""Launch several fanout.py runs: runs on the same account one after the other (each waits for the previous one's 'launched'
line: two fanouts at once on one account can overwrite each other's code dataset version), different accounts in parallel.
Each run is registered in <runs>/active_runs.txt for automation/watch_done.py.

Usage: launch_many.py <runs_dir> <plan.json>    plan = [{"acc": "quynhtran35", "name": "e0", "jobs": "automation/jobs/v35_e0.jobs",
                                                         "args": ["--ms", "...", "--ks", "...", "--ds", "..."], "est_h": 3.0}, ...]
12 h session limit (user 10/11): est_h > 11 h is split over the account's 2 kernels ("independent": true), run as a chain of
checkpointed sessions ("checkpointed": true) or refused; inside the kernel make_kernel.py's runner enforces the deadline.
GPU quota (user rule 10/11, automation/quota.py): before each run the account's quota is read; the run is skipped when the
account has < 6 h left and the task's estimate (est_h wall hours x quota.FACTOR; default 2.5 h per job line) does not fit, or when
the estimate exceeds what is left at all.
fanout.py is run from this file's repository (its code is what the kernels get)."""
import os, sys, json, time, subprocess, threading

ROOT = os.environ.get("ZC_LAUNCH_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # a frozen worktree may be given


MAX_WALL_H = 11.0          # a Kaggle commit session stops at 12 h and loses its outputs; ~1 h for setup + saving (user 10/11)


def kernels_for(it, n_jobs):
    """(number of kernels, reason) for a plan item under the 12 h session limit: estimate (est_h, wall hours of the whole job
    list run one after another) <= 11 h -> 1 kernel; longer with independent job lines ("independent": true) -> the lines
    split over 2 kernels (the account's 2 GPU sessions); a checkpointed training ("checkpointed": true: arms_job / full_job /
    sft / dpo stop before the deadline with a resumable checkpoint) -> 1 kernel, continued by a new kernel that mounts its
    output (--ks) with --resume-from auto; else 0 (refused: split the plan)."""
    import math
    est = float(it.get("est_h") or 0)
    if est <= MAX_WALL_H: return 1, f"~{est:.1f} h of the 12 h session" if est else "no estimate (12 h guard in the kernel)"
    n = math.ceil(est / MAX_WALL_H)
    if it.get("independent") and n_jobs >= n and n <= 2:
        return n, f"~{est:.1f} h > {MAX_WALL_H:.0f} h: job lines split over {n} kernels"
    if it.get("checkpointed"):
        return 1, (f"~{est:.1f} h > {MAX_WALL_H:.0f} h: part 1 stops at the session deadline with a checkpoint; continue it with a "
                   f"kernel mounting this one's output (--ks) and --resume-from auto ({n} sessions)")
    return 0, (f"NOT LAUNCHED: ~{est:.1f} h > {MAX_WALL_H:.0f} h per session (Kaggle stops at 12 h): split the jobs into several plan "
               f"items / accounts, or mark a checkpointed training \"checkpointed\": true")


def run_acc(runs, items, log):
    sys.path.insert(0, os.path.join(ROOT, "automation")); import quota
    reserved = 0.0                                                      # quota hours this launch already gave the account
    for it in items:
        name = it["name"]; fl = os.path.join(runs, name + ".fanout.log")
        jf = it["jobs"] if os.path.isabs(it["jobs"]) else os.path.join(ROOT, it["jobs"])
        n_jobs = len([l for l in open(jf).read().replace(";", "\n").splitlines() if l.strip()]) if os.path.isfile(jf) else 1
        nk, why = kernels_for(it, n_jobs)                               # Kaggle's 12 h session limit (user 10/11)
        log(f"{it['acc']} {name}: {why}")
        if nk == 0: continue
        args = list(it.get("args", []))
        if it.get("checkpointed") and float(it.get("est_h") or 0) > MAX_WALL_H:
            left = (quota.remaining(it["acc"]) or 0) - reserved          # this part: what the session limit and the quota allow
            part_h = min(MAX_WALL_H, left / quota.FACTOR - 0.5)          # (quota.FACTOR quota h per wall h on 4xL4)
            if part_h < 1: log(f"NOT LAUNCHED: {it['acc']} has {left:.1f} h GPU left: < 1 h for a checkpointed part"); continue
            need = part_h * quota.FACTOR; ok = True
            env = f"ZC_SESSION_S={int((part_h + 2400 / 3600) * 3600)}"   # the kernel's own deadline = this part's length
            if "--env" in args: args[args.index("--env") + 1] += "," + env
            else: args += ["--env", env]
            log(f"{it['acc']} {name}: checkpointed part of {part_h:.1f} h ({need:.1f} quota h of {left:.1f} left)")
        else:
            need = quota.need_hours(n_jobs, it.get("est_h"), l4=True)   # user rule 10/11: < 6 h left -> the task must fit
            ok, msg = quota.check(it["acc"], need, reserved)
            log(msg)
            if not ok and not it.get("force_quota"): continue
        reserved += need
        with open(os.path.join(runs, "active_runs.txt"), "a") as f: f.write(name + "\n")
        rep = os.path.join(runs, ".reported")                          # a run relaunched under an old name: forget the old kernel,
        if os.path.isfile(rep):                                        # else watch_done.py never reports the new one
            keep = [l for l in open(rep).read().split() if not l.startswith(name + "/")]
            open(rep, "w").write("\n".join(keep))
        cmd = [sys.executable, os.path.join(ROOT, "automation", "fanout.py"), os.path.join(runs, name), "zc-" + name.replace("_", "-"), it["jobs"],
               str(nk), "--accounts", it["acc"], "--l4"] + args
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
