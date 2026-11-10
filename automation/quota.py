#!/usr/bin/env python3
"""Kaggle GPU quota of the accounts (`kaggle quota`: used / remaining / refresh time of the weekly 30 h) and the launch rule
(user 10/11): an account with less than 6 h of GPU left gets a new task only after checking that the task fits in what is left.

Quota hours = wall hours x GPUs of the session: a 4xL4 session costs ~4 quota hours per wall hour (measured 10/11: thnliu
1.69 h after ~21 min of one 4xL4 kernel; phuongnehe 4.13 h after ~33 min of two). A session that started is not stopped
when the quota runs out (accounts at 31-36 h of 30), but a new one must not be started on the hope of that.

    remaining(acc) -> hours left (None when the query fails)
    need_hours(n_jobs, est_wall_h=None, l4=True) -> quota hours a kernel needs (est given, else 2.5 wall h per job line)
    check(acc, need, reserved=0) -> (ok, message)   ok = the need fits in what is left (an unknown quota refuses)
Usage: quota.py [acc,...]   (default: every account with a token) -> one line per account."""
import os, sys, json, glob, subprocess

LOW = 6.0                       # below this, a task is launched only when its estimate fits (user rule 10/11)
PER_JOB_WALL_H = 2.5            # default wall time of one job line when the plan gives no estimate (eval of pub_full + held-out
                                # ~1.3 h, a training job 1.5-4 h): deliberately on the high side


def accounts():
    return sorted(os.path.basename(f)[4:] for f in glob.glob(os.path.expanduser("~/.kaggle/tok_*")))


def info(acc):
    tok = os.path.expanduser(f"~/.kaggle/tok_{acc}")
    if not os.path.isfile(tok): return None
    env = dict(os.environ, KAGGLE_API_TOKEN=open(tok).read().strip())
    try:
        r = subprocess.run(["kaggle", "quota", "--format", "json"], capture_output=True, text=True, env=env, timeout=60)
        d = json.loads(r.stdout)
    except Exception:
        return None
    rows = d if isinstance(d, list) else d.get("quotas", [])
    g = next((x for x in rows if str(x.get("resource", "")).upper() == "GPU"), None)
    if not g: return None
    h = lambda v: float(str(v).rstrip("h") or 0)
    return {"used": h(g.get("used")), "remaining": h(g.get("remaining")), "total": h(g.get("total")), "refresh": g.get("refreshAt")}


def remaining(acc):
    i = info(acc)
    return None if i is None else i["remaining"]


def need_hours(n_jobs, est_wall_h=None, l4=True):
    wall = float(est_wall_h) if est_wall_h else PER_JOB_WALL_H * max(1, int(n_jobs))
    return wall * (4 if l4 else 2)


def check(acc, need, reserved=0.0):
    """(ok, message): the account may take a task of `need` quota hours (minus what this launch already gave it)."""
    left = remaining(acc)
    if left is None: return False, f"{acc}: GPU quota unknown (kaggle quota failed): not launched"
    left -= reserved
    if left < LOW and need > left:
        return False, f"{acc}: {left:.1f} h GPU left (< {LOW:.0f} h) but the task needs ~{need:.1f} h: not launched"
    if need > left:
        return False, f"{acc}: {left:.1f} h GPU left, the task needs ~{need:.1f} h: not launched"
    return True, f"{acc}: {left:.1f} h GPU left, task ~{need:.1f} h: ok"


def main():
    accs = sys.argv[1].split(",") if len(sys.argv) > 1 else accounts()
    for a in accs:
        i = info(a)
        print(f"{a:16s} " + ("quota unknown" if i is None else f"used {i['used']:5.2f} h  left {i['remaining']:5.2f} h  refresh {i['refresh']}"
                                + ("  LOW" if i["remaining"] < LOW else "")), flush=True)


if __name__ == "__main__":
    main()
