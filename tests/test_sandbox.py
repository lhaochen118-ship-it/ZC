#!/usr/bin/env python3
"""Sandbox time-outs and the robust pool map (the Kaggle crash of 02/11: a stray _Timeout escaped a worker and
pool.imap_unordered re-raised it in the parent, aborting two of four GPU shards).
Checks: `except Exception` in a generated program cannot swallow the time-out; a SIGALRM after disarm() is ignored;
pmap() turns a crashing task into an error result for that key and keeps the others."""
import os, sys, time, multiprocessing as mp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import agent

# 1. a busy loop that swallows exceptions still times out
t = time.time()
try:
    agent._exec("while True:\n    try:\n        x = sum(range(1000))\n    except Exception:\n        pass\n", {}, 1); ok = False
except agent._Timeout: ok = True
assert ok and time.time() - t < 3, "the time-out was swallowed"
# 2. a late alarm (armed, disarmed, signal arrives anyway) is ignored
import signal
agent.arm(1); agent.disarm(); signal.setitimer(signal.ITIMER_REAL, 0.2); time.sleep(0.5)   # raw SIGALRM, nothing armed
# 3. pmap: a task that raises yields an error result for its key, the others are kept; a worker killed by a
#    BaseException (its result never arrives) is reported when the deadline passes instead of hanging the stage


def work(task):
    key, x = task
    if x == 3: raise RuntimeError("boom")
    if x == 4: raise SystemExit(1)                                     # kills the worker process: result lost
    return key, {"ok": True, "v": x * x}


pool = mp.get_context("fork").Pool(2)
t = time.time(); res = dict(agent.pmap(pool, work, [(f"k{i}", i) for i in range(6)], deadline=4))
pool.terminate()
assert res["k3"]["ok"] is False and "RuntimeError" in res["k3"]["error"], res["k3"]
assert res["k4"]["ok"] is False and "worker failed" in res["k4"]["error"] and time.time() - t < 15, (res["k4"], time.time() - t)
assert res["k5"]["v"] == 25 and len(res) == 6, res
print("PASS")
