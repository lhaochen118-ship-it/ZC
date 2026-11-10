#!/usr/bin/env python3
"""Kaggle's 12 h commit-session limit (user 10/11), checked offline on the kernel runner automation/make_kernel.py writes:
the runner is built for real (--no-upload), its /kaggle paths moved to a scratch folder and its session shortened to 40 s
(ZC_SESSION_S, margin 25 s -> deadline 15 s after the start); sleeping jobs then show that
  - every job gets ZC_DEADLINE;
  - a sequential job that would end after the deadline (as long as the longest job so far) is not started and is listed
    in results/_skipped_jobs.json;
  - a job still running past the deadline is killed (the session ends cleanly, the finished outputs are kept);
and the trainers' deadline helper (training/deadline.py) finds the newest checkpoint.
Usage: test_session_limit.py [scratch]"""
import os, sys, re, json, shutil, subprocess, tempfile, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def runner(T, jobs, env):
    kd = os.path.join(T, "k"); shutil.rmtree(kd, ignore_errors=True)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "automation", "make_kernel.py"), kd, "zc-t12", "run_kaggle.py x OUT/y", "",
                        "--owner", "nobody", "--no-upload"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    src = open(os.path.join(kd, "run.py")).read().replace("/kaggle/working", f"{T}/working").replace("/kaggle/input", f"{T}/input")
    src = re.sub(r'if _tok != "[^"]+":', 'if _tok != "TOKEN":', src)
    src = re.sub(r"^JOBS = .*$", "JOBS = " + json.dumps(jobs), src, flags=re.M)
    src = re.sub(r"^ENV = .*$", "ENV = " + json.dumps(env), src, flags=re.M)
    src = src.replace("MULTI = NGPU and", "MULTI = True or NGPU and")    # sequential, as GPU jobs run
    p = os.path.join(T, "run_test.py"); open(p, "w").write(src)
    shutil.rmtree(os.path.join(T, "working"), ignore_errors=True); os.makedirs(os.path.join(T, "working"))
    t = time.time(); out = subprocess.run([sys.executable, p], capture_output=True, text=True, timeout=300)
    return out.stdout + out.stderr, time.time() - t


def main():
    T = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
    os.makedirs(os.path.join(T, "input", "code"), exist_ok=True)
    open(os.path.join(T, "input", "code", "ZC_CODE_MARKER"), "w").write("TOKEN")
    open(os.path.join(T, "input", "code", "sl.py"), "w").write(
        "import sys, time, os\nopen(os.path.join(os.environ['ZC_OUT'], 'dl_' + sys.argv[1]), 'w').write(os.environ.get('ZC_DEADLINE', ''))\n"
        "time.sleep(float(sys.argv[1]))\n")
    env = {"ZC_SESSION_S": "40", "ZC_SESSION_MARGIN": "25"}
    out, dt = runner(T, [["sl.py", "6"], ["sl.py", "6.1"], ["sl.py", "6.2"]], env)
    R = os.path.join(T, "working", "results")
    assert out.count("done sl.py") == 2 and "SKIPPED (12 h session limit" in out, out
    assert json.load(open(os.path.join(R, "_skipped_jobs.json"))) == ["sl.py 6.2"], out
    assert open(os.path.join(R, "dl_6")).read().isdigit(), "ZC_DEADLINE not passed"
    out, dt = runner(T, [["sl.py", "200"], ["sl.py", "1"]], env)
    assert "rc=killed_at_deadline" in out and dt < 120, (out, dt)
    assert json.load(open(os.path.join(R, "_skipped_jobs.json"))) == ["sl.py 1"], out
    sys.path.insert(0, os.path.join(ROOT, "training")); import deadline
    ck = os.path.join(T, "ck"); [os.makedirs(os.path.join(ck, f"checkpoint-{n}"), exist_ok=True) for n in (5, 12, 9)]
    for n in (5, 12, 9): open(os.path.join(ck, f"checkpoint-{n}", "trainer_state.json"), "w").write("{}")
    assert deadline.latest_checkpoint(ck).endswith("checkpoint-12"), deadline.latest_checkpoint(ck)
    assert deadline.latest_checkpoint(os.path.join(T, "none")) is None
    os.environ["ZC_DEADLINE"] = str(time.time() + 100); assert 90 < deadline.left_s() <= 100
    print("test_session_limit PASS")


if __name__ == "__main__":
    main()
