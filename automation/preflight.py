#!/usr/bin/env python3
"""Pre-run gate (user rule, 2026-11-02): nothing is pushed to Kaggle before this passes.

Given the exact job string that will be passed to fanout.py ("run_kaggle.py <split> OUT/x <agent args>;..."):
 1. every agent flag is parsed by agent.build_parser() (typos / stale flags fail here, not after a 10-min setup);
 2. the kernel files are generated offline (make_kernel --no-upload): run.py must compile, and every OUT/... path must
    resolve inside /kaggle/working/results (the code directory is deleted at the end of a kernel);
 3. the CPU test suite passes (tests/run_all.sh);
 4. the exact agent command runs end-to-end on CPU with a fake model (--fake-responder, --subset N, same --dp /
    --pipeline / --budget flags): every question of the subset gets one answer line, logs and score are written.
Usage: preflight.py "<jobs>" [--subset 8] [--skip-tests]      exit code 0 = OK to launch"""
import os, sys, re, json, shlex, shutil, subprocess, tempfile, py_compile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
FAKE = '''import re
def responder(conv):
    sysmsg, user = conv[0]["content"], conv[-1]["content"]
    if sysmsg.startswith("You are DataMate's data engineer"):
        return "```python\\nraise ValueError('fake mapping')\\n```"
    return "```python\\nANSWER = {'type': 'number', 'answer': 0}\\n```"
'''


def fail(msg): print("PREFLIGHT FAIL:", msg, flush=True); sys.exit(1)


def main():
    argv = sys.argv[1:]
    subset = int(argv[argv.index("--subset") + 1]) if "--subset" in argv else 8
    jobs = [j.split() for j in argv[0].split(";") if j.strip()]
    import agent
    ap = agent.build_parser()
    work = tempfile.mkdtemp(prefix="preflight_")
    import atexit, signal                                               # removed on success, failure or kill (SIGTERM / Ctrl-C):
    atexit.register(shutil.rmtree, work, True)                          # stopped preflights once left 22 GB in /tmp
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    # 1. flags
    for j in jobs:
        if j[0] != "run_kaggle.py" or len(j) < 3: fail(f"job must be 'run_kaggle.py <split> OUT/... <args>': {j}")
        if "--script" in j: continue                                    # another src/ job: checked by its own smoke run below
        try: ap.parse_args(["SPLIT", "OUT"] + j[3:])
        except SystemExit: fail(f"agent rejects the flags of job {' '.join(j)}")
        if not (j[2] == "OUT" or j[2].startswith("OUT/")): fail(f"output dir must be OUT or OUT/...: {j[2]}")
    print(f"[1] flags OK for {len(jobs)} jobs", flush=True)
    # 2. kernel files offline + OUT substitution of the generated runner
    kd = os.path.join(work, "kernel")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "automation", "make_kernel.py"), kd, "zc-preflight", ";".join(" ".join(j) for j in jobs), "",
                        "--no-upload", "--l4"], capture_output=True, text=True)
    if r.returncode != 0: fail("make_kernel: " + r.stderr[-800:])
    run_py = os.path.join(kd, "run.py"); py_compile.compile(run_py, doraise=True)
    src = open(run_py).read()
    m = re.search(r"\n    def sub\(x\):.*?return x\n", src, re.S)
    if not m: fail("generated run.py has no prefix-aware token substitution")
    import textwrap
    g = {"SUB": {"DATA": "/kaggle/input/zc-2026", "INPUT": "/kaggle/input", "OUT": "/kaggle/working/results"}}
    exec(textwrap.dedent(m.group(0)), g)                               # the runner's own sub(), bound to SUB above
    sub = g["sub"]
    for j in jobs:
        if not sub(j[2]).startswith("/kaggle/working/results"): fail(f"{j[2]} -> {sub(j[2])} is not under /kaggle/working/results")
    meta = json.load(open(os.path.join(kd, "kernel-metadata.json")))
    print(f"[2] kernel files OK (run.py compiles, OUT paths -> {sub(jobs[0][2])}, machine {meta.get('machine_shape')})", flush=True)
    # 3. CPU test suite
    if "--skip-tests" not in argv:
        r = subprocess.run(["bash", os.path.join(ROOT, "tests", "run_all.sh")], capture_output=True, text=True)
        if r.returncode != 0: fail("CPU tests:\n" + (r.stdout + r.stderr)[-2500:])
        print("[3] CPU tests PASS", flush=True)
    # 4. end-to-end smoke of each job with a fake model
    rf = os.path.join(work, "fake_responder.py"); open(rf, "w").write(FAKE)
    for j in jobs:
        split = os.path.join(ROOT, "data", "raw", j[1], j[1])
        if not os.path.isfile(os.path.join(split, "questions.jsonl")): fail(f"local split {split} missing (download the data first)")
        out = os.path.join(work, "out_" + re.sub(r"\W", "_", j[2]))
        args = [x for x in j[3:]]
        if "--script" in args:                                          # e.g. qnorm_llm.py: runs on the whole split with a fake LLM
            i = args.index("--script"); script = args[i + 1]; del args[i:i + 2]
            if "--cache" in args: args[args.index("--cache") + 1] = os.path.join(work, "cache")
            ident = os.path.join(work, "ident.py"); open(ident, "w").write("def responder(conv):\n    return conv[-1]['content']\n")
            r = subprocess.run([sys.executable, os.path.join(ROOT, "src", script), split, out] + args + ["--fake-responder", ident],
                               capture_output=True, text=True, timeout=3600)
            if r.returncode != 0: fail(f"smoke of {script}:\n" + (r.stdout + r.stderr)[-3000:])
            print(f"[4] smoke {script} on {j[1]}: rc 0, outputs {sorted(os.listdir(out))}", flush=True); continue
        for flag in ("--cache",):                                       # local paths for the smoke
            if flag in args: args[args.index(flag) + 1] = os.path.join(work, "cache")
        if "--budget" in args: args[args.index("--budget") + 1] = "600"
        cmd = [sys.executable, os.path.join(ROOT, "src", "agent.py"), split, out] + args + ["--fake-responder", rf, "--subset", str(subset), "--workers", "4"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0: fail(f"smoke of {' '.join(j[:3])}:\n" + (r.stdout + r.stderr)[-3000:])
        qs = [json.loads(l) for l in open(os.path.join(split, "questions.jsonl"), encoding="utf-8")]
        n = len(agent.select_subset(qs, subset))
        subm = [json.loads(l) for l in open(os.path.join(out, "submission.jsonl"), encoding="utf-8")]
        logs = [json.loads(l) for l in open(os.path.join(out, "logs.jsonl"), encoding="utf-8")]
        if len(subm) != n or len({x["id"] for x in subm}) != n or len(logs) != n:
            fail(f"smoke of {j[2]}: {len(subm)} answers / {len(logs)} logs for {n} questions")
        print(f"[4] smoke {j[1]} {j[2]}: {n} questions answered once, logs complete", flush=True)
    shutil.rmtree(work, ignore_errors=True)
    print("PREFLIGHT OK", flush=True)


if __name__ == "__main__":
    main()
