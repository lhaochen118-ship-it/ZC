#!/usr/bin/env python3
"""The submitted system end to end (docker/predict.sh runs it; the Kaggle validation runs it through src/stress_flow.py).

Stage 1, rules (CPU, minutes, no model): src/rules_first.py. Every answer it gives is final. The submission is complete
  after this stage (questions it leaves are written as abstain).
Stage 1.5 (one GPU, minutes; --canon 1): the model restates the questions the rules left in the training questions' style
  (src/canon_llm.py); the rules answer the restatements they read (final, after the stage-1 answers).
Stage 2, model (GPU), only for the questions the rules leave: agent.py (Qwen3.5-9B, the only model) on a split holding
  just those questions; mapping = the stage-1 mapper when it runs without a heavy warning (--prefer-map fmt:automap),
  LLM mapping samples otherwise; independent runs voted (src/vote.py) while the time allows. The submission is
  re-merged (rules first) every 20 s while the model runs, so a time-out keeps every answer already decided.

Usage: predict_flow.py <split> <result_dir> [--model DIR] [--budget S] [--runs N] [--work DIR] [--llm 0|1] [--dp N]
                       [--rules-workers N] [-- <extra agent.py flags>]
Writes <result_dir>/submission.jsonl, <result_dir>/logs/<question id>.json (one per question) and logs/run.log."""
import os, sys, json, time, shutil, argparse, threading, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import merge_sub

AGENT_FLAGS = ("--k 4 --kc 8 --repairs 2 --shots 4 --qw 4 --think-map --escalate 4 --confirm 1 --max-len 24576 --max-tokens 1024 "
               "--compile-tokens 6144 --pipeline --quant fp8 --prefer-map fmt:automap --adj-rows move").split()


def n_gpus():
    try: out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60).stdout
    except Exception: return 0
    return sum(1 for l in out.splitlines() if l.startswith("GPU"))


class Log:
    def __init__(self, path): self.f = open(path, "a", encoding="utf-8")
    def __call__(self, *a):
        s = " ".join(str(x) for x in a); print(s, flush=True); self.f.write(s + "\n"); self.f.flush()


def run(cmd, log):
    """run a command, its output into the log; -> return code."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout: log(line.rstrip("\n"))
    return p.wait()


def smoke_split(split, dst, n_biz=2):
    """a copy of split holding only the questions of its first n_biz businesses (data linked, not copied)."""
    os.makedirs(dst, exist_ok=True)
    if not os.path.exists(os.path.join(dst, "data")): os.symlink(os.path.abspath(os.path.join(split, "data")), os.path.join(dst, "data"))
    for f in ("calendar.csv", "rulebook.md"): shutil.copy(os.path.join(split, f), dst)
    lines = [l for l in open(os.path.join(split, "questions.jsonl"), encoding="utf-8") if l.strip()]
    keep = sorted({json.loads(l)["business_id"] for l in lines})[:n_biz]
    with open(os.path.join(dst, "questions.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(l for l in lines if json.loads(l)["business_id"] in keep)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("result")
    ap.add_argument("--model", default=os.environ.get("MODEL", "/models/Qwen3.5-9B"))
    ap.add_argument("--budget", type=float, default=20400, help="seconds for both stages (6 h limit incl. start-up)")
    ap.add_argument("--runs", type=int, default=3); ap.add_argument("--work", default="/tmp/zc")
    ap.add_argument("--llm", type=int, default=1); ap.add_argument("--dp", type=int, default=0)
    ap.add_argument("--rules-workers", type=int, default=0)
    ap.add_argument("--canon", type=int, default=1, help="stage 1.5: the model restates the pending questions in the training "
                    "questions' style (src/canon_llm.py); the rules answer the restatements they read, the rest goes to stage 2")
    ap.add_argument("--min-conf", type=float, default=0.75, help="model answers kept only when the mean vote share of the runs is at "
                    "least this (src/vote.py --min-conf)")
    ap.add_argument("--fake-responder", default="", help="tests / preflight: no model, agent.py --fake-responder; at most "
                    "--max-pending questions go to it")
    ap.add_argument("--max-pending", type=int, default=0, help="at most this many questions for the model, spread over the "
                    "businesses (0 = all; 8 with --fake-responder)")
    argv = sys.argv[1:]; extra = []
    if "--" in argv: i = argv.index("--"); argv, extra = argv[:i], argv[i + 1:]
    if "--fake-responder" in extra:                                   # preflight appends it at the very end (after '--')
        i = extra.index("--fake-responder"); argv += extra[i:i + 2]; del extra[i:i + 2]
    a = ap.parse_args(argv)
    t0 = time.time()
    logs = os.path.join(a.result, "logs"); os.makedirs(logs, exist_ok=True); os.makedirs(a.work, exist_ok=True)
    log = Log(os.path.join(logs, "run.log"))
    cache, rules = os.path.join(a.work, "cache"), os.path.join(a.work, "rules")
    final = os.path.join(a.result, "submission.jsonl"); rules_sub = os.path.join(rules, "submission.jsonl")
    log(f"split: {a.split}  start: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    if os.environ.get("ZC_SMOKE") == "1":                             # preflight: the same code path on 2 businesses
        a.split = smoke_split(a.split, os.path.join(a.work, "smoke_split")); log(f"smoke: split reduced to {a.split}")

    # ---- stage 1: rules
    cmd = [sys.executable, os.path.join(HERE, "rules_first.py"), a.split, rules, "--cache", cache, "--log-dir", logs]
    if a.rules_workers: cmd += ["--workers", str(a.rules_workers)]
    rc = run(cmd, log)
    if rc != 0: log(f"rules stage exit code {rc}: every question goes to the model")
    n = merge_sub.merge(a.split, final, [rules_sub], quiet=True)
    pend_file = os.path.join(rules, "pending.jsonl")
    pending = [l for l in open(pend_file, encoding="utf-8") if l.strip()] if rc == 0 and os.path.isfile(pend_file) else \
        [l for l in open(os.path.join(a.split, "questions.jsonl"), encoding="utf-8") if l.strip()]
    ng = n_gpus(); dp = a.dp or ng
    if a.fake_responder:                                              # smoke: CPU, a fake model, a few questions
        ng = max(ng, 1); dp = a.dp or 1; extra = extra + ["--fake-responder", a.fake_responder, "--workers", "4"]
        a.max_pending = a.max_pending or 8
    if a.max_pending and len(pending) > a.max_pending:
        import agent as _ag
        keep = {q["id"] for q in _ag.select_subset([json.loads(l) for l in pending], a.max_pending)}
        pending = [l for l in pending if json.loads(l)["id"] in keep]
    log(f"stage 1: {n[0]} answered by the rules, {len(pending)} left for the model ({time.time() - t0:.0f}s); gpus {ng}")
    if not pending or not a.llm or not ng:
        log(f"end: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}  answers: {n[0]}  ({time.time() - t0:.0f}s)"); return

    # ---- stage 2: the model on the pending questions (same data, pending questions only)
    psplit = os.path.join(a.work, "pending_split"); os.makedirs(psplit, exist_ok=True)
    if not os.path.exists(os.path.join(psplit, "data")): os.symlink(os.path.abspath(os.path.join(a.split, "data")), os.path.join(psplit, "data"))
    for f in ("calendar.csv", "rulebook.md"): shutil.copy(os.path.join(a.split, f), psplit)
    with open(os.path.join(psplit, "questions.jsonl"), "w", encoding="utf-8") as f: f.writelines(pending)
    canon_sub = os.path.join(a.work, "rules_canon", "submission.jsonl")
    shutil.rmtree(os.path.dirname(canon_sub), ignore_errors=True)     # never a stale answer of an earlier run in the same --work
    if a.canon:                                                       # stage 1.5 (one GPU, ~minutes): the model restates the
        # pending questions in the training questions' style; the rules answer the restatements they read (held-out precision
        # 0.90 train / 0.84 public paraphrases, above the model's: EXPERIMENT_LOG entry 26); the rest goes to stage 2 as asked
        csplit = os.path.join(a.work, "pending_canon"); rdir = os.path.join(csplit, "restated")
        cmd = [sys.executable, os.path.join(HERE, "canon_llm.py"), psplit, csplit, "--model", a.model, "--quant", "fp8", "--cache", cache]
        if a.fake_responder: cmd += ["--fake-responder", a.fake_responder]
        if run(cmd, log) == 0 and os.path.isfile(os.path.join(rdir, "questions.jsonl")):
            rc2 = run([sys.executable, os.path.join(HERE, "rules_first.py"), rdir, os.path.dirname(canon_sub), "--cache", cache, "--log-dir", logs]
                      + (["--workers", str(a.rules_workers)] if a.rules_workers else []), log)
            if rc2 == 0 and os.path.isfile(canon_sub):
                done = {json.loads(l)["id"] for l in open(canon_sub, encoding="utf-8") if l.strip()}
                pending = [l for l in pending if json.loads(l)["id"] not in done]
                with open(os.path.join(psplit, "questions.jsonl"), "w", encoding="utf-8") as f: f.writelines(pending)
                n = merge_sub.merge(a.split, final, [rules_sub, canon_sub], quiet=True)
                log(f"stage 1.5: rules answered {len(done)} restated questions; {len(pending)} left for the model")
            else: log(f"stage 1.5: rules on the restatements exit code {rc2}: ignored")
        else: log("stage 1.5 failed: stage 2 reads the questions as they are")
        if not pending:
            log(f"end: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}  ({time.time() - t0:.0f}s)"); return
    vote_sub = os.path.join(a.work, "llm_vote.jsonl")
    srcs = [rules_sub, canon_sub, vote_sub]
    log(f"stage 2: {len(pending)} questions  model {a.model}  dp {dp}  runs up to {a.runs}")

    def agent(i, budget, logdir):
        out = os.path.join(a.work, f"llm_{i}"); os.makedirs(out, exist_ok=True)
        cmd = [sys.executable, os.path.join(HERE, "agent.py"), psplit, out, "--model", a.model, "--dp", str(dp), "--budget", str(int(budget)),
               "--cache", cache, "--seed-base", str(i), "--log-dir", logdir] + AGENT_FLAGS + extra
        return out, run(cmd, log)

    stop = threading.Event()
    def merger(src):
        while not stop.wait(20):
            try: merge_sub.merge(a.split, final, [rules_sub, canon_sub, src], quiet=True)
            except Exception as e: log(f"merge: {type(e).__name__}: {e}")
    th = threading.Thread(target=merger, args=(os.path.join(a.work, "llm_0", "submission.jsonl"),), daemon=True); th.start()
    s0 = time.time(); out0, rc = agent(0, a.budget - (time.time() - t0), logs); last = time.time() - s0
    stop.set(); th.join()
    if rc != 0: log(f"model run 1 exit code {rc}")
    run([sys.executable, os.path.join(HERE, "vote.py"), vote_sub, out0, "--min-conf", str(a.min_conf), "--log-dir", logs], log)   # one run:
    # the same confidence floor as several (vote.py --min-conf)
    n = merge_sub.merge(a.split, final, srcs, quiet=True); log(f"after model run 1: rules {n[0]}, rules on restatements {n[1]}, model {n[2]}")
    dirs = [out0]
    for i in range(1, a.runs):
        left = a.budget - (time.time() - t0)
        if last * 1.15 >= left: log(f"model run {i + 1} skipped: {left:.0f}s left, a run takes {last:.0f}s"); break
        s0 = time.time(); ld = os.path.join(a.work, f"llm_{i}", "logs"); os.makedirs(ld, exist_ok=True)
        out_i, rc = agent(i, left, ld); last = time.time() - s0; dirs.append(out_i)
        run([sys.executable, os.path.join(HERE, "vote.py"), vote_sub] + dirs + ["--min-conf", str(a.min_conf), "--log-dir", logs], log)
        n = merge_sub.merge(a.split, final, srcs, quiet=True); log(f"after model run {i + 1}: rules {n[0]}, rules on restatements {n[1]}, model {n[2]}")
    log(f"end: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
