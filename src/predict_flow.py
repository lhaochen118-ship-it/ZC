#!/usr/bin/env python3
"""The submitted system end to end (docker/predict.sh runs it): the hybrid flow (docs/08_MASTER_PLAN.md §4.1). Every question is
answered by a program the model writes (agent.py --flow hybrid: schema by automap, hints, retrieved checked examples, grammar,
precheck, tools, execution, vote, abstain); no rule answers a question by itself (user decision 08/11).

Flags come from --config (configs/<name>.json, default configs/flow_final.json when present) + DEFAULTS below; flags after '--'
are passed to agent.py and override both. The submission is re-merged every 20 s while the model runs (merge_sub: one line per
question, abstain for the ones not answered yet), so a time-out keeps every answer already decided; independent runs (--runs)
are voted (src/vote.py) while the time allows.

Usage: predict_flow.py <split> <result_dir> [--model DIR] [--budget S] [--config F] [--runs N] [--work DIR] [--dp N]
                       [--fake-responder F] [--max-pending N] [-- <agent.py flags>]
Writes <result_dir>/submission.jsonl, <result_dir>/logs/<question id>.json and logs/run.log."""
import os, sys, json, time, shutil, argparse, threading, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import merge_sub

DEFAULTS = ("--flow hybrid --pipeline --quant fp8 --prefer-map fmt:automap --adj-rows move --qw 0 --kc 8 --think-map --compile-tokens 6144 "
            "--max-len 8192 --max-tokens 512 --k-fast 2 --k-max 8 --repairs 2 --max-turns 2 --shots 3 --prompt compact --grammar 0").split()


def defaults_for(cfg):
    """DEFAULTS minus every flag the config file sets: a flag on the command line beats the config's value
    (agent.parse_with_config), so passing both made '--prompt compact' override a config's 'prompt': 'full'."""
    keys = set()
    if cfg:
        keys = {k.lstrip("-").replace("-", "_") for k in json.load(open(cfg, encoding="utf-8")) if not k.startswith("_")}
    out, i = [], 0
    while i < len(DEFAULTS):
        t = DEFAULTS[i]; nxt = DEFAULTS[i + 1] if i + 1 < len(DEFAULTS) else None
        has_val = nxt is not None and not nxt.startswith("--")
        if t.lstrip("-").replace("-", "_") not in keys: out += [t] + ([nxt] if has_val else [])
        i += 2 if has_val else 1
    return out


def n_gpus():
    try: out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60).stdout
    except Exception: return 0
    return sum(1 for l in out.splitlines() if l.startswith("GPU"))


class Log:
    def __init__(self, path): self.f = open(path, "a", encoding="utf-8")
    def __call__(self, *a):
        s = " ".join(str(x) for x in a); print(s, flush=True); self.f.write(s + "\n"); self.f.flush()


def run(cmd, log):
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


def prepare_units(cfg, split, work, model, log, fake="", cache=""):
    """plan 4 W (docs/17): with "units_gen": 1 in the config, the model writes executable rule units for this split from its INPUTS
    only (src/lrl_gen.py --mode unit: the rows of each shop's industry section + rows no known rulebook has -> one function each,
    kept when it runs on the shop's data). Returns the --units path, or None."""
    if not cfg: return None
    c = json.load(open(cfg, encoding="utf-8"))
    if not int(c.get("units_gen", 0) or 0): return None
    out = os.path.join(work, "units"); os.makedirs(out, exist_ok=True); t = time.time()
    base = [sys.executable, os.path.join(HERE, "lrl_gen.py"), split, out, "--model", model, "--mode", "unit", "--quant", "fp8",
            "--samples", str(int(c.get("units_samples", 3) or 3)), "--think-units", str(int(c.get("units_think", 0) or 0))] + \
           (["--cache", cache] if cache else []) + (["--fake-responder", fake] if fake else [])
    ng = 1 if fake else max(1, n_gpus())                               # one shard of the rows per GPU, in parallel
    lim = float(c.get("units_max_s", 2700) or 2700)                    # the writer's time cap (a private test with many new rows must
    procs = [subprocess.Popen(base + (["--shard", f"{i}/{ng}"] if ng > 1 else []),   # not eat the solver's budget): a shard still
                              env=dict(os.environ, **({"CUDA_VISIBLE_DEVICES": str(i)} if ng > 1 else {})),   # running then is killed
                              stdout=open(os.path.join(out, f"shard{i}.log"), "w"), stderr=subprocess.STDOUT,   # with its vLLM engine
                              start_new_session=True) for i in range(ng)]                                     # (process group) and
    rcs = []                                                           # the run goes on with the units of the shards that finished
    for i, p in enumerate(procs):
        try: rcs.append(p.wait(timeout=max(1.0, t + lim - time.time())))
        except subprocess.TimeoutExpired:
            import signal
            try: os.killpg(p.pid, signal.SIGKILL)
            except Exception: p.kill()
            p.wait(); rcs.append("timeout"); log(f"units shard {i}: killed at the {lim:.0f}s cap")
    for i in range(ng):
        for l in open(os.path.join(out, f"shard{i}.log"), encoding="utf-8", errors="replace").readlines()[-3:]: log(f"units shard {i}: {l.rstrip()}")
    f = os.path.join(out, "cards_unit.jsonl")
    if ng > 1:                                                         # merge the shards' units
        with open(f, "w", encoding="utf-8") as w:
            for i in range(ng):
                p = os.path.join(out, f"cards_unit_{i}of{ng}.jsonl")
                if os.path.isfile(p): w.write(open(p, encoding="utf-8").read())
    n = sum(1 for _ in open(f, encoding="utf-8")) if os.path.isfile(f) else 0
    log(f"units: rc={rcs}, {n} verified units ({time.time() - t:.0f}s)")
    return f if n else None


def prepare_lrl(cfg, split, work, model, log, fake=""):
    """plan 3 (docs/16 §2.3, P2 / P3): when the config uses learned rule cards, a split whose rulebook no shipped compiled card
    belongs to (a new test) gets its own cards first, written by the model from the split's INPUTS only (rulebook rows; the
    categorical values of its shops' data). Returns the --lrl value (shipped files + the new ones) or None (config without cards)."""
    if not cfg: return None
    c = json.load(open(cfg, encoding="utf-8"))
    if not c.get("lrl"): return None
    import lrl
    refs = os.path.join(HERE, "refs", "lrl")
    names = [x for x in c["lrl"].split(",") if x]
    have = [os.path.join(refs, x) for x in names if os.path.isfile(os.path.join(refs, x))]
    rk = lrl.rb_key(open(os.path.join(split, "rulebook.md"), encoding="utf-8").read())
    keys = {(cd.get("trigger") or {}).get("rulebook") for f in have if "compile" in os.path.basename(f) for cd in map(json.loads, open(f, encoding="utf-8"))}
    if rk in keys:
        log(f"lrl: compiled cards of this rulebook are shipped ({len(have)} card files)"); return ",".join(have)
    out = os.path.join(work, "lrl"); os.makedirs(out, exist_ok=True)
    t = time.time()
    for mode in ("compile", "aliases"):
        cmd = [sys.executable, os.path.join(HERE, "lrl_gen.py"), split, out, "--model", model, "--mode", mode, "--quant", "fp8"] + \
              (["--fake-responder", fake] if fake else [])
        rc = run(cmd, log); log(f"lrl: {mode} for this split rc={rc} ({time.time() - t:.0f}s)")
    new = [os.path.join(out, f) for f in ("cards_compile.jsonl", "cards_aliases.jsonl") if os.path.isfile(os.path.join(out, f))]
    keep = [f for f in have if not any(k in os.path.basename(f) for k in ("compile", "alias"))]   # other splits' cards do not apply
    return ",".join(keep + new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("result")
    ap.add_argument("--model", default=os.environ.get("MODEL", "/models/Qwen3.5-9B"))
    ap.add_argument("--budget", type=float, default=20400, help="seconds for the whole run (6 h limit incl. start-up)")
    ap.add_argument("--config", default="", help="configs/<name>.json (default: configs/flow_final.json when it exists)")
    ap.add_argument("--runs", type=int, default=1); ap.add_argument("--work", default="/tmp/zc")
    ap.add_argument("--dp", type=int, default=0); ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--fake-responder", default=""); ap.add_argument("--max-pending", type=int, default=0)
    ap.add_argument("--lrl-fake", default="", help="tests: fake model of the card writer (src/lrl_gen.py --fake-responder)")
    argv = sys.argv[1:]; extra = []
    if "--" in argv: i = argv.index("--"); argv, extra = argv[:i], argv[i + 1:]
    if "--fake-responder" in extra:
        i = extra.index("--fake-responder"); argv += extra[i:i + 2]; del extra[i:i + 2]
    a = ap.parse_args(argv)
    t0 = time.time()
    logs = os.path.join(a.result, "logs"); os.makedirs(logs, exist_ok=True); os.makedirs(a.work, exist_ok=True)
    log = Log(os.path.join(logs, "run.log"))
    final = os.path.join(a.result, "submission.jsonl"); cache = os.path.join(a.work, "cache")
    log(f"split: {a.split}  start: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    if os.environ.get("ZC_SMOKE") == "1":
        a.split = smoke_split(a.split, os.path.join(a.work, "smoke_split")); log(f"smoke: split reduced to {a.split}")
    if a.config and not os.path.isfile(a.config): log(f"config {a.config} not found: defaults"); a.config = ""
    cfg = a.config or next((p for p in (os.path.join(os.path.dirname(HERE), "configs", "flow_final.json"), os.path.join(HERE, "configs", "flow_final.json"))
                            if os.path.isfile(p)), "")
    merge_sub.merge(a.split, final, [], quiet=True)                  # a complete (abstain) file from the first second
    ng = n_gpus(); dp = a.dp or ng
    if a.fake_responder:
        ng = max(ng, 1); dp = a.dp or 1; extra = extra + ["--fake-responder", a.fake_responder, "--workers", "4"]
        extra += ["--subset", str(a.max_pending or 8)]
    if not ng:
        log("no GPU: every question abstained"); return
    log(f"model {a.model}  dp {dp}  config {cfg or '(defaults)'}  runs up to {a.runs}")
    lv = prepare_lrl(cfg, a.split, a.work, a.model, log, a.lrl_fake)
    if lv is not None: extra = extra + ["--lrl", lv]
    uv = prepare_units(cfg, a.split, a.work, a.model, log, a.lrl_fake)
    if uv is not None: extra = extra + ["--units", uv]
    vote_sub = os.path.join(a.work, "llm_vote.jsonl")

    def agent(i, budget, logdir):
        out = os.path.join(a.work, f"llm_{i}"); os.makedirs(out, exist_ok=True)
        cmd = [sys.executable, os.path.join(HERE, "agent.py"), a.split, out, "--model", a.model, "--dp", str(dp), "--budget", str(int(budget)),
               "--cache", cache, "--seed-base", str(i), "--log-dir", logdir] + defaults_for(cfg) + (["--config", cfg] if cfg else []) + extra
        return out, run(cmd, log)

    stop = threading.Event()
    def merger(src):
        while not stop.wait(20):
            try: merge_sub.merge(a.split, final, [src], quiet=True)
            except Exception as e: log(f"merge: {type(e).__name__}: {e}")
    th = threading.Thread(target=merger, args=(os.path.join(a.work, "llm_0", "submission.jsonl"),), daemon=True); th.start()
    s0 = time.time(); out0, rc = agent(0, a.budget - (time.time() - t0), logs); last = time.time() - s0
    stop.set(); th.join()
    if rc != 0: log(f"model run 1 exit code {rc}")
    n = merge_sub.merge(a.split, final, [os.path.join(out0, "submission.jsonl")], quiet=True); log(f"after run 1: {n}")
    dirs = [out0]
    for i in range(1, a.runs):
        left = a.budget - (time.time() - t0)
        if last * 1.15 >= left: log(f"run {i + 1} skipped: {left:.0f}s left, a run takes {last:.0f}s"); break
        s0 = time.time(); ld = os.path.join(a.work, f"llm_{i}", "logs"); os.makedirs(ld, exist_ok=True)
        out_i, rc = agent(i, left, ld); last = time.time() - s0; dirs.append(out_i)
        run([sys.executable, os.path.join(HERE, "vote.py"), vote_sub] + dirs + ["--min-conf", str(a.min_conf), "--log-dir", logs], log)
        n = merge_sub.merge(a.split, final, [vote_sub], quiet=True); log(f"after run {i + 1}: {n}")
    log(f"end: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
