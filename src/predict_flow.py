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


def visible_gpus():
    """the GPU ids this process may use: CUDA_VISIBLE_DEVICES when set (one experiment variant per GPU: soup_job --parallel-gpus),
    else every GPU nvidia-smi lists."""
    vis = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    if vis: return vis
    try: out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60).stdout
    except Exception: return []
    return [str(i) for i, l in enumerate(l for l in out.splitlines() if l.startswith("GPU"))]


def n_gpus():
    return len(visible_gpus())


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
    if c.get("units_union") and not c.get("units_file"):                 # v2 I7: the functions of several writers (e.g. S and WT-b v2
        parts = []                                                      # cover different rows), concatenated; I6 then checks them
        for k, spec in enumerate([{}] + list(c["units_union"])):
            ck = {x: y for x, y in c.items() if x != "units_union"}
            if k: ck = {x: y for x, y in ck.items() if x not in ("units_lora", "units_lora_on", "units_model")}
            ck.update(spec); wk = os.path.join(work, f"union{k}"); os.makedirs(wk, exist_ok=True)
            cf = os.path.join(wk, "cfg.json"); json.dump(ck, open(cf, "w", encoding="utf-8"))
            f = prepare_units(cf, split, wk, model, log, fake, cache); free_w_models(wk, log)
            if f: parts.append([json.loads(l) for l in open(f, encoding="utf-8") if l.strip()])
        out, seen = [], set()
        for us in parts:
            for u in us:
                key = (u["bid"], u.get("_sig") or u["code"])
                if key in seen: continue
                seen.add(key)
                if any(o["bid"] == u["bid"] and o["name"] == u["name"] for o in out):
                    nn = u["name"] + f"_{len(out)}"; u = dict(u, code=u["code"].replace(f"def {u['name']}(", f"def {nn}(", 1),
                                                           signature=str(u.get("signature") or u["name"]).replace(u["name"], nn, 1), name=nn)
                out.append(u)
        if not out: return None
        f = os.path.join(work, "units_union.jsonl")
        with open(f, "w", encoding="utf-8") as w:
            for u in out: w.write(json.dumps(u, ensure_ascii=False) + "\n")
        log(f"units union: {[len(x) for x in parts]} -> {len(out)} functions"); return f
    if c.get("units_file"):                                             # v2 grid (experiments only): the units of an earlier run with
        f = find_input(c["units_file"])                                 # the same writer -> solver variants compared on the SAME
        if f: log(f"units: reused {f}"); return f                       # functions (and ~10-15 min saved); missing -> stop
        raise SystemExit(f"units_file {c['units_file']} not found")
    out = os.path.join(work, "units"); os.makedirs(out, exist_ok=True); t = time.time()
    if c.get("units_model") == "base" or (c.get("units_lora") and c.get("units_lora_on", "base") == "base"):
        model = os.environ.get("ZC_BASE_MODEL") or model                # v2 G: the tool writer WT-b is a LoRA of the BASE weights
    if c.get("units_lora") and not fake:                                # v2 G: the trained tool writer, merged into its base (one
        model, _ = w_model(resolve_adapter(c["units_lora"]), model, work, log, "merge")   # copy under work/, ~1.5 min)
        log(f"units: tool writer {c['units_lora']} on {c.get('units_lora_on', 'base')}")
    base = [sys.executable, os.path.join(HERE, "lrl_gen.py"), split, out, "--model", model, "--mode", "unit", "--quant", "fp8",
            "--samples", str(int(c.get("units_samples", 3) or 3)), "--think-units", str(int(c.get("units_think", 0) or 0)),
            "--unit-repairs", str(int(c.get("units_repairs", 1))), "--think-tokens", str(int(c.get("units_think_tokens", 4096))),
            "--max-len", str(int(c.get("units_max_len", 16384 if int(c.get("units_think", 0) or 0) else 8192))),
            "--min-agree", str(int(c.get("units_min_agree", 0) or 0)), "--units-multi", str(int(c.get("units_multi", 0) or 0)),
            "--units-qctx", str(int(c.get("units_qctx", 0) or 0))] + \
           (["--cache", cache] if cache else []) + (["--fake-responder", fake] if fake else [])
    devs = visible_gpus()
    ng = 1 if fake else max(1, len(devs))                              # one shard of the rows per (visible) GPU, in parallel
    lim = float(c.get("units_max_s", 2700) or 2700)                    # the writer's time cap (a private test with many new rows must
    procs = [subprocess.Popen(base + (["--shard", f"{i}/{ng}"] if ng > 1 else []),   # not eat the solver's budget): a shard still
                              env=dict(os.environ, **({"CUDA_VISIBLE_DEVICES": devs[i]} if ng > 1 else {})),   # running then is killed
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
        ls = open(os.path.join(out, f"shard{i}.log"), encoding="utf-8", errors="replace").readlines()
        if rcs[i] not in (0, "timeout"):                               # a failed shard: its traceback (vLLM's shutdown lines come
            tb = [k for k, l in enumerate(ls) if l.startswith("Traceback")]   # after it), not just the last lines (phmqx 15/11)
            ls = ls[tb[-1]:tb[-1] + 40] if tb else ls[-40:]
        else: ls = ls[-3:]
        for l in ls: log(f"units shard {i}: {l.rstrip()[:400]}")
    f = os.path.join(out, "cards_unit.jsonl")
    if ng > 1:                                                         # merge the shards' units
        with open(f, "w", encoding="utf-8") as w:
            for i in range(ng):
                p = os.path.join(out, f"cards_unit_{i}of{ng}.jsonl")
                if os.path.isfile(p): w.write(open(p, encoding="utf-8").read())
    n = sum(1 for _ in open(f, encoding="utf-8")) if os.path.isfile(f) else 0
    log(f"units: rc={rcs}, {n} verified units ({time.time() - t:.0f}s)")
    return f if n else None


def prepare_unit_examples(cfg, split, work, model, log, units_path, fake="", cache=""):
    """v2 I3 (15/11): test-time examples of the shop's OWN functions. The solver called the model-written shop functions in only ~10 %
    of its samples (its few-shot examples come from training shops, which have no such functions). With "unit_examples": n in the
    config, the model writes questions an owner may ask that need each verified shop function + the program calling it
    (lrl_gen --mode unitq, inputs only: the split's rulebook functions and question days), kept when the program runs on the shop's
    data; agent / hybrid then shows up to n of them (most similar to the question, same shop) among the examples. Returns a file or None."""
    if not cfg or not units_path: return None
    c = json.load(open(cfg, encoding="utf-8"))
    n = int(c.get("unit_examples", 0) or 0)
    if n <= 0 and not int(c.get("unit_verify", 0) or 0): return None   # (I6 needs the questions even without I3)
    out = os.path.join(work, "unitq"); os.makedirs(out, exist_ok=True); t = time.time()
    base = [sys.executable, os.path.join(HERE, "lrl_gen.py"), split, out, "--model", model, "--mode", "unitq", "--quant", "fp8", "--rows", units_path,
            "--samples", str(int(c.get("unit_examples_per_unit", 3) or 3)), "--seed", "4242"] + (["--cache", cache] if cache else []) + \
           (["--fake-responder", fake] if fake else [])
    devs = visible_gpus(); ng = 1 if fake else max(1, len(devs))
    procs = [subprocess.Popen(base + (["--shard", f"{i}/{ng}"] if ng > 1 else []), env=dict(os.environ, **({"CUDA_VISIBLE_DEVICES": devs[i]} if ng > 1 else {})),
                              stdout=open(os.path.join(out, f"shard{i}.log"), "w"), stderr=subprocess.STDOUT, start_new_session=True) for i in range(ng)]
    lim = float(c.get("unit_examples_max_s", 900) or 900)
    for p in procs:
        try: p.wait(timeout=max(1.0, t + lim - time.time()))
        except subprocess.TimeoutExpired:
            import signal
            try: os.killpg(p.pid, signal.SIGKILL)
            except Exception: p.kill()
            p.wait()
    import glob
    rows = []
    for f in sorted(glob.glob(os.path.join(out, "cards_unitq*.jsonl"))):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                q = json.loads(l)
                rows.append({"id": f"ux_{len(rows)}", "business_id": q["bid"], "question": q["question"], "as_of_date": q["as_of_date"],
                             "code": q["program"], "unit": q.get("unit_name"), "answer": q.get("answer")})
    if not rows: log(f"unit examples: none ({time.time() - t:.0f}s)"); return None
    f = os.path.join(out, "shop_examples.jsonl")
    with open(f, "w", encoding="utf-8") as w:
        for r in rows: w.write(json.dumps(r, ensure_ascii=False) + "\n")
    if n > 0: os.environ["ZC_SHOP_EXAMPLES"] = f; os.environ["ZC_SHOP_EXAMPLES_K"] = str(n)
    log(f"unit examples: {len(rows)} for {len({r['business_id'] for r in rows})} shops ({time.time() - t:.0f}s)")
    return f


def verify_units_by_solver(cfg, a, units_path, ux_path, dp, cache, extra, log):
    """v2 I6 (15/11): semantic check of the model-written shop functions by a second model path. The solver answers the model-written
    questions of each function (I3 file, with the function's own answer from execution) WITHOUT the shop functions (k 4, vote, no
    judge); a function is dropped when the solver answered at least 2 of its questions and agreed with none (wrong functions made every
    sample agree on a wrong value: e.g. 0 / 32 calls right for one counting function). Returns the kept units file (or the input)."""
    import glob, score as scoremod
    rows = [json.loads(l) for l in open(ux_path, encoding="utf-8") if l.strip()]
    if not rows: return units_path
    vs = os.path.join(a.work, "uverify"); sp = os.path.join(vs, "split"); os.makedirs(os.path.join(sp, "data"), exist_ok=True); t = time.time()
    with open(os.path.join(sp, "questions.jsonl"), "w", encoding="utf-8") as w:
        for r in rows: w.write(json.dumps({"id": r["id"], "question": r["question"], "business_id": r["business_id"], "as_of_date": r["as_of_date"]}, ensure_ascii=False) + "\n")
    for b in {r["business_id"] for r in rows}:
        d = os.path.join(sp, "data", b)
        if not os.path.exists(d): os.symlink(os.path.abspath(os.path.join(a.split, "data", b)), d)
    for f in ("calendar.csv", "rulebook.md"):
        if os.path.isfile(os.path.join(a.split, f)): shutil.copy(os.path.join(a.split, f), sp)
    if os.path.isdir(os.path.join(a.split, "rulebooks")): shutil.copytree(os.path.join(a.split, "rulebooks"), os.path.join(sp, "rulebooks"), dirs_exist_ok=True)
    out = os.path.join(vs, "out"); os.makedirs(out, exist_ok=True); os.makedirs(os.path.join(vs, "logs"), exist_ok=True)
    c = json.load(open(cfg, encoding="utf-8"))
    cmd = [sys.executable, os.path.join(HERE, "agent.py"), sp, out, "--model", a.model, "--dp", str(dp), "--budget", str(int(c.get("unit_verify_max_s", 1200))),
           "--cache", os.path.join(vs, "cache"), "--log-dir", os.path.join(vs, "logs")] + defaults_for(cfg) + ["--config", cfg] + extra + ["--k-max", "4", "--k-fast", "4"]
    rc = run(cmd, lambda *m: None)
    sub = {}
    f = os.path.join(out, "submission.jsonl")
    if os.path.isfile(f):
        for l in open(f, encoding="utf-8"):
            if l.strip(): x = json.loads(l); sub[x["id"]] = x
    from collections import Counter
    agree, answered = Counter(), Counter()
    for r in rows:
        x = sub.get(r["id"]); key = (r["business_id"], r.get("unit"))
        if not x or x.get("type") == "abstain" or not r.get("answer"): continue
        answered[key] += 1
        try: agree[key] += scoremod.judge(x, r["answer"]) == "correct"
        except Exception: pass
    us = [json.loads(l) for l in open(units_path, encoding="utf-8") if l.strip()]
    keep, drop = [], []
    for u in us:
        k = (u["bid"], u["name"])
        (drop if answered[k] >= 2 and agree[k] == 0 else keep).append(u)
    dl = [(u["bid"], u["name"], f"{agree[(u['bid'], u['name'])]}/{answered[(u['bid'], u['name'])]}") for u in drop]
    log(f"unit verify (rc {rc}, {time.time() - t:.0f}s): {len(rows)} questions, kept {len(keep)} / {len(us)} functions; dropped {dl}")
    if not keep: return None
    fo = os.path.join(vs, "cards_unit_verified.jsonl")
    with open(fo, "w", encoding="utf-8") as w:
        for u in keep: w.write(json.dumps(u, ensure_ascii=False) + "\n")
    shutil.copy(fo, os.path.join(a.result, "logs", "cards_unit_verified.jsonl")) if os.path.isdir(os.path.join(a.result, "logs")) else None
    return fo


def resolve_adapter(name):
    """a LoRA adapter directory given by path or by name (searched next to the code, under /models and /kaggle/input:
    <root>/**/<name>/adapter/adapter_config.json or <root>/**/<name>/adapter_config.json); '' when none.
    'A+B' (v2): stacked adapters (B trained on top of the merged A) -> 'dirA+dirB', merged together with weight 1.0 each."""
    if not name: return ""
    if "+" in name: return "+".join(resolve_adapter(x) for x in name.split("+") if x)
    if os.path.isfile(os.path.join(name, "adapter_config.json")): return name
    import glob
    for root in [r for r in (os.environ.get("ZC_ADAPTER_ROOT", ""), os.path.dirname(HERE), HERE, "/models", "/kaggle/input") if r]:
        for pat in (os.path.join(root, "**", name, "adapter", "adapter_config.json"), os.path.join(root, "**", name, "adapter_config.json")):
            c = sorted(glob.glob(pat, recursive=True))
            if c: return os.path.dirname(c[0])
    raise SystemExit(f"adapter {name} not found (no silent fallback)")


def w_model(adapter, model, work, log, mode="merge"):
    """(model, lora) for a W phase: mode 'merge' = the adapter merged into a copy of the solver's weights (training/merge.py merge;
    cached under work/), 'lora' = the solver + the adapter converted for vLLM's LoRA loader (merge.py vllm)."""
    if not adapter: return model, ""
    stack = adapter.split("+")                                          # v2: stacked adapters, weight 1.0 each
    if len(stack) > 1 and mode != "merge": raise SystemExit("stacked W adapters need w_mode merge")
    mp = next((p for p in (os.path.join(os.path.dirname(HERE), "training", "merge.py"), os.path.join(HERE, "training", "merge.py"),
                           os.path.join(HERE, "merge.py")) if os.path.isfile(p)), None)
    if mp is None: raise SystemExit("training/merge.py missing (needed for the W adapters)")
    nm = lambda d: os.path.basename(os.path.dirname(d.rstrip("/"))) if os.path.basename(d.rstrip("/")) == "adapter" else os.path.basename(d.rstrip("/"))
    name = "+".join(nm(d) for d in stack)
    t = time.time()
    if mode == "lora":
        out = os.path.join(work, f"lora_{name}")
        if not os.path.isfile(os.path.join(out, "adapter_config.json")):
            rc = run([sys.executable, mp, "vllm", model, adapter, out], log)
            if rc != 0: raise SystemExit(f"merge.py vllm {adapter} failed")
        return model, out
    out = os.path.join(work, f"model_{name}")
    if not os.path.isfile(os.path.join(out, "config.json")):
        rc = run([sys.executable, mp, "merge", model, out, ",".join(stack), "--weights", ",".join(["1.0"] * len(stack))], log)
        if rc != 0: raise SystemExit(f"merge.py merge {adapter} failed")
        log(f"W adapter {name} merged into the solver's weights ({time.time() - t:.0f}s)")
    return out, ""


def find_input(name):
    """a file given by path or by name under the code folders or /kaggle/input (v2 grid: reused units / links)."""
    if os.path.isfile(name): return name
    import glob
    for root in (os.path.dirname(HERE), HERE, "/kaggle/input"):
        c = sorted(glob.glob(os.path.join(root, "**", os.path.basename(name)), recursive=True))
        if c: return c[0]
    return None


def free_w_models(work, log):
    """v2: the merged copies of the W phases (work/model_<adapter>, ~19 GB each) are removed once their phase is over (units, links
    and the judge would otherwise hold ~57 GB of disk in the container); ZC_KEEP_W_MODELS=1 keeps them (Kaggle studies)."""
    if os.environ.get("ZC_KEEP_W_MODELS") == "1": return
    import glob
    for d in glob.glob(os.path.join(work, "model_*")):
        if os.path.isdir(d): shutil.rmtree(d, ignore_errors=True); log(f"freed {os.path.basename(d)}")


def resolve_model(name):
    """P5: a W-phase model given by name in the config (a merged checkpoint mounted from a kernel output: <mount>/**/models/<name>/)."""
    if not name or os.path.isdir(name): return name
    import glob
    for pat in (f"/tmp/models/{name}/config.json", f"/kaggle/input/**/models/{name}/config.json"):
        c = glob.glob(pat, recursive=True)
        if c: return os.path.dirname(c[0])
    raise SystemExit(f"model {name} not found (no silent fallback)")


def prepare_link(cfg, split, work, model, log, fake="", cache="", cards=""):
    """plan 5 W2 (docs/18): with "links_gen": 1 in the config, the model links every question of the split to the rulebook rows it
    uses (src/link.py; inputs only), one shard of the shops per GPU. Returns the --links path, or None (config without it). The
    time cap "links_max_s" kills a shard still running (its questions keep the keyword cards)."""
    if not cfg: return None
    c = json.load(open(cfg, encoding="utf-8"))
    if not int(c.get("links_gen", 0) or 0): return None
    if c.get("links_file"):                                             # v2 grid: links of an earlier run (same linker WL1)
        f = find_input(c["links_file"])
        if f: log(f"links: reused {f}"); return f
        raise SystemExit(f"links_file {c['links_file']} not found")
    out = os.path.join(work, "links"); os.makedirs(out, exist_ok=True); t = time.time()
    if not cards:                                                       # the compiled cards of the config (their toolkit hints)
        refs = os.path.join(HERE, "refs", "lrl")
        cards = ",".join(os.path.join(refs, x) for x in str(c.get("lrl") or "").split(",") if x and "compile" in x and os.path.isfile(os.path.join(refs, x)))
    if c.get("links_model") and not fake: model = resolve_model(c["links_model"]); log(f"links: model {model}")
    lora = resolve_adapter(c.get("links_lora") or os.environ.get("LINK_LORA") or "")
    if lora and not fake: model, lora = w_model(lora, model, work, log, c.get("w_mode", "merge"))
    base = [sys.executable, os.path.join(HERE, "link.py"), split, out, "--model", model, "--quant", "fp8", "--k", str(int(c.get("links_k", 3) or 3)), "--max-df", str(c.get("links_max_df", 0.0) or 0.0)] + \
           (["--lora", lora] if lora else []) + \
           (["--cards", cards] if cards else []) + (["--cache", cache] if cache else []) + (["--fake-responder", fake] if fake else [])
    devs = visible_gpus()
    ng = 1 if fake else max(1, len(devs))
    lim = float(c.get("links_max_s", 1800) or 1800)
    procs = [subprocess.Popen(base + (["--shard", f"{i}/{ng}"] if ng > 1 else []),
                              env=dict(os.environ, **({"CUDA_VISIBLE_DEVICES": devs[i]} if ng > 1 else {})),
                              stdout=open(os.path.join(out, f"shard{i}.log"), "w"), stderr=subprocess.STDOUT, start_new_session=True) for i in range(ng)]
    rcs = []
    for i, p in enumerate(procs):
        try: rcs.append(p.wait(timeout=max(1.0, t + lim - time.time())))
        except subprocess.TimeoutExpired:
            import signal
            try: os.killpg(p.pid, signal.SIGKILL)
            except Exception: p.kill()
            p.wait(); rcs.append("timeout"); log(f"links shard {i}: killed at the {lim:.0f}s cap")
    for i in range(ng):
        for l in open(os.path.join(out, f"shard{i}.log"), encoding="utf-8", errors="replace").readlines()[-2:]: log(f"links shard {i}: {l.rstrip()[:400]}")
    f = os.path.join(out, "links.jsonl")
    if ng > 1:
        with open(f, "w", encoding="utf-8") as w:
            for i in range(ng):
                p = os.path.join(out, f"links_{i}of{ng}.jsonl")
                if os.path.isfile(p): w.write(open(p, encoding="utf-8").read())
    n = sum(1 for _ in open(f, encoding="utf-8")) if os.path.isfile(f) else 0
    log(f"links: rc={rcs}, {n} questions linked ({time.time() - t:.0f}s)")
    if not n: raise SystemExit("links_gen is set but the linker wrote nothing (no silent fallback)")
    return f


def judge_phase(cfg, a, dirs, units, log, t0, links=None):
    """plan 5 W3 (docs/18): with "judge": 1 in the config, src/judge.py reads the solver's logged samples (run 1) and the judge model
    (W: "judge_model" = a merged S+W checkpoint, or the solver's model + "judge_lora" = a LoRA-W adapter applied by vLLM) picks
    the candidate / unanswerable / abstain under the config's policy. Returns its submission file, or None (no judge configured,
    or not enough time left: the vote's answers stand). The judge's own time cap is "judge_max_s"."""
    if not cfg: return None
    c = json.load(open(cfg, encoding="utf-8"))
    if not int(c.get("judge", 0) or 0): return None
    logs_f = os.path.join(dirs[0], "logs.jsonl")
    if not os.path.isfile(logs_f): log("judge: no solver logs"); return None
    left = a.budget - (time.time() - t0); lim = min(float(c.get("judge_max_s", 1800) or 1800), left - 300)
    if lim < 300: log(f"judge: skipped ({left:.0f}s left)"); return None
    out = os.path.join(a.work, "judge"); os.makedirs(out, exist_ok=True)
    wm = (resolve_model(c["judge_model"]) if c.get("judge_model") and not a.fake_responder else None) or os.environ.get("W_MODEL") or a.model
    lora = resolve_adapter(c.get("judge_lora") or os.environ.get("W_LORA") or "")
    if lora and not a.fake_responder: wm, lora = w_model(lora, wm, a.work, log, c.get("w_mode", "merge"))
    mode = c.get("judge_mode", "list")
    cmd = [sys.executable, os.path.join(HERE, "judge.py"), a.split, out, "--model", wm, "--logs", logs_f, "--variants", mode,
           "--k", str(int(c.get("judge_k", 5))), "--tau", str(c.get("judge_tau", 0.6)), "--denom", str(c.get("judge_denom", "all")),
           "--u-answers", str(int(c.get("judge_u_answers", 1))), "--ensemble", str(c.get("judge_ensemble", 0) or 0), "--keep-unanimous", str(c.get("judge_keep_unanimous", 0.0)), "--only-split", str(int(c.get("judge_only_split", 1))),
           "--cache", os.path.join(a.work, "cache")] + (["--units", units] if units else []) + (["--lora", lora] if lora else []) + \
          (["--fake-responder", a.fake_responder] if a.fake_responder else []) + (["--links", links] if links and int(c.get("judge_links", 1)) else [])
    p = subprocess.Popen(cmd, stdout=open(os.path.join(out, "judge.log"), "w"), stderr=subprocess.STDOUT, start_new_session=True)
    try: rc = p.wait(timeout=lim)
    except subprocess.TimeoutExpired:
        import signal
        try: os.killpg(p.pid, signal.SIGKILL)
        except Exception: p.kill()
        p.wait(); log(f"judge: killed at the {lim:.0f}s cap; the vote's answers stand"); return None
    for l in open(os.path.join(out, "judge.log"), encoding="utf-8", errors="replace").readlines()[-3:]: log(f"judge: {l.rstrip()[:300]}")
    ens = float(c.get("judge_ensemble", 0) or 0) > 0 and "," in mode
    sub = os.path.join(out, "ensemble" if ens else mode.replace("+", "_"), "submission.jsonl")
    if rc != 0 or not os.path.isfile(sub): log(f"judge: rc {rc}, no submission; the vote's answers stand"); return None
    import shutil
    vd = os.path.join(out, "ensemble" if ens else mode.replace("+", "_"), "verdicts.jsonl")
    if os.path.isfile(vd): shutil.copy(vd, os.path.join(a.result, "logs", "judge_verdicts.jsonl"))
    return sub


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
    if rk in keys and not int(c.get("lrl_regen", 0) or 0):            # v2 public grid: lrl_regen 1 = W1 rewrites the cards even when
        log(f"lrl: compiled cards of this rulebook are shipped ({len(have)} card files)"); return ",".join(have)   # shipped (as on private)
    out = os.path.join(work, "lrl"); os.makedirs(out, exist_ok=True)
    t = time.time()
    cm = c.get("lrl_compile_mode", "compile")                         # plan 5 W1: "bind" = executable, checked toolkit bindings
    devs = visible_gpus(); ng = 1 if fake else max(1, len(devs))
    def job(mode, i, n):
        cmd = [sys.executable, os.path.join(HERE, "lrl_gen.py"), split, out, "--model", model, "--mode", mode, "--quant", "fp8"] + \
              (["--samples", str(int(c.get("lrl_bind_samples", 4)))] if mode == "bind" else []) + \
              (["--shard", f"{i}/{n}"] if n > 1 else []) + (["--fake-responder", fake] if fake else [])
        env = dict(os.environ, **({"CUDA_VISIBLE_DEVICES": devs[i % len(devs)]} if n > 1 and devs else {}))
        return subprocess.Popen(cmd, env=env, stdout=open(os.path.join(out, f"{mode}{i}.log"), "w"), stderr=subprocess.STDOUT)
    n1 = ng if cm == "bind" else 1                                      # compile / bind shards on every GPU, aliases after them
    procs = [job(cm, i, n1) for i in range(n1)]
    rcs = [p.wait() for p in procs]
    if n1 > 1:
        with open(os.path.join(out, "cards_compile.jsonl"), "w", encoding="utf-8") as w:
            for i in range(n1):
                p = os.path.join(out, f"cards_compile_{i}of{n1}.jsonl")
                if os.path.isfile(p): w.write(open(p, encoding="utf-8").read())
    log(f"lrl: {cm} for this split rc={rcs} ({time.time() - t:.0f}s)")
    rc = job("aliases", 0, 1).wait(); log(f"lrl: aliases for this split rc={rc} ({time.time() - t:.0f}s)")
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
    try:                                                              # mode H (user 14/11): hidden toolkit names for every phase
        hz = json.load(open(cfg, encoding="utf-8")).get("hide_api") if cfg else ""
        if hz: os.environ["ZC_HIDE_API"] = hz; log(f"hide_api: {hz} (exported to every phase)")
        # N4-A experimental flag is derived ONLY from the explicit config
        n4_enabled = bool(cfg and int(json.load(open(cfg, encoding="utf-8")).get("n4_graph", 0) or 0))
        if n4_enabled:
            os.environ["ZC_N4_GRAPH"] = "1"
            log("N4-A schema graph candidate evidence enabled")
        else:
            os.environ.pop("ZC_N4_GRAPH", None)

        um = json.load(open(cfg, encoding="utf-8")).get("units_max") if cfg else None
        if um: os.environ["ZC_UNITS_MAX"] = str(int(um)); log(f"units_max: {um}")
        if cfg and int(json.load(open(cfg, encoding="utf-8")).get("units_code", 0) or 0): os.environ["ZC_UNITS_CODE"] = "1"; log("units_code: on")
        if cfg and int(json.load(open(cfg, encoding="utf-8")).get("api_extra", 0) or 0): os.environ["ZC_API_EXTRA"] = "1"; log("api_extra: on")
        tkw = json.load(open(cfg, encoding="utf-8")).get("think_kw") if cfg else ""   # legacy (path rule of the 0.952 runs) | explicit | default
        if tkw: os.environ["ZC_THINK_KW"] = str(tkw); log(f"think_kw: {tkw} (every phase)")
        kvd = json.load(open(cfg, encoding="utf-8")).get("kv_dtype") if cfg else ""   # P5 plan: bigger backbones (14B) on one L4
        if kvd: os.environ["ZC_KV_DTYPE"] = str(kvd); log(f"kv_dtype: {kvd} (every phase)")
    except Exception as e: log(f"hide_api: config unreadable ({e})")
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
    uv = prepare_units(cfg, a.split, a.work, a.model, log, a.lrl_fake or a.fake_responder)
    free_w_models(a.work, log)
    if uv is not None: extra = extra + ["--units", uv]
    ux = prepare_unit_examples(cfg, a.split, a.work, a.model, log, uv, a.lrl_fake or a.fake_responder)   # v2 I3 (env for hybrid)
    if uv and ux and cfg and int(json.load(open(cfg, encoding="utf-8")).get("unit_verify", 0) or 0):    # v2 I6: drop functions the
        base_extra = [x for i, x in enumerate(extra) if x != "--units" and (i == 0 or extra[i - 1] != "--units")]   # solver contradicts
        uv2 = verify_units_by_solver(cfg, a, uv, ux, dp, cache, base_extra, log)
        extra = base_extra + (["--units", uv2] if uv2 else []); uv = uv2
    kv = prepare_link(cfg, a.split, a.work, a.model, log, a.lrl_fake or a.fake_responder, cards=",".join(x for x in (lv or "").split(",") if "compile" in os.path.basename(x)))
    free_w_models(a.work, log)
    if kv is not None: extra = extra + ["--links", kv]
    vote_sub = os.path.join(a.work, "llm_vote.jsonl")

    seed_off = int((json.load(open(cfg, encoding="utf-8")).get("seed_base", 0) if cfg else 0) or 0) * 1000   # P5 plan: seed 2 = "seed_base": 1
    if seed_off: log(f"seed_base offset {seed_off}")

    def agent(i, budget, logdir):
        out = os.path.join(a.work, f"llm_{i}"); os.makedirs(out, exist_ok=True)
        cmd = [sys.executable, os.path.join(HERE, "agent.py"), a.split, out, "--model", a.model, "--dp", str(dp), "--budget", str(int(budget)),
               "--cache", cache, "--seed-base", str(i + seed_off), "--log-dir", logdir] + defaults_for(cfg) + (["--config", cfg] if cfg else []) + extra
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
    jv = judge_phase(cfg, a, dirs, uv, log, t0, kv)                       # plan 5 W3: the judge re-decides from the solver's samples
    free_w_models(a.work, log)
    if jv: n = merge_sub.merge(a.split, final, [jv, os.path.join(dirs[0], "submission.jsonl")], quiet=True, first_decides=True); log(f"after the judge: {n}")
    log(f"end: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
