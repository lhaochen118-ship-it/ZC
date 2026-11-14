#!/usr/bin/env python3
"""v2 TG1 (handoff/04 §3.G, handoff/05 step 2.1): teacher data for the tool-writer LoRA WT, on TRAINING shops only.

Per Kaggle session (4 GPUs, --dp 4) the job takes its part of the training shops (--part i/n), splits them over the GPUs (each GPU
its own shops: a shop's data is loaded by one process only) and runs, per GPU:
  A  src/lrl_gen.py --mode rowgen : the base model (thinking) invents --rows-per-shop NEW rulebook rows computable on the shop's
     tables (variants of existing rows that change one element of the definition + new metrics of the kind a shop defines);
  B  src/lrl_gen.py --mode unit --rows : for every row of the shop — a sample of the training rulebook's rows, the rows of A, and
     the model-written new terms of TG0 (--newterm: cards_newterm*.jsonl of src/lrl_gen.py --mode newterm, terms whose meaning a
     checked program on a training question fixes) — the base model writes --samples functions (thinking), each run on the shop's
     data over 13 periods (units.verify); a function is kept when >= --min-agree samples return the same results (self-consistency).
Outputs: OUT/units_teacher.jsonl (kept functions with provenance), OUT/rows.jsonl, OUT/g<k>/{A,B}/ raw outputs, OUT/tg1_summary.json.
No answer of any split is read; held-out shops are processed too and flagged later (training/data/wt_build.py keeps them for
measurement only).
Usage: tg1_job.py <train_split> <out> --model M [--part i/n] [--dp 4] [--newterm F[,F]] [--samples 6] [--rows-per-shop 20]
                  [--train-rows-per-shop 30] [--newterm-per-shop 60] [--think-tokens 7168] [--min-agree 3] [--fake-responder f.py]"""
import os, sys, json, glob, time, zlib, argparse, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src") if os.path.isdir(os.path.join(ROOT, "src")) else ROOT
sys.path.insert(0, SRC)


def find(name):
    if os.path.isfile(name): return name
    c = glob.glob(f"/kaggle/**/{os.path.basename(name)}", recursive=True)
    return c[0] if c else ""


def newterm_rows(paths, shops, per_shop):
    out = {}
    for p in [x for x in paths.split(",") if x]:
        fs = sorted(glob.glob(os.path.join(p, "**", "cards_newterm*.jsonl"), recursive=True)) if os.path.isdir(p) else [find(p)]
        for f in fs:
            if not f or not os.path.isfile(f): continue
            for l in open(f, encoding="utf-8"):
                if not l.strip(): continue
                r = json.loads(l); b = r.get("business_id") or r.get("bid")
                if b not in shops or not r.get("term") or not r.get("definition"): continue
                out.setdefault(b, []).append({"bid": b, "term": r["term"], "definition": r["definition"][:500], "section": "thuật ngữ mới",
                                              "src": "newterm", "qid": r.get("id"), "question": r.get("question"), "orig_question": r.get("orig_question"),
                                              "program": r.get("program"), "answer": r.get("answer"), "as_of_date": r.get("as_of_date")})
    for b in out:                                                       # a stable sample per shop
        out[b] = sorted(out[b], key=lambda r: zlib.crc32(f"{b}|{r['term']}".encode()))[:per_shop]
    return out


def train_rows(split, shops, per_shop):
    import hints
    rows = list(hints.rulebook_rows(open(os.path.join(split, "rulebook.md"), encoding="utf-8").read()).values())
    out = {}
    for b in shops:
        pick = sorted(rows, key=lambda td: zlib.crc32(f"{b}|{td[0]}".encode()))[:per_shop]
        out[b] = [{"bid": b, "term": t, "definition": d[:500], "section": "rulebook", "src": "train_rule"} for t, d in pick]
    return out


def lrl(split, out, mode, model, gpu, extra, a, log):
    cmd = [sys.executable, os.path.join(SRC, "lrl_gen.py"), split, out, "--model", model, "--mode", mode, "--quant", "fp8",
           "--max-len", str(a.max_len), "--max-num-seqs", str(a.max_num_seqs)] + extra
    if a.fake_responder: cmd += ["--fake-responder", a.fake_responder]
    env = dict(os.environ)
    if gpu is not None and not a.fake_responder: env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.makedirs(out, exist_ok=True)
    return subprocess.Popen(cmd, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)


def wait_all(procs, deadline):
    rcs = []
    for p in procs:
        try: rcs.append(p.wait(timeout=max(30, deadline - time.time())))
        except subprocess.TimeoutExpired:
            import signal
            try: os.killpg(p.pid, signal.SIGKILL)
            except Exception: p.kill()
            p.wait(); rcs.append("killed_at_deadline")
    return rcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("out"); ap.add_argument("--model", required=True)
    ap.add_argument("--part", default="0/1"); ap.add_argument("--dp", type=int, default=4); ap.add_argument("--newterm", default="")
    ap.add_argument("--samples", type=int, default=6); ap.add_argument("--rows-per-shop", type=int, default=20)
    ap.add_argument("--train-rows-per-shop", type=int, default=30); ap.add_argument("--newterm-per-shop", type=int, default=60)
    ap.add_argument("--think-tokens", type=int, default=7168); ap.add_argument("--min-agree", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=16384); ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--fake-responder", default=""); ap.add_argument("--cache", default="")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    model = a.model
    if not os.path.isdir(model):                                         # Kaggle: the mounted Kaggle Model (as run_kaggle resolves it)
        c = [os.path.dirname(x) for x in glob.glob("/kaggle/input/**/config.json", recursive=True) if "qwen" in x.lower()]
        model = c[0] if c else model
    deadline = float(os.environ.get("ZC_DEADLINE") or (t0 + 11 * 3600)) - 600
    i, n = map(int, a.part.split("/"))
    shops = sorted(os.path.basename(d) for d in glob.glob(os.path.join(a.split, "data", "*")) if os.path.isdir(d))[i::n]
    if os.environ.get("ZC_SMOKE") == "1": shops = shops[:2]
    groups = [shops[g::a.dp] for g in range(a.dp)]
    groups = [g for g in groups if g]
    print(f"[tg1] part {a.part}: {len(shops)} shops over {len(groups)} GPUs: {groups}", flush=True)
    # A: synthetic rows per shop
    procs = [lrl(a.split, os.path.join(a.out, f"g{k}", "A"), "rowgen", model, k, ["--shops", ",".join(g), "--rows-per-shop", str(a.rows_per_shop),
                 "--think-tokens", "8192"], a, os.path.join(a.out, f"g{k}_A.log")) for k, g in enumerate(groups)] if a.rows_per_shop > 0 else []
    rcA = wait_all(procs, deadline - 3 * 3600)                           # --rows-per-shop 0: no synthetic rows (a newterm-only run)
    print(f"[tg1] A rowgen rc={rcA} ({time.time() - t0:.0f}s)", flush=True)
    synth = {}
    for f in glob.glob(os.path.join(a.out, "g*", "A", "cards_rowgen*.jsonl")):
        for l in open(f, encoding="utf-8"):
            if l.strip(): r = json.loads(l); synth.setdefault(r["bid"], []).append(r)
    nt = newterm_rows(a.newterm, set(shops), a.newterm_per_shop) if a.newterm else {}
    tr = train_rows(a.split, shops, a.train_rows_per_shop) if a.train_rows_per_shop > 0 else {}
    allrows = []
    procs = []
    for k, g in enumerate(groups):
        rows, seen = [], set()
        for b in g:
            for r in tr.get(b, []) + synth.get(b, []) + nt.get(b, []):
                key = (b, r["term"].strip().lower())
                if key in seen: continue
                seen.add(key); rows.append(r)
        rf = os.path.join(a.out, f"g{k}", "rows.jsonl"); os.makedirs(os.path.dirname(rf), exist_ok=True)   # (no rowgen: no g<k>/ yet)
        with open(rf, "w", encoding="utf-8") as w:
            for r in rows: w.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        allrows += rows
        procs.append(lrl(a.split, os.path.join(a.out, f"g{k}", "B"), "unit", model, k,
                         ["--rows", rf, "--samples", str(a.samples), "--think-units", "1", "--think-tokens", str(a.think_tokens),
                          "--min-agree", str(a.min_agree), "--unit-repairs", "1"], a, os.path.join(a.out, f"g{k}_B.log")))
    with open(os.path.join(a.out, "rows.jsonl"), "w", encoding="utf-8") as w:
        for r in allrows: w.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    rcB = wait_all(procs, deadline)
    print(f"[tg1] B unit rc={rcB} ({time.time() - t0:.0f}s)", flush=True)
    kept = []
    for f in sorted(glob.glob(os.path.join(a.out, "g*", "B", "cards_unit*.jsonl"))):
        kept += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    with open(os.path.join(a.out, "units_teacher.jsonl"), "w", encoding="utf-8") as w:
        for u in kept: w.write(json.dumps(u, ensure_ascii=False, default=str) + "\n")
    from collections import Counter
    summ = {"part": a.part, "shops": shops, "rows": len(allrows), "rows_by_src": dict(Counter(r["src"] for r in allrows)),
            "kept": len(kept), "kept_by_src": dict(Counter(u.get("src", "?") for u in kept)), "rcA": rcA, "rcB": rcB, "s": round(time.time() - t0)}
    json.dump(summ, open(os.path.join(a.out, "tg1_summary.json"), "w"), indent=1, default=str)
    print("TG1", json.dumps(summ, default=str), flush=True)


if __name__ == "__main__":
    main()
