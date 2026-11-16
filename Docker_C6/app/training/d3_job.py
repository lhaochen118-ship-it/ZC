#!/usr/bin/env python3
import os, sys, re, json, glob, time, shutil, argparse, subprocess

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
PY = sys.executable


def find_model(key):
    for c in sorted(glob.glob("/kaggle/input/**/config.json", recursive=True)):
        d = os.path.dirname(c)
        if key.lower() in d.lower() and glob.glob(d + "/*.safetensors"): return d
    return None


def parse_questions(text, n_max=3):
    t = text.split("</think>")[-1]
    m = re.search(r"\{.*\}", t, re.S)
    qs = []
    if m:
        try: qs = json.loads(m.group(0)).get("questions") or []
        except Exception: qs = []
    if not qs: qs = re.findall(r'"([^"\n]{8,300})"', t)
    out = []
    for q in qs:
        q = " ".join(str(q).split())
        if not (8 <= len(q) <= 300): continue
        if re.search(r"\b[a-z]+_[a-z_]+\(|\b(?:item_name|order_date|line_amount|revenue\(|ANSWER)\b", q): continue
        out.append(q)
    return list(dict.fromkeys(out))[:n_max]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("out")
    ap.add_argument("--model", required=True, help="the student (round trip) / the 9B when --teacher self")
    ap.add_argument("--teacher", default="self"); ap.add_argument("--tp", type=int, default=1); ap.add_argument("--instances", type=int, default=4)
    ap.add_argument("--quant", default=""); ap.add_argument("--data", default="d3"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rt-config", default="configs/e0.json"); ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--temp", type=float, default=0.8); ap.add_argument("--tag", default="")
    ap.add_argument("--max-model-len", type=int, default=4096); ap.add_argument("--max-num-seqs", type=int, default=128)
    a = ap.parse_args()
    os.chdir(ROOT)
    t0 = time.time(); os.makedirs(a.out, exist_ok=True)
    D = os.path.join(HERE, "data", "out", a.data)
    progs = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(D, "d3_programs.jsonl"), encoding="utf-8") if l.strip()}
    prompts = [l for l in open(os.path.join(D, "d3_prompts.jsonl"), encoding="utf-8") if l.strip()]
    if a.limit: prompts = prompts[:a.limit]
    tm = a.model if a.teacher == "self" else find_model(a.teacher)
    tag = a.tag or (("self" if a.teacher == "self" else a.teacher.replace("/", "_")))
    if not tm: print(f"TEACHER NOT FOUND: {a.teacher}", flush=True); sys.exit(2)
    print(f"[d3] teacher {tm} tp {a.tp} x {a.instances}; {len(prompts)} prompts", flush=True)
    gpus = [",".join(str(g) for g in range(i * a.tp, (i + 1) * a.tp)) for i in range(a.instances)]
    procs = []
    for i, g in enumerate(gpus):
        shard = os.path.join(a.out, f"prompts_{i}.jsonl"); open(shard, "w", encoding="utf-8").writelines(prompts[i::a.instances])
        cmd = [PY, os.path.join(HERE, "vllm_gen.py"), "--model", tm, "--prompts", shard, "--out", os.path.join(a.out, f"gen_{i}.jsonl"),
               "--tp", str(a.tp), "--max-tokens", str(a.max_tokens), "--temp", str(a.temp), "--max-model-len", str(a.max_model_len),
               "--max-num-seqs", str(a.max_num_seqs), "--seed", str(17 * i)] + (["--quant", a.quant] if a.quant else [])
        procs.append(subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=g), stdout=open(os.path.join(a.out, f"gen_{i}.log"), "w"), stderr=subprocess.STDOUT))
    rcs = [p.wait() for p in procs]
    summ = []
    for i in range(a.instances):
        txt = open(os.path.join(a.out, f"gen_{i}.log"), errors="ignore").read()
        m = [l for l in txt.splitlines() if l.startswith("SUMMARY ")]
        summ.append(json.loads(m[-1][8:]) if m else {"rc": rcs[i], "tail": txt[-1500:]})
    print(f"[d3] teacher done rc {rcs} in {time.time() - t0:.0f}s: {json.dumps(summ)[:1500]}", flush=True)
    rows = []
    for i in range(a.instances):
        p = os.path.join(a.out, f"gen_{i}.jsonl")
        if not os.path.isfile(p): continue
        for l in open(p, encoding="utf-8"):
            o = json.loads(l); pr = progs.get(o["id"])
            if pr is None: continue
            for j, q in enumerate(parse_questions(o["text"])):
                rows.append({"id": f"{pr['id']}~t{j}", "business_id": pr["business_id"], "question": q, "as_of_date": pr["as_of_date"]})
    print(f"[d3] {len(rows)} questions from {len(progs)} programs", flush=True)
    if not rows: sys.exit(3)
    rt = os.path.join("/tmp", "d3_rt"); shutil.rmtree(rt, ignore_errors=True); os.makedirs(os.path.join(rt, "data"))
    for b in sorted({r["business_id"] for r in rows}): os.symlink(os.path.abspath(os.path.join(a.split, "data", b)), os.path.join(rt, "data", b))
    for f in ("calendar.csv", "rulebook.md"): shutil.copy(os.path.join(a.split, f), rt)
    with open(os.path.join(rt, "questions.jsonl"), "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    t1 = time.time(); ro = os.path.join("/tmp", "d3_rt_out")
    cmd = [PY, os.path.join(ROOT, "agent.py") if os.path.isfile(os.path.join(ROOT, "agent.py")) else os.path.join(ROOT, "src", "agent.py"),
           rt, ro, "--model", a.model, "--dp", "4", "--config", a.rt_config, "--k-fast", "1", "--k-max", "1", "--max-turns", "1",
           "--cache", "/tmp/zc_cache"]
    rc = subprocess.run(cmd).returncode
    print(f"[d3] round trip rc {rc} in {time.time() - t1:.0f}s", flush=True)
    sys.path.insert(0, ROOT if os.path.isfile(os.path.join(ROOT, "agent.py")) else os.path.join(ROOT, "src"))
    import agent
    sub = {}
    if os.path.isfile(os.path.join(ro, "submission.jsonl")):
        for l in open(os.path.join(ro, "submission.jsonl"), encoding="utf-8"):
            x = json.loads(l); sub[x["id"]] = x
    kept = 0
    with open(os.path.join(a.out, f"d3_{tag}.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            pr = progs[r["id"].split("~")[0]]; s = sub.get(r["id"]); ok = False
            if s and s.get("type") not in (None, "abstain"):
                try: ok = agent.canon_of({k: s.get(k) for k in ("type", "answer")}) == agent.canon_of(pr["answer"])
                except Exception: ok = False
            kept += ok
            f.write(json.dumps(dict(r, program=pr["code"], answer=pr["answer"], teacher=tag, rt_answer=s, kept=ok), ensure_ascii=False) + "\n")
    print(f"[d3] KEPT {kept} / {len(rows)} questions ({kept / max(1, len(rows)):.2%}); total {time.time() - t0:.0f}s", flush=True)
    for f in glob.glob(os.path.join(ro, "*.log")) + [os.path.join(ro, "logs.jsonl")]:
        if os.path.isfile(f) and os.path.getsize(f) < 200 * 2**20: shutil.copy(f, a.out)


if __name__ == "__main__":
    main()
