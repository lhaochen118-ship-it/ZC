#!/usr/bin/env python3
"""GPU validation of the submitted flow on question styles the rules were not written for (Kaggle job via run_kaggle.py
--script stress_flow.py): every question of the split rewritten by each perturbation (src/perturb.py: accent-free, chat
noise with a seed; the same rewrite as tests/eval_split.py --perturb-q), plus the original questions when 'orig' is
listed, all in ONE split (id + '~' + tag) so that each business is mapped once; then src/predict_flow.py on it (rules
first, the model on what they leave). Scored offline against the reference answers by the original id.

Usage: stress_flow.py <split> <out> [--modes orig,accentfree,noise:1,noise:2,noise:3,para:<paraphrases.jsonl>] [predict_flow options]
       [-- agent flags]   (para: every kept paraphrase of src/paraphrase_llm.py, id + '~p' + k)"""
import os, sys, json, random, shutil, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from perturb import perturb_question


def main():
    split, out = sys.argv[1], sys.argv[2]; rest = sys.argv[3:]
    modes = "accentfree,noise:1,noise:2,noise:3"
    if "--modes" in rest: i = rest.index("--modes"); modes = rest[i + 1]; del rest[i:i + 2]
    sp = os.path.join(out, "stress_split"); os.makedirs(sp, exist_ok=True)
    if not os.path.exists(os.path.join(sp, "data")): os.symlink(os.path.abspath(os.path.join(split, "data")), os.path.join(sp, "data"))
    for f in ("calendar.csv", "rulebook.md"): shutil.copy(os.path.join(split, f), sp)
    qs = [json.loads(l) for l in open(os.path.join(split, "questions.jsonl"), encoding="utf-8") if l.strip()]
    by = {}
    for q in qs: by.setdefault(q["business_id"], []).append(q)
    if os.environ.get("ZC_SMOKE") == "1": by = {b: by[b] for b in sorted(by)[:2]}   # preflight: 2 businesses
    n = 0
    para = {}                                                           # 'para:<file>': paraphrases written by src/paraphrase_llm.py
    for m in modes.split(","):
        if m.startswith("para:"):
            path = m[5:]
            if not os.path.isfile(path):                               # a Kaggle dataset: its mount path depends on the layout
                import glob
                path = next(iter(sorted(glob.glob("/kaggle/input/**/" + os.path.basename(path), recursive=True))), "")
            if not path:
                if os.environ.get("ZC_SMOKE") == "1": print(f"smoke: {m[5:]} not found, mode skipped", flush=True); continue
                sys.exit(f"paraphrase file {m[5:]} not found")
            for r in map(json.loads, open(path, encoding="utf-8")):
                if r.get("ok"): para.setdefault(r["id"], []).append((r["k"], r["paraphrase"]))
    with open(os.path.join(sp, "questions.jsonl"), "w", encoding="utf-8") as f:
        for b in sorted(by):
            for m in modes.split(","):
                if m.startswith("para:"):
                    for q in by[b]:
                        for k, text in para.get(q["id"], []):
                            f.write(json.dumps(dict(q, id=f"{q['id']}~p{k}", question=text), ensure_ascii=False) + "\n"); n += 1
                    continue
                rng = random.Random(f"{m}:{b}")                      # tests/eval_split.py: one generator per business and mode
                for q in by[b]:
                    text = q["question"] if m == "orig" else perturb_question(q["question"], m.split(":")[0], rng)
                    f.write(json.dumps(dict(q, id=f"{q['id']}~{m.replace(':', '')}", question=text), ensure_ascii=False) + "\n"); n += 1
    print(f"stress split: {n} questions ({modes})", flush=True)
    work = os.path.join("/tmp", "zc_stress"); shutil.rmtree(work, ignore_errors=True)
    flow = os.path.join(work, "flow")                                   # per-question logs stay in /tmp (thousands of files made
    rc = subprocess.run([sys.executable, os.path.join(HERE, "predict_flow.py"), sp, flow, "--work", work] + rest).returncode   # the
    os.makedirs(os.path.join(out, "flow"), exist_ok=True)               # Kaggle output download hit the API rate limit, zc-v30)
    for f in ("submission.jsonl", os.path.join("logs", "run.log")):
        if os.path.isfile(os.path.join(flow, f)): shutil.copy(os.path.join(flow, f), os.path.join(out, "flow", os.path.basename(f)))
    for f in ("rules/submission.jsonl", "rules/pending.jsonl", "rules/rules_log.jsonl", "llm_vote.jsonl"):   # keep the stage outputs
        if os.path.isfile(os.path.join(work, f)): shutil.copy(os.path.join(work, f), os.path.join(out, f.replace("/", "_")))
    for d in sorted(os.listdir(work)) if os.path.isdir(work) else []:
        if d.startswith("llm_") and os.path.isdir(os.path.join(work, d)):
            for f in ("submission.jsonl", "logs.jsonl", "compile_logs.jsonl"):
                if os.path.isfile(os.path.join(work, d, f)):
                    os.makedirs(os.path.join(out, d), exist_ok=True); shutil.copy(os.path.join(work, d, f), os.path.join(out, d, f))
    sys.exit(rc)


if __name__ == "__main__":
    main()
