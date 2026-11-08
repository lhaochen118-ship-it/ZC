#!/usr/bin/env python3
"""Fixed evaluation sets of docs/08_MASTER_PLAN.md §3 -> tests/splits/<name>.txt (one question id per line, seed 20261108).

Ids: an original question keeps its id; a paraphrase of src/paraphrase_llm.py is '<id>~p<k>' (as src/stress_flow.py writes
them); an accent-free public question is '<id>~accentfree' (src/perturb.py). src/eval_flow.py builds the sub-split of a set
from these ids (the data, rulebook and calendar of the original split).

  pub_full   1000 public questions (dev; scored with data/raw/public_gt.jsonl)
  pub_k0     public paraphrases k=0 kept by the paraphraser (dev)
  pub_ho     public paraphrases k=1,2 (held-out)
  tr_ho      the 6 training businesses kept out of every training set / example bank: originals + their paraphrases (held-out)
  quick400   150 public (stratified by business x answer type) + 150 pub_k0 (by business) + 100 tr_ho (50 originals +
             50 paraphrases, by business)
  pub_noind  public questions of dev_011..dev_014 whose rule program calls an industry function (src/industry.py), run with
             the industry API hidden (simulates the private test's new industries); needs --rules-log (src/rules_first.py
             log on the public split)
  pub_noacc  the 1000 public questions without accents

Usage: make_splits.py [--para-pub F] [--para-train F] [--rules-log F] [--out tests/splits]"""
import os, sys, json, random, argparse
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
SEED = 20261108
TR_HO = ("biz_105", "biz_109", "biz_114", "biz_123", "biz_127", "biz_130")
IND_SHOPS = ("dev_011", "dev_012", "dev_013", "dev_014")


def jl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def stratified(items, key, n, rng):
    """n items, each stratum (key(item)) gets its proportional share (largest remainder), random inside a stratum."""
    by = defaultdict(list)
    for it in items: by[key(it)].append(it)
    tot = len(items); n = min(n, tot)
    quota = {k: n * len(v) / tot for k, v in by.items()}
    take = {k: int(q) for k, q in quota.items()}
    for k in sorted(by, key=lambda k: (-(quota[k] - take[k]), str(k)))[:n - sum(take.values())]: take[k] += 1
    out = []
    for k in sorted(by, key=str):
        v = sorted(by[k], key=lambda x: x["id"]); rng.shuffle(v); out += v[:take[k]]
    return sorted(out, key=lambda x: x["id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--para-pub", default=os.environ.get("ZC_PARA_PUB", ""))
    ap.add_argument("--para-train", default=os.environ.get("ZC_PARA_TRAIN", ""))
    ap.add_argument("--rules-log", default="", help="src/rules_first.py rules_log.jsonl of the public split (for pub_noind)")
    ap.add_argument("--out", default=os.path.join(ROOT, "tests", "splits"))
    a = ap.parse_args()
    pub = jl(os.path.join(ROOT, "data", "raw", "public_test", "public_test", "questions.jsonl"))
    tr = jl(os.path.join(ROOT, "data", "raw", "train", "train", "questions.jsonl"))
    gt = {r["id"]: r for r in jl(os.path.join(ROOT, "data", "raw", "public_gt.jsonl"))}
    pp = [r for r in jl(a.para_pub) if r.get("ok")]
    pt = [r for r in jl(a.para_train) if r.get("ok")]
    rng = random.Random(SEED)
    S = {}
    S["pub_full"] = sorted(q["id"] for q in pub)
    S["pub_k0"] = sorted(f"{r['id']}~p{r['k']}" for r in pp if r["k"] == 0)
    S["pub_ho"] = sorted(f"{r['id']}~p{r['k']}" for r in pp if r["k"] in (1, 2))
    ho_orig = [q for q in tr if q["business_id"] in TR_HO]
    ho_para = [{"id": f"{r['id']}~p{r['k']}", "business_id": r["business_id"]} for r in pt if r["business_id"] in TR_HO]
    S["tr_ho"] = sorted([q["id"] for q in ho_orig] + [x["id"] for x in ho_para])
    q1 = stratified([dict(q, t=gt[q["id"]]["type"]) for q in pub], lambda x: (x["business_id"], x["t"]), 150, rng)
    k0 = [{"id": f"{r['id']}~p0", "business_id": r["business_id"]} for r in pp if r["k"] == 0]
    q2 = stratified(k0, lambda x: x["business_id"], 150, rng)
    q3 = stratified(ho_orig, lambda x: x["business_id"], 50, rng) + stratified(ho_para, lambda x: x["business_id"], 50, rng)
    S["quick400"] = sorted(x["id"] for x in q1 + q2 + q3)
    if a.rules_log:
        import industry
        names = set(industry.INDUSTRY_NAMES)
        log = jl(a.rules_log); bq = {q["id"]: q["business_id"] for q in pub}
        def uses_ind(r):
            code = r.get("code") or r.get("program") or ""
            return str(r.get("tpl") or r.get("template") or "").startswith("ind_") or any(f"{n}(" in code for n in names)
        S["pub_noind"] = sorted(r["id"] for r in log if bq.get(r["id"]) in IND_SHOPS and uses_ind(r))
    S["pub_noacc"] = sorted(f"{q['id']}~accentfree" for q in pub)
    os.makedirs(a.out, exist_ok=True)
    for k, v in S.items():
        assert len(v) == len(set(v)), k
        open(os.path.join(a.out, f"{k}.txt"), "w").write("\n".join(v) + "\n")
        print(f"{k}: {len(v)}")


if __name__ == "__main__":
    main()
