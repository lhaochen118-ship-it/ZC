#!/usr/bin/env python3
"""Plan 5 W3 training data (docs/18 §2, DW3): judge records from the solver's OWN logged samples on the TRAINING shops.

Each logged question of a solver run on training shops (Process/runs/<run>/**/logs.jsonl.gz: c1a/b, rq5_tr_dev_para_a/b,
v2v4s{1,2}{a,b}, rq5_tr_dev_all) gives its distinct candidate programs with what they returned (src/judge.py candidates); the
training reference answer says which candidates are right (src/score.py). The label is the judge's output: the plan of a right
program (grammar.auto_plan: quantity | period | filters) and the choice (candidate number, 0 = none, U = unanswerable).
The prompt is the judge's inference prompt: the shop's solver prefix (system + # BUSINESS, taken from the shop's records of a
solver training mix) + src/judge.py question_block. No public / private test question is read.

  list  records: one per (run, question): every candidate, with sample shares; half of the multi-candidate ones shuffled
  point records: one per candidate (1 = right, 0 / U = wrong)
Label noise guard: a paraphrase whose candidates are all wrong is kept only when some run answered that same paraphrase right
(a paraphrase that changed the meaning is wrong against its original's reference in every run).
Validation: base question ids with hash % 10 == 0 are left out of training; their rows of one run become a small solver log
(<out>/tr_val.logs.jsonl.gz) for src/judge.py on the training split (policy calibration without the public test).

Usage: w_judge.py --runs c1a,c1b,... --mix training/data/out/v3a/v3a_mix.jsonl --answers data/raw/train/train/answers.jsonl
                  --out training/data/out/wj [--n-right 1300 --cap 3 --max-chars 13000 --seed 7]"""
import os, sys, json, glob, gzip, random, hashlib, argparse
from collections import Counter, defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
import judge, score, grammar


def is_val(base_id):
    return int(hashlib.md5(base_id.encode()).hexdigest(), 16) % 10 == 0


def prefixes(mix):
    """{bid: (system, business)} from a solver training mix: the user turn up to the first per-question section."""
    out = {}
    op = gzip.open if mix.endswith(".gz") else open
    with op(mix, "rt", encoding="utf-8") as f:
        for l in f:
            r = json.loads(l); b = r.get("business_id")
            if not b or b in out: continue
            m = r["messages"]; u = m[1]["content"]
            cuts = [c for c in (u.find(k) for k in ("\n\n# EXAMPLES", "\n\n# RULES", "\n\n# HINTS", "\n\n# QUESTION")) if c > 0]
            if not cuts or not u.startswith("# BUSINESS"): continue
            out[b] = (m[0]["content"], u[:min(cuts)])
    return out


def programs(mix):
    """{question id: checked program} of the mix's records (base questions and paraphrases keep their own ids)."""
    out = {}
    op = gzip.open if mix.endswith(".gz") else open
    with op(mix, "rt", encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            if r.get("program") and r.get("id"): out.setdefault(r["id"], r["program"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="c1a,c1b,rq5_tr_dev_para_a,rq5_tr_dev_para_b,v2v4s1a,v2v4s1b,v2v4s2a,v2v4s2b,rq5_tr_dev_all")
    ap.add_argument("--runs-dir", default=os.path.join(ROOT, "Process", "runs"))
    ap.add_argument("--mix", default=os.path.join(ROOT, "Process", "train_data", "p3", "v3a_mix.jsonl.gz"))
    ap.add_argument("--answers", default=os.path.join(ROOT, "data", "raw", "train", "train", "answers.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "training", "data", "out", "wj"))
    ap.add_argument("--n-right", type=int, default=1300, help="list: all-right questions sampled (the rest of the mix is every "
                    "mixed / all-wrong question)"); ap.add_argument("--cap", type=int, default=3, help="records per question id and kind")
    ap.add_argument("--max-chars", type=int, default=13000); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--val-run", default="c1a")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rnd = random.Random(a.seed)
    A = score.load(a.answers); P = prefixes(a.mix); D1 = programs(a.mix)
    print(f"answers {len(A)}, shop prefixes {len(P)}, checked programs {len(D1)}", flush=True)
    rows = []                                                           # (run, row, cands, n, right flags)
    good = {}                                                           # question id (as logged) -> a right program of some run
    for run in [x for x in a.runs.split(",") if x]:
        fs = sorted(glob.glob(os.path.join(a.runs_dir, run, "**", "logs.jsonl.gz"), recursive=True))
        if not fs: print(f"  {run}: no logs", flush=True); continue
        L = judge.load_logs(fs[0]); k = 0
        for i, r in L.items():
            ref = A.get(i.split("~")[0])
            if ref is None or r.get("business_id") not in P: continue
            cands, n = judge.candidates(r)
            if not cands: continue
            ok = [score.judge(c["answer"], ref) == "correct" for c in cands]
            rows.append((run, r, cands, n, ok)); k += 1
            for c, o in zip(cands, ok):
                if o and c["answer"].get("type") != "unanswerable": good.setdefault(i, c["code"])
        print(f"  {run}: {k} questions with candidates", flush=True)
    answered_right = {r["id"] for _, r, _, _, ok in rows if any(ok)}    # ids some run answered right (label noise guard)

    def plan_of(i, cands, ok, ref):
        if ref["type"] == "unanswerable": return "Unanswerable | - | -"
        code = next((c["code"] for c, o in zip(cands, ok) if o), None) or good.get(i) or D1.get(i) or D1.get(i.split("~")[0])
        return grammar.auto_plan(code) if code else None

    def rec(kind, run, r, cands, n, plan, choice, mode):
        sysm, biz = P[r["business_id"]]
        u = biz + "\n\n" + judge.question_block(r["question"], r["as_of_date"], cands, mode, n)
        if len(sysm) + len(u) > a.max_chars: return None
        return {"id": r["id"], "business_id": r["business_id"], "source": f"WJ_{mode}_{kind}", "run": run, "choice": choice,
                "messages": [{"role": "system", "content": sysm}, {"role": "user", "content": u},
                             {"role": "assistant", "content": judge.label_text(plan, choice)}]}

    L_mixed, L_wrong, L_right, PT_pos, PT_neg = [], [], [], [], []
    seen = Counter(); val_rows = []; st = Counter()
    order = list(range(len(rows))); rnd.shuffle(order)
    for idx in order:
        run, r, cands, n, ok = rows[idx]; i = r["id"]; base = i.split("~")[0]; ref = A[base]
        if is_val(base):
            if run == a.val_run: val_rows.append(r)
            st["val"] += 1; continue
        plan = plan_of(i, cands, ok, ref)
        if plan is None: st["no_plan"] += 1; continue
        unans = ref["type"] == "unanswerable"
        kind = "right" if all(ok) else ("mixed" if any(ok) else "wrong")
        if kind == "wrong" and "~" in i and i not in answered_right: st["wrong_dropped(noise guard)"] += 1; continue
        st[kind] += 1
        # ---- list record
        if seen[(i, kind, "l")] < a.cap:
            cs, okk = list(cands), list(ok)
            if len(cs) > 1 and rnd.random() < 0.5:                       # order by votes half of the time, shuffled otherwise: the judge
                z = list(zip(cs, okk)); rnd.shuffle(z); cs, okk = map(list, zip(*z))   # must read the programs, not the position
            choice = (okk.index(True) + 1) if any(okk) else ("U" if unans else 0)
            x = rec(kind, run, r, cs, n, plan, choice, "list")
            if x is not None:
                seen[(i, kind, "l")] += 1
                (L_right if kind == "right" else L_mixed if kind == "mixed" else L_wrong).append(x)
        # ---- point records
        for c, o in zip(cands, ok):
            key = (i, "p", o, c["key"])
            if seen[key] or seen[(i, "p", o)] >= a.cap: continue
            x = rec("pos" if o else "neg", run, r, [c], n, plan, 1 if o else ("U" if unans else 0), "point")
            if x is None: continue
            seen[key] += 1; seen[(i, "p", o)] += 1
            (PT_pos if o else PT_neg).append(x)
    rnd.shuffle(L_right); L_right_s = L_right[:a.n_right]
    lst = L_mixed + L_wrong + L_right_s; rnd.shuffle(lst)
    rnd.shuffle(PT_pos); pt = PT_neg + PT_pos[:max(len(PT_neg), 600)]; rnd.shuffle(pt)
    mixd = lst[:] + pt[:len(pt) // 2]; rnd.shuffle(mixd)

    def dump(name, recs):
        p = os.path.join(a.out, name)
        with open(p, "w", encoding="utf-8") as f:
            for x in recs: f.write(json.dumps(x, ensure_ascii=False) + "\n")
        ch = Counter(str(x["choice"]) for x in recs); src = Counter(x["source"] for x in recs)
        chars = sum(len(m["content"]) for x in recs for m in x["messages"])
        print(f"{name}: {len(recs)} records, ~{chars / 3.1 / 1e6:.1f}M tokens (chars / 3.1); choices {dict(ch.most_common(8))}; {dict(src)}", flush=True)
    dump("wj_list.jsonl", lst); dump("wj_point.jsonl", pt); dump("wj_both.jsonl", mixd)
    with gzip.open(os.path.join(a.out, "tr_val.logs.jsonl.gz"), "wt", encoding="utf-8") as f:
        for r in val_rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"stats {dict(st)}; list: mixed {len(L_mixed)}, wrong {len(L_wrong)}, right {len(L_right_s)} of {len(L_right)}; "
          f"point: pos {len(PT_pos)}, neg {len(PT_neg)}; validation rows ({a.val_run}) {len(val_rows)}", flush=True)


if __name__ == "__main__":
    main()
