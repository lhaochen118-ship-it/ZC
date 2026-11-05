#!/usr/bin/env python3
"""Vote across several complete runs of the agent (multi-run ensemble of the final system).

Each run is an independent draw of the sampled mappings and programs; the per-question plurality over runs is more
accurate and more stable than any single run (train, 4 runs: PA 0.980 vs 0.928-0.969 for the single runs, wrong answers
19 vs 29-82). An abstaining run gives no vote; the top answer needs at least `min_share` of the runs (0.4: 2 of 4,
2 of 3, 1 of 2) of the runs that reached the question; ties are broken by the summed confidence, then by run order.
Train, every subset of the 4 runs: 2 runs 0.956-0.972 (about the mean of the two), 3 runs 0.976-0.981, 4 runs 0.980.
The submission is written atomically.
Usage: vote.py <out_submission.jsonl> <run_dir> ... [--min-share 0.4] [--min-conf 0.0] [--log-dir DIR]
--min-conf: abstain unless the mean vote share of the runs that reached the question is at least this (the submitted flow
uses 0.75: Kaggle v30 noise + v31 paraphrase k=0 questions left by the rules, net +126 at 0.5 -> +137 at 0.75; held-out
paraphrases k=1,2 +73 -> +100)."""
import os, sys, json, glob, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent


def load(run):
    p = os.path.join(run, "logs.jsonl")
    return {r["id"]: r for r in map(json.loads, open(p, encoding="utf-8"))} if os.path.isfile(p) else {}


def penalties(run):
    """{business_id: penalty of the mapping the run chose} (canon.warning_penalty summed): map_penalties.json when present
    (recomputed offline), else the compile log's chosen_penalty; {} when unknown."""
    p = os.path.join(run, "map_penalties.json")
    if os.path.isfile(p): return {k: float(v) for k, v in json.load(open(p, encoding="utf-8")).items()}
    out = {}
    for f in [os.path.join(run, "compile_logs.jsonl")] + sorted(glob.glob(os.path.join(run, "shard*", "compile_logs.jsonl"))):
        if not os.path.isfile(f): continue
        for l in open(f, encoding="utf-8"):
            try: r = json.loads(l)
            except Exception: continue
            if "chosen_penalty" in r: out[r["business_id"]] = float(r["chosen_penalty"])
        if out: break
    return out


TIE_GAP = 10.0   # a split is broken only by a run whose mapping is clearly cleaner (a revenue / short-name / staff warning less)


def vote(runs, min_share=0.4, pens=None, min_conf=0.0):
    """[logs of each run] -> {qid: (final answer, {canon answer: votes})} for every question of any run. pens: [{business:
    penalty of the run's chosen mapping}] per run; a question no answer wins (split) takes the answer of the ONE run whose
    mapping penalty for that business is lower than every other answering run's by >= TIE_GAP (the runs disagree because
    their mappings differ, and the checks say which mapping is wrong); otherwise it abstains as before."""
    ids = list(dict.fromkeys(q for L in runs for q in L))
    out = {}
    for q in ids:
        v, conf, rep, first = collections.Counter(), collections.Counter(), {}, {}
        n = sum(1 for L in runs if q in L)                             # runs that reached q (a run cut short has no say)
        for i, L in enumerate(runs):
            r = L.get(q)
            if r is None or r["final"].get("type") == "abstain": continue
            c = agent.canon_of(r["final"]); v[c] += 1; conf[c] += float(r.get("conf") or 0); rep.setdefault(c, r["final"]); first.setdefault(c, i)
        if not v: out[q] = ({"type": "abstain"}, {}); continue
        mean_conf = sum(float(L[q].get("conf") or 0) for L in runs if q in L) / max(n, 1)
        if mean_conf < min_conf: out[q] = ({"type": "abstain"}, {str(k): m for k, m in v.items()}); continue   # --min-conf
        top = max(v, key=lambda c: (v[c], conf[c], -first[c]))
        if v[top] >= min_share * n: out[q] = (rep[top], {str(k): m for k, m in v.items()}); continue
        final = {"type": "abstain"}
        if pens:
            b = next((L[q].get("business_id") for L in runs if q in L), None)
            ans = [(pens[i].get(b), agent.canon_of(L[q]["final"]), L[q]["final"]) for i, L in enumerate(runs)
                   if i < len(pens) and q in L and L[q]["final"].get("type") != "abstain"]
            if ans and all(p is not None for p, _, _ in ans):
                best = min(p for p, _, _ in ans); win = {c for p, c, _ in ans if p == best}
                rest = [p for p, c, _ in ans if p != best]
                if len(win) == 1 and rest and min(rest) - best >= TIE_GAP: final = next(f for p, c, f in ans if p == best)
        out[q] = (final, {str(k): m for k, m in v.items()})
    return out


def main():
    args = sys.argv[1:]
    ms = 0.4; log_dir = ""
    if "--min-share" in args: i = args.index("--min-share"); ms = float(args[i + 1]); del args[i:i + 2]
    if "--log-dir" in args: i = args.index("--log-dir"); log_dir = args[i + 1]; del args[i:i + 2]
    mc = 0.0                                                           # mean confidence over the runs that reached a question
    if "--min-conf" in args: i = args.index("--min-conf"); mc = float(args[i + 1]); del args[i:i + 2]
    out, run_dirs = args[0], args[1:]
    pairs = [(load(r), r) for r in run_dirs]; pairs = [(L, r) for L, r in pairs if L]
    runs = [L for L, _ in pairs]; pens = [penalties(r) for _, r in pairs]
    res = vote(runs, ms, pens if all(pens) else None, mc)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for q, (final, _) in res.items(): f.write(json.dumps({"id": q, **final}, ensure_ascii=False) + "\n")
    os.replace(tmp, out)
    if log_dir and os.path.isdir(log_dir):                             # the per-question logs carry the voted answer
        for q, (final, votes) in res.items():
            p = os.path.join(log_dir, f"{q}.json")
            if os.path.isfile(p):
                try:
                    rec = json.load(open(p, encoding="utf-8")); rec["final_voted"] = final; rec["run_votes"] = votes; rec["runs"] = len(runs)
                    json.dump(rec, open(p, "w", encoding="utf-8"), ensure_ascii=False, default=str, indent=1)
                except Exception: pass
    print(f"vote over {len(runs)} runs: {len(res)} questions, {sum(1 for f, _ in res.values() if f.get('type') == 'abstain')} abstain -> {out}", flush=True)


if __name__ == "__main__":
    main()
