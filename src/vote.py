#!/usr/bin/env python3
"""Vote across several complete runs of the agent (multi-run ensemble of the final system).

Each run is an independent draw of the sampled mappings and programs; the per-question plurality over runs is more
accurate and more stable than any single run (train, 4 runs: PA 0.980 vs 0.928-0.969 for the single runs, wrong answers
19 vs 29-82). An abstaining run gives no vote; the top answer needs at least `min_share` of the runs (0.4: 2 of 4,
2 of 3, 1 of 2) of the runs that reached the question; ties are broken by the summed confidence, then by run order.
Train, every subset of the 4 runs: 2 runs 0.956-0.972 (about the mean of the two), 3 runs 0.976-0.981, 4 runs 0.980.
The submission is written atomically.
Usage: vote.py <out_submission.jsonl> <run_dir> ... [--min-share 0.4] [--log-dir DIR]"""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent


def load(run):
    p = os.path.join(run, "logs.jsonl")
    return {r["id"]: r for r in map(json.loads, open(p, encoding="utf-8"))} if os.path.isfile(p) else {}


def vote(runs, min_share=0.4):
    """[logs of each run] -> {qid: (final answer, {canon answer: votes})} for every question of any run."""
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
        top = max(v, key=lambda c: (v[c], conf[c], -first[c]))
        out[q] = (rep[top] if v[top] >= min_share * n else {"type": "abstain"}, {str(k): m for k, m in v.items()})
    return out


def main():
    args = sys.argv[1:]
    ms = 0.4; log_dir = ""
    if "--min-share" in args: i = args.index("--min-share"); ms = float(args[i + 1]); del args[i:i + 2]
    if "--log-dir" in args: i = args.index("--log-dir"); log_dir = args[i + 1]; del args[i:i + 2]
    out, run_dirs = args[0], args[1:]
    runs = [L for L in (load(r) for r in run_dirs) if L]
    res = vote(runs, ms)
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
