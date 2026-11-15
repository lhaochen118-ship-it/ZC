#!/usr/bin/env python3
"""P5 plan A01 / A02 (+ V-B3 'vote only'): replay the solver's vote offline from its logged samples (public answers used to SCORE only).
A01 fixed K: the first K samples (sample index j) vote; A02 adaptive: k_fast samples, more (up to k_max) only when they disagree
(top share < agree), as the pipeline does. Same rule as hybrid.finish: flagged samples weigh 0.5, share = top / samples, abstain < theta.
Usage: replay_k.py <solver_logs.jsonl[.gz]> [--gt data/raw/public_gt.jsonl] [--theta 0.5]"""
import sys, json, gzip, argparse
from collections import Counter
sys.path.insert(0, "/home/user/ZC/src")
import score, agent


def rd(p):
    with (gzip.open if p.endswith(".gz") else open)(p, "rt", encoding="utf-8") as f:
        for l in f:
            if l.strip(): yield json.loads(l)


def decide(smp, theta):
    votes, rep = Counter(), {}
    for s in smp:
        if not s.get("ok") or not isinstance(s.get("answer"), dict): continue
        c = agent.canon_of(s["answer"]); votes[c] += 0.5 if s.get("flags") else 1.0; rep.setdefault(c, s["answer"])
    if not votes: return {"type": "abstain"}, 0.0
    top, v = votes.most_common(1)[0]; conf = v / max(1, len(smp))
    return (rep[top] if conf >= theta else {"type": "abstain"}), conf


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("logs"); ap.add_argument("--gt", default="/home/user/ZC/data/raw/public_gt.jsonl")
    ap.add_argument("--theta", type=float, default=0.5); a = ap.parse_args()
    gt = score.load(a.gt); rows = list(rd(a.logs))
    pol = {f"K{k}": ("fixed", k) for k in (1, 2, 4, 8, 16)}
    pol.update({"adapt4->8": ("adapt", 4, 8, 1.0), "adapt4->16 (run)": ("adapt", 4, 16, 1.0), "adapt2->8": ("adapt", 2, 8, 1.0),
                "adapt4->16 agree.75": ("adapt", 4, 16, 0.75)})
    preds = {p: {} for p in pol}; used = Counter(); logged = {}
    for r in rows:
        smp = sorted((r.get("samples") or {}).get("samples") or [], key=lambda s: s.get("j", 0))
        logged[r["id"]] = r.get("final")
        for name, p in pol.items():
            if p[0] == "fixed": sub = smp[:p[1]]
            else:
                sub = smp[:p[1]]; f, c = decide(sub, a.theta)
                if c < p[3]: sub = smp[:p[2]]
            preds[name][r["id"]] = decide(sub, a.theta)[0]; used[name] += len(sub)
    n = len(rows); ref = {k: v for k, v in gt.items() if k in preds["K1"]} if len(rows) < len(gt) else gt
    base, _ = score.score(ref, {k: v for k, v in logged.items()})
    print(f"{n} questions; logged final (vote of the run): PA {base['PA']}  ({base.get('correct', 0)}/{base.get('wrong', 0)}/{base.get('abstain', 0)})")
    for name in pol:
        r, _ = score.score(ref, preds[name])
        print(f"  {name:22s} PA {r['PA']:.4f}  c/w/a {r.get('correct', 0)}/{r.get('wrong', 0)}/{r.get('abstain', 0)}  samples/q {used[name] / max(1, n):.2f}")


if __name__ == "__main__":
    main()
