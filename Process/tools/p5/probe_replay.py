#!/usr/bin/env python3
"""P5 plan H01 / H02: selection with the hidden-state probe (offline). For each question the candidate answers (probe items) are
scored share + lam * p (p = probe probability of being correct); the best one is answered when (share + lam p) / (1 + lam) >= tau,
else abstained. lam and tau are chosen on the TRAINING items' out-of-fold probabilities (max training PA), then applied unchanged
to the public items; public answers only score the result.
Usage: probe_replay.py <items.jsonl> <probe_probs.jsonl> [--gt data/raw/public_gt.jsonl] [--col p_lr|p_mlp]"""
import sys, json, argparse, itertools
from collections import defaultdict
sys.path.insert(0, "/home/user/ZC/src")
import score


def pick(groups, lam, tau, col):
    out = {}
    for q, g in groups.items():
        best = max(g, key=lambda it: it["share"] + lam * it[col])
        s = (best["share"] + lam * best[col]) / (1 + lam)
        out[q] = best["answer"] if s >= tau else {"type": "abstain"}
    return out


def pa_train(groups, lam, tau, col):
    sc = {"c": 0, "w": 0}
    for q, g in groups.items():
        best = max(g, key=lambda it: it["share"] + lam * it[col]); s = (best["share"] + lam * best[col]) / (1 + lam)
        if s < tau: continue
        sc["c" if best["label"] else "w"] += 1
    return (sc["c"] - sc["w"]) / max(1, len(groups))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("items"); ap.add_argument("probs")
    ap.add_argument("--gt", default="/home/user/ZC/data/raw/public_gt.jsonl"); ap.add_argument("--col", default="p_lr")
    ap.add_argument("--split", default="pub"); a = ap.parse_args()
    P = {r["uid"]: r for r in (json.loads(l) for l in open(a.probs) if l.strip())}
    tr, pu = defaultdict(list), defaultdict(list)
    for it in (json.loads(l) for l in open(a.items, encoding="utf-8") if l.strip()):
        if it["uid"] not in P: continue
        it["p_lr"], it["p_mlp"] = P[it["uid"]]["p_lr"], P[it["uid"]]["p_mlp"]
        (tr if it["split"] == "train" else pu)[it["qid"]].append(it)
    grid = [(l, t) for l in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0) for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)]
    res = sorted(((pa_train(tr, l, t, a.col), l, t) for l, t in grid), reverse=True)
    base = max((pa_train(tr, 0.0, t, a.col), t) for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7))
    print(f"train ({len(tr)} q): vote only best PA {base[0]:.4f} (tau {base[1]}); with probe {a.col}: {res[0][0]:.4f} (lam {res[0][1]}, tau {res[0][2]})")
    gt = score.load(a.gt); ref = {k: v for k, v in gt.items()}
    for name, (l, t) in (("vote only, tau 0.5 (run rule)", (0.0, 0.5)), (f"probe {a.col} lam {res[0][1]} tau {res[0][2]}", (res[0][1], res[0][2]))):
        pr = pick(pu, l, t, a.col); r, _ = score.score(ref, pr)
        print(f"public {name}: PA {r['PA']}  c/w/a {r.get('correct', 0)}/{r.get('wrong', 0)}/{r.get('abstain', 0)}  (questions with candidates: {len(pu)})")


if __name__ == "__main__":
    main()
