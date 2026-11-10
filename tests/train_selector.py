#!/usr/bin/env python3
"""Train / evaluate the learned answer selector (src/selector.py, track M WP-M6) offline on logged samples.
Train: logs of the training shops that are not held out (tr_dev_all) and their paraphrases (tr_dev_para) — never public or
private test. theta chosen on the training logs (PA). Evaluation: replay on other runs' logs (pub_full; held-out pub_ho / tr_ho as
aggregates), compared with the flow's own vote (theta 0.5).
Usage: train_selector.py --train <run_dir>[,<run_dir>] --eval <run_dir>[,...] [--sets pub_full,pub_ho,tr_ho] [--out selector.json]
       [--feats all|inv]   (inv = shares / fractions only, src/selector.FEATS_INV)"""
import os, sys, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import score, selector as SEL, agent
from collections import Counter

HO = ("biz_105", "biz_109", "biz_114", "biz_123", "biz_127", "biz_130")


def ids_of(name):
    p = os.path.join(ROOT, "tests", "splits", name + ".txt")
    return {l.strip() for l in open(p) if l.strip()} if os.path.isfile(p) else set()


def load(run):
    p = os.path.join(run, "logs.jsonl")
    out = {}
    for l in open(p, encoding="utf-8"):
        r = json.loads(l); out[r["id"]] = r
    return out


def vote_decide(r, theta=0.5):
    lg = r.get("samples") if isinstance(r.get("samples"), dict) else {}
    S = lg.get("samples") or []; votes = Counter(); rep = {}
    for s in S:
        if not s.get("ok"): continue
        k = agent.canon_of(s["answer"]); votes[k] += 0.5 if s.get("flags") else 1.0; rep.setdefault(k, s["answer"])
    if not votes: return {"type": "abstain"}
    top = votes.most_common(1)[0][0]
    return rep[top] if votes[top] / max(1, len(S)) >= theta else {"type": "abstain"}


def pa(recs, refs, decide):
    tot = 0.0; ab = 0
    for r in recs:
        ref = refs.get(r["id"].split("~")[0])
        f = decide(r)
        if f.get("type") == "abstain": ab += 1; continue
        try: tot += 1 if score.judge(f, ref) == "correct" else -1
        except Exception: tot -= 1
    return tot / max(1, len(recs)), ab


def main():
    arg = lambda k, d="": sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    refs = {}
    for p in (os.path.join(ROOT, "data", "raw", "public_gt.jsonl"), os.path.join(ROOT, "data", "raw", "train", "train", "answers.jsonl")):
        if os.path.isfile(p): refs.update(score.load(p))
    d3 = os.path.join(ROOT, "tests", "splits", "d3sel.answers.jsonl")       # teacher questions of the training shops (D3, not
    if os.path.isfile(d3): refs.update(score.load(d3))                    # used to train the final soup: out-of-sample)
    tr_ids = ids_of("tr_dev_all") | ids_of("tr_dev_para") | ids_of("d3sel")
    train = []
    for run in arg("--train").split(","):
        train += [r for i, r in load(run).items() if i in tr_ids and r.get("business_id") not in HO]
    print(f"train: {len(train)} question logs (training shops not held out + their paraphrases)")
    sel = SEL.Selector(feats=SEL.FEATS_INV if arg("--feats", "all") == "inv" else SEL.FEATS).fit(train, refs, score.judge)
    best = max(((pa(train, refs, lambda r, t=t: sel.decide(r, t)[0])[0], t) for t in np.arange(0.2, 0.81, 0.05)))
    sel.theta = float(best[1])
    print(f"theta {sel.theta:.2f} (train PA {best[0]:.4f}; vote 0.5: {pa(train, refs, vote_decide)[0]:.4f})")
    print("weights:", {k: round(float(w), 2) for k, w in zip(["bias"] + sel.feats, sel.w)})
    sets = arg("--sets", "pub_full,pub_ho,tr_ho").split(",")
    for run in [x for x in arg("--eval").split(",") if x]:
        L = load(run); line = []
        for s in sets:
            ids = ids_of(s); recs = [r for i, r in L.items() if i in ids]
            if not recs: continue
            a, ab = pa(recs, refs, vote_decide); b, ab2 = pa(recs, refs, lambda r: sel.decide(r)[0])
            c, ab3 = pa(recs, refs, sel.rerank)
            line.append(f"{s}: vote {a:.4f} (ab {ab}) | selector {b:.4f} (ab {ab2}) | rerank {c:.4f} (ab {ab3})")
        print(f"== {run.rstrip('/').split('/')[-1]}: " + " | ".join(line))
    if arg("--out"): sel.save(arg("--out"))


if __name__ == "__main__":
    main()
