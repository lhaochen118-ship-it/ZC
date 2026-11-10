#!/usr/bin/env python3
"""M7 (docs/12_PLAN2.md §6, several generators instead of one merged soup): offline pooling of the samples several runs logged
for the same questions (one run per generator: the M0 soup, WF members alone, ...), then the flow's own vote (weights 0.5 for a
flagged sample, abstain below theta of all samples launched) and the learned selector (src/selector.py) on the pooled samples.
Scores PA on pub_full and, as aggregates only, on the held-out sets (pub_ho, tr_ho). No GPU.
Usage: pool_gens.py <run_dir>[,<run_dir>...] [--combos "a+b,a+c,a+b+c+d"] [--sets pub_full,pub_ho,tr_ho]
       [--selector src/refs/selector/selector_m0.json] [--theta 0.5]
Runs are named a, b, c ... in the order given; --combos picks the pools (default: each run alone, then all of them)."""
import os, sys, json, glob
from collections import Counter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "tests"))
import score, agent, selector as SEL


def ids_of(name):
    p = os.path.join(ROOT, "tests", "splits", name + ".txt")
    return {l.strip() for l in open(p) if l.strip()} if os.path.isfile(p) else set()


def load(run):
    ps = [run] if run.endswith(".jsonl") else sorted(glob.glob(os.path.join(run, "**", "logs.jsonl"), recursive=True))
    out = {}
    for p in ps:
        for l in open(p, encoding="utf-8"):
            if l.strip():
                r = json.loads(l); out.setdefault(r["id"], r)
    return out


def pool(recs):
    """one record whose samples are the samples of every generator (hints of the first one)."""
    r0 = recs[0]; lg0 = r0.get("samples") if isinstance(r0.get("samples"), dict) else {}
    S = []
    for r in recs:
        lg = r.get("samples") if isinstance(r.get("samples"), dict) else {}
        S += lg.get("samples") or []
    return dict(r0, samples=dict(lg0, samples=S))


def vote(r, theta):
    lg = r.get("samples") if isinstance(r.get("samples"), dict) else {}
    S = lg.get("samples") or []; v = Counter(); rep = {}
    for s in S:
        if not s.get("ok"): continue
        k = agent.canon_of(s["answer"]); v[k] += 0.5 if s.get("flags") else 1.0; rep.setdefault(k, s["answer"])
    if not v: return {"type": "abstain"}
    top = v.most_common(1)[0][0]
    return rep[top] if v[top] / max(1, len(S)) >= theta else {"type": "abstain"}


def pa(recs, refs, decide):
    tot = ab = 0
    for r in recs:
        ref = refs.get(r["id"].split("~")[0]); f = decide(r)
        if f.get("type") == "abstain": ab += 1; continue
        try: tot += 1 if score.judge(f, ref) == "correct" else -1
        except Exception: tot -= 1
    return tot / max(1, len(recs)), ab


def main():
    arg = lambda k, d="": sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    runs = [x for x in sys.argv[1].split(",") if x]; L = [load(r) for r in runs]
    names = "abcdefghij"[: len(runs)]
    combos = [c.split("+") for c in arg("--combos").split(",") if c] if arg("--combos") else [[n] for n in names] + ([list(names)] if len(runs) > 1 else [])
    refs = {}
    for p in (os.path.join(ROOT, "data", "raw", "public_gt.jsonl"), os.path.join(ROOT, "data", "raw", "train", "train", "answers.jsonl")):
        if os.path.isfile(p): refs.update(score.load(p))
    sel = SEL.Selector.load(arg("--selector", os.path.join(ROOT, "src", "refs", "selector", "selector_m0.json")))
    th = float(arg("--theta", "0.5"))
    for n, r in zip(names, runs): print(f"{n} = {r}")
    for c in combos:
        idx = [names.index(x) for x in c]; line = []
        for s in arg("--sets", "pub_full,pub_ho,tr_ho").split(","):
            ids = ids_of(s)
            common = [i for i in ids if all(i in L[k] for k in idx)]
            if not common: continue
            P = [pool([L[k][i] for k in idx]) for i in sorted(common)]
            a, ab = pa(P, refs, lambda r: vote(r, th)); b, ab2 = pa(P, refs, lambda r: sel.decide(r)[0])
            line.append(f"{s} ({len(common)}): vote {a:.4f} (ab {ab}) | sel {b:.4f} (ab {ab2})")
        print(f"[{'+'.join(c)}] " + " | ".join(line), flush=True)


if __name__ == "__main__":
    main()
