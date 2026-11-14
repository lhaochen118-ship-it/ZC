#!/usr/bin/env python3
"""Plan 5 W3 offline scoring (docs/18): a judge run's verdicts (src/judge.py verdicts.jsonl: the candidates of every question
and the judge's parsed choices) scored against the reference answers under a grid of decision policies (judge.decide), next to
the plain vote on the same candidates. No GPU: the policy is chosen here, then written into the flow's config.
Usage: replay_judge.py <verdicts.jsonl>[,<more>...] [--ref data/raw/public_gt.jsonl] [--show] [--val]   (--val: grid on training rows)"""
import os, sys, json, itertools
from collections import Counter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import score, judge


def load_ref(path):
    R = score.load(path)
    return R


def verdict(fin, ref):
    if not fin or fin.get("type") == "abstain": return "abstain"
    try: return score.judge(fin, ref)
    except Exception: return "wrong"


def pa(rows, R, **pol):
    c = Counter(); how = Counter(); detail = []
    for r in rows:
        ref = R.get(r["id"].split("~")[0])
        if ref is None: continue
        ch = r["choices"]
        if r["mode"] == "point": ch = {int(k): v for k, v in ch.items()}
        fin, conf, h = judge.decide(r["cands"], r["n_samples"], ch, r["mode"], **pol)
        v = verdict(fin, ref); c[v] += 1; how[(h, v)] += 1; detail.append((r["id"], v, h))
    n = max(1, sum(c.values()))
    return (c["correct"] - c["wrong"]) / n, c, how, detail


def main():
    files = sys.argv[1].split(",")
    ref = sys.argv[sys.argv.index("--ref") + 1] if "--ref" in sys.argv else os.path.join(ROOT, "data", "raw", "public_gt.jsonl")
    R = load_ref(ref)
    for f in files:
        rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        if not rows: continue
        mode = rows[0]["mode"]
        flat = [c for r in rows for c in (r["choices"] if mode == "list" else [x for v in r["choices"].values() for x in v])]
        print(f"== {f}: {len(rows)} questions, mode {mode}, judge samples {len(flat)}, parsed {sum(1 for c in flat if c is not None)}; "
              f"choices {dict(Counter(map(str, flat)).most_common(8))}")
        base = {}
        for den in ("all", "valid"):
            p, c, _, d = pa([dict(r, choices=([] if mode == "list" else {})) for r in rows], R, denom=den)
            base[den] = (p, d); print(f"   vote (denom {den}): PA {p:.4f}  {c['correct']}/{c['wrong']}/{c['abstain']}")
        best = []
        for tau, den, ua, ku in itertools.product((0.5, 0.6, 0.8, 1.0), ("all", "valid"), (True, False), (0.0, 0.8, 1.0)):
            p, c, how, d = pa(rows, R, tau=tau, denom=den, u_answers=ua, keep_unanimous=ku)
            best.append((p, tau, den, ua, ku, c, how, d))
        best.sort(key=lambda x: -x[0])
        for p, tau, den, ua, ku, c, how, d in best[:6]:
            print(f"   judge tau {tau} denom {den} U->{'unanswerable' if ua else 'abstain'} keep_unanimous {ku}: PA {p:.4f}  {c['correct']}/{c['wrong']}/{c['abstain']}")
        p, tau, den, ua, ku, c, how, d = best[0]
        b = dict((i, v) for i, v, _ in base[den][1])
        fixed = [(i, v, h) for i, v, h in d if v == "correct" and b.get(i) != "correct"]
        broke = [(i, v, h) for i, v, h in d if v != "correct" and b.get(i) == "correct"]
        print(f"   best vs vote({den}): fixed {len(fixed)}, broken {len(broke)}; outcomes by path: " +
              ", ".join(f"{h}:{v}={n}" for (h, v), n in sorted(how.items())))
        if "--show" in sys.argv:
            for tag, L in (("fixed", fixed), ("broken", broke)):
                for i, v, h in L: print(f"     {tag} {i} -> {v} via {h}")


if __name__ == "__main__":
    main()
