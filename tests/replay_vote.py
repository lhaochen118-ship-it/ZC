#!/usr/bin/env python3
"""Offline replay of the hybrid flow's last step (vote -> confidence -> abstain) on the samples a GPU run logged, to measure a
change of that step without a GPU (docs/12_PLAN2.md WP-P2-03). Variants:
  - fmt: answer format normalised to the hinted answer type before the vote (src/hybrid.fmt_fix: a one-element list for a
    'number' question, an ISO date string for a 'date' question);
  - den: the confidence denominator: all samples launched (current), or valid samples + lam x failed ones;
  - theta: the abstain threshold.
Scores: PA on the dev sets (pub_full, pub_k0) and, as aggregates only, on the held-out sets (pub_ho; tr_ho when logged).
  --hints F: hints of the new code (tests/hint_audit.py --out) for the original public questions: answer type + postcheck
    flags recomputed with them (the samples stay those of the old run).
Usage: replay_vote.py <run_dir>[,<run_dir>...] [--sets pub_full,pub_k0,pub_ho,tr_ho] [--hints hints.jsonl]"""
import os, sys, json
from collections import Counter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import score, agent, hybrid


def ids_of(name):
    p = os.path.join(ROOT, "tests", "splits", name + ".txt")
    return {l.strip() for l in open(p) if l.strip()} if os.path.isfile(p) else set()


def decide(samples, hint_type, fmt, lam, theta, reflag=None):
    """reflag: (question key, hinted type) -> the postcheck flags recomputed with a new hint (--hints), else the logged ones."""
    votes, rep = Counter(), {}
    for s in samples:
        if not s.get("ok"): continue
        ans = hybrid.fmt_fix(s["answer"], hint_type) if fmt else s["answer"]
        flags = s.get("flags")
        if reflag is not None:
            import tools
            flags = tools.postcheck(ans, reflag[1], None, reflag[0], hybrid.top_k_asked(reflag[0]))
        c = agent.canon_of(ans); votes[c] += 0.5 if flags else 1.0; rep.setdefault(c, ans)
    if not votes: return {"type": "abstain"}, 0.0
    n_ok = sum(1 for s in samples if s.get("ok")); n_bad = len(samples) - n_ok
    n = max(1, len(samples)) if lam is None else max(1.0, n_ok + lam * n_bad)
    top = votes.most_common(1)[0][0]; conf = votes[top] / n
    return (rep[top] if conf >= theta else {"type": "abstain"}), conf


def main():
    runs = sys.argv[1].split(",")
    sets = (sys.argv[sys.argv.index("--sets") + 1] if "--sets" in sys.argv else "pub_full,pub_k0,pub_ho,tr_ho").split(",")
    R = {}
    for p in (os.path.join(ROOT, "data", "raw", "public_gt.jsonl"), os.path.join(ROOT, "data", "raw", "train", "train", "answers.jsonl")):
        if os.path.isfile(p): R.update(score.load(p))
    S = {s: ids_of(s) for s in sets}
    H = {}
    if "--hints" in sys.argv:                                            # hint_audit.py --out of the new code (public only)
        for l in open(sys.argv[sys.argv.index("--hints") + 1], encoding="utf-8"):
            r = json.loads(l); H[r["id"]] = r["hints"]
    variants = [("current", False, None, 0.5)] + [(f"fmt den={lam} th={th}", True, lam, th) for lam in (None, 0.0, 0.25, 0.5) for th in (0.4, 0.5, 0.6)]
    for run in runs:
        recs = {}
        for l in open(os.path.join(run, "logs.jsonl"), encoding="utf-8"):
            r = json.loads(l); recs[r["id"]] = r
        print(f"== {run}")
        for name, fmt, lam, th in variants:
            out = []
            for s in sets:
                ids = [i for i in recs if i in S[s]]
                if not ids: continue
                pa = 0.0; ab = 0
                for i in ids:
                    lg = recs[i]["samples"] if isinstance(recs[i].get("samples"), dict) else {}
                    ht = (lg.get("hints") or {}).get("answer_type"); rf = None
                    if i in H and fmt:
                        import dm
                        ht = H[i].get("answer_type"); rf = (dm.key(H[i].get("clean") or recs[i]["question"]), ht)
                    fin, _ = decide(lg.get("samples") or [], ht, fmt, lam, th, rf)
                    ref = R.get(i.split("~")[0])
                    if ref is None: continue
                    if fin.get("type") == "abstain": ab += 1; continue
                    try: pa += 1 if score.judge(fin, ref) == "correct" else -1
                    except Exception: pa -= 1
                out.append(f"{s} {pa / len(ids):.4f} (abst {ab})")
            print(f"  {name:28s} " + " | ".join(out))


if __name__ == "__main__":
    main()
