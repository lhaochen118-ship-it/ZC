#!/usr/bin/env python3
"""Scores of a hybrid-flow run per evaluation set (docs/08 §3): PA, right / wrong / abstain, seconds per question, stop kinds.
A paraphrase '<id>~p<k>' or '<id>~accentfree' is scored against the reference answer of <id> (public: data/raw/public_gt.jsonl,
used only here; train: answers.jsonl).
Usage: score_sets.py <run_dir (submission.jsonl + logs.jsonl)> [--sets quick400,pub_full,...] [--json out.json]"""
import os, sys, json
from collections import Counter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import score

SETS = ("quick400", "pub_full", "pub_k0", "pub_ho", "tr_ho", "pub_noind", "pub_noacc")


def refs():
    R = {}
    for p in (os.path.join(ROOT, "data", "raw", "public_gt.jsonl"), os.path.join(ROOT, "data", "raw", "train", "train", "answers.jsonl")):
        if os.path.isfile(p): R.update(score.load(p))
    return R


def main():
    run = sys.argv[1]
    sets = sys.argv[sys.argv.index("--sets") + 1].split(",") if "--sets" in sys.argv else list(SETS)
    sub_p = run if run.endswith(".jsonl") else os.path.join(run, "submission.jsonl")
    sub = score.load(sub_p)
    logs = {}
    lp = os.path.join(os.path.dirname(sub_p), "logs.jsonl")
    if os.path.isfile(lp):
        for l in open(lp, encoding="utf-8"):
            try: r = json.loads(l); logs[r["id"]] = r
            except Exception: pass
    R = refs(); table = {}
    for s in sets:
        f = os.path.join(ROOT, "tests", "splits", s + ".txt")
        if not os.path.isfile(f): continue
        ids = [l.strip() for l in open(f) if l.strip() and l.strip() in sub]
        if not ids: continue
        ref = {i: R[i.split("~")[0]] for i in ids if i.split("~")[0] in R}
        res, _ = score.score(ref, {i: sub[i] for i in ref}, by="type")
        secs = [logs[i].get("elapsed_s", 0) for i in ref if i in logs]
        stops = Counter((logs[i].get("samples") or {}).get("stop", "?") for i in ref if i in logs and isinstance(logs[i].get("samples"), dict))
        table[s] = {"n": res["n"], "PA": round(res["PA"], 4), "correct": res["correct"], "wrong": res["wrong"], "abstain": res.get("abstain", 0),
                    "s_per_q": round(sum(secs) / max(1, len(secs)), 2), "stops": dict(stops)}
    print(f"{'set':10s} {'n':>5s} {'PA':>7s} {'right':>6s} {'wrong':>6s} {'abst':>5s} {'s/q':>6s}  stops")
    for s, t in table.items():
        print(f"{s:10s} {t['n']:5d} {t['PA']:7.4f} {t['correct']:6d} {t['wrong']:6d} {t['abstain']:5d} {t['s_per_q']:6.2f}  {t['stops']}")
    if "--json" in sys.argv: json.dump(table, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
