#!/usr/bin/env python3
"""v2: PA of the held-out sets (pub_ho400c, tr_ho400c, pub_hoind, tr_newterm) from judge verdicts of a held-out solver log judged in two
halves (soup_job --runner judge on split public_test -> <root>/<pub>/judge/<variant>/, on split train -> <root>/<tr>/judge/<variant>/):
plain vote vs the ensemble policy (judge.ensemble, margin). References: public_gt (scoring only) + training answers.
Usage: score_heldout_judge.py <root> [--pub jh0b_pub --tr jh0b_tr] [--variants list,point,point_think] [--margin 0.5]"""
import os, sys, json, glob, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "src"))
import judge, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--pub", default="jh0b_pub"); ap.add_argument("--tr", default="jh0b_tr")
    ap.add_argument("--variants", default="list,point,point_think"); ap.add_argument("--margin", type=float, default=0.5)
    a = ap.parse_args()
    R = {}
    for f in ("data/raw/public_gt.jsonl", "data/raw/train/train/answers.jsonl"): R.update(score.load(os.path.join(ROOT, f)))
    ref = lambda i: R.get(i) or R.get(i.split("~")[0])
    V = {}
    for side, pub in ((a.pub, True), (a.tr, False)):
        for v in a.variants.split(","):
            ps = glob.glob(os.path.join(a.root, side, "**", v, "verdicts.jsonl"), recursive=True)
            if not ps: continue
            for r in map(json.loads, open(ps[0], encoding="utf-8")):
                if r["cands"] and r["business_id"].startswith("dev") == pub: V.setdefault(v, {})[r["id"]] = r
    ens = {i: f for i, (f, c, h) in judge.ensemble(V, a.margin, "valid").items()}
    any_v = next(iter(V.values()))
    vote = {i: judge.vote_decision(r["cands"], r["n_samples"], 0.5, "valid")[0] for i, r in any_v.items()}
    def pa(fin, ids):
        c = 0
        for i in ids:
            f = fin.get(i)
            if f is None or f.get("type") == "abstain": continue
            c += 1 if score.judge(f, ref(i)) == "correct" else -1
        return round(c / len(ids), 4)
    for s in ("pub_ho400c", "tr_ho400c", "pub_hoind", "tr_newterm"):
        ids = [l.strip() for l in open(os.path.join(ROOT, "tests", "splits", s + ".txt")) if l.strip()]
        print(f"{s}: vote {pa(vote, ids)} ensemble {pa(ens, ids)} (missing {sum(1 for i in ids if i not in vote)})")


if __name__ == "__main__":
    main()
