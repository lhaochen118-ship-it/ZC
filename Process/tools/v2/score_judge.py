#!/usr/bin/env python3
"""v2: score judge verdicts (src/judge.py verdicts.jsonl[.gz]) of several solver logs and variants against the public reference
(scoring only): parse rate, PA of the plain vote, of each variant alone and of ensembles (judge.ensemble, margin grid).
Variant dirs are looked up under each root as <root>/<log>/<variant>/verdicts.jsonl[.gz] (several roots: later ones fill in
variants the earlier ones lack, e.g. new '+c' runs next to plan-5 point_think verdicts).
Usage: score_judge.py <root>[,<root>...] --logs f5a,f5b,pva3k16,g1v3apf2 --combos "list_c,point_c,point_think;list,point,point_think"
       [--margins 0.5,0.6,0.7] [--ref data/raw/public_gt.jsonl]"""
import os, sys, json, gzip, argparse
from collections import Counter
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "src"))
import judge, score


def load(p):
    op = gzip.open if p.endswith(".gz") else open
    return {r["id"]: r for r in map(json.loads, op(p, "rt", encoding="utf-8")) if r}


def find(roots, log, v):
    for r in roots:
        for f in ("verdicts.jsonl", "verdicts.jsonl.gz"):
            for p in (os.path.join(r, log, v, f), os.path.join(r, "judge", log, v, f)):
                if os.path.isfile(p): return p
    return None


def pa(final, R, ids):
    c = Counter()
    for i in ids:
        f = final.get(i)
        if f is None or f.get("type") == "abstain": c["abstain"] += 1; continue
        try: c[score.judge(f, R[i])] += 1
        except Exception: c["wrong"] += 1
    n = len(ids); return round((c["correct"] - c["wrong"]) / max(1, n), 4), c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots"); ap.add_argument("--logs", required=True); ap.add_argument("--combos", default="")
    ap.add_argument("--margins", default="0.5,0.6,0.7"); ap.add_argument("--ref", default=os.path.join(ROOT, "data", "raw", "public_gt.jsonl"))
    ap.add_argument("--singles", default="list_c,point_c,list,point,point_think,list_think")
    a = ap.parse_args(); R = score.load(a.ref); roots = a.roots.split(",")
    for log in a.logs.split(","):
        V = {}
        for v in set(a.singles.split(",")) | {x for c in a.combos.split(";") for x in c.split(",") if x}:
            p = find(roots, log, v)
            if p: V[v] = load(p)
        if not V: print(log, "no verdicts"); continue
        any_v = next(iter(V.values())); ids = [i for i in any_v if i in R]
        vote = {i: judge.vote_decision(r["cands"], r["n_samples"], 0.5, "valid")[0] for i, r in any_v.items()}
        line = [f"{log}: vote {pa(vote, R, ids)[0]}"]
        for v, rows in sorted(V.items()):
            req = sum(len(r["choices"]) if isinstance(r["choices"], list) else sum(len(x) for x in r["choices"].values()) for r in rows.values())
            ok = sum(sum(1 for c in (r["choices"] if isinstance(r["choices"], list) else [y for x in r["choices"].values() for y in x]) if c is not None) for r in rows.values())
            fin = {i: f for i, (f, c, h) in judge.ensemble({v: rows}, 0.5, "valid").items()}
            line.append(f"{v} parse {ok}/{req} PA {pa(fin, R, ids)[0]}")
        print(" | ".join(line))
        for combo in [c for c in a.combos.split(";") if c]:
            vs = [x for x in combo.split(",") if x in V]
            if len(vs) < 2: continue
            for m in map(float, a.margins.split(",")):
                fin = {i: f for i, (f, c, h) in judge.ensemble({v: V[v] for v in vs}, m, "valid").items()}
                print(f"   {'+'.join(vs)} m{m}: {pa(fin, R, ids)}")


if __name__ == "__main__":
    main()
