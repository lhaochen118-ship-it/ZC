#!/usr/bin/env python3
import os, sys, json, glob, random, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import score


def refs():
    out = score.load(os.path.join(common.ROOT, "data", "raw", "train", "train", "answers.jsonl"))
    p = os.path.join(common.ROOT, "tests", "splits", "d3sel.answers.jsonl")
    if os.path.isfile(p): out.update(score.load(p))
    return out


def chosen(r):
    for s in ((r.get("samples") or {}).get("samples") or []):
        if s.get("ok") and s.get("answer") == r.get("final"):
            return s["turns"][-1].get("code"), s.get("answer")
    return None, r.get("final")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("logs"); ap.add_argument("--max", type=int, default=1500); ap.add_argument("--seed", type=int, default=20261113)
    a = ap.parse_args()
    R = refs()
    gold = {r["id"]: r["code"] for r in common.jl(os.path.join(common.SRC, "refs", "answer_examples.jsonl"))}
    recs = []
    for f in a.logs.split(","):
        recs += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    for r in recs:
        base = r["id"].split("~")[0]
        if base in R and base not in gold:
            for s in ((r.get("samples") or {}).get("samples") or []):
                if s.get("ok") and score.judge(s.get("answer"), R[base]) == "correct":
                    gold[base] = s["turns"][-1].get("code"); break
    cases, seen = [], set()
    for r in recs:
        base = r["id"].split("~")[0]
        if r.get("business_id") in common.TR_HO or base not in R or r["id"] in seen: continue
        v = score.judge(r.get("final"), R[base])
        if v == "correct": continue
        seen.add(r["id"]); prog, ans = chosen(r)
        cases.append({"id": r["id"], "business_id": r["business_id"], "question": r["question"], "as_of_date": r["as_of_date"],
                      "verdict": v, "program": prog, "answer": ans, "reference": {"type": R[base].get("type"), "answer": R[base].get("answer")},
                      "gold_program": gold.get(base)})
    random.Random(a.seed).shuffle(cases); cases = cases[:a.max]
    common.write_jl(a.out, cases)
    print(f"lrl cases: {len(cases)} (wrong {sum(c['verdict'] == 'wrong' for c in cases)}, abstain {sum(c['verdict'] == 'abstain' for c in cases)}, "
          f"with a correct program {sum(bool(c['gold_program']) for c in cases)}) from {len(recs)} logged questions -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
