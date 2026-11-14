#!/usr/bin/env python3
"""Plan v2 (handoff/04 §3.G source 1): every training question with a CHECKED program (its answer matches the organisers'
training answer: Process/train_data/p3/d1_all.jsonl.gz) -> one input of src/lrl_gen.py --mode newterm (the model invents a
new business term + a rulebook row defining it + the question reworded with it). set = "ho" for the 6 held-out training
shops (tests/splits/tr_ho*.txt: never trained on, kept to measure the tool writer), else "dev".
Usage: wt_newterm_in.py [--d1 Process/train_data/p3/d1_all.jsonl.gz] [--out training/data/out/wt/newterm_in.jsonl]"""
import os, sys, json, gzip, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def holdout_shops():
    ids = {l.strip().split("~")[0] for f in ("tr_ho.txt", "tr_ho400c.txt") for l in open(os.path.join(ROOT, "tests", "splits", f)) if l.strip()}
    qs = os.path.join(ROOT, "data", "raw", "train", "train", "questions.jsonl")
    return sorted({q["business_id"] for q in map(json.loads, open(qs, encoding="utf-8")) if q["id"] in ids})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d1", default=os.path.join(ROOT, "Process", "train_data", "p3", "d1_all.jsonl.gz"))
    ap.add_argument("--out", default=os.path.join(ROOT, "training", "data", "out", "wt", "newterm_in.jsonl"))
    a = ap.parse_args()
    ho = set(holdout_shops()); n = {"dev": 0, "ho": 0}
    with open(a.out, "w", encoding="utf-8") as w:
        for l in gzip.open(a.d1, "rt", encoding="utf-8"):
            r = json.loads(l)
            if not r.get("program") or r.get("answer") is None: continue
            s = "ho" if r["business_id"] in ho else "dev"; n[s] += 1
            w.write(json.dumps({"id": r["id"], "business_id": r["business_id"], "question": r["question"], "as_of_date": r["as_of_date"],
                                "program": r["program"], "answer": r["answer"], "set": s}, ensure_ascii=False) + "\n")
    print(f"held-out shops {sorted(ho)}; inputs {n} -> {a.out}")


if __name__ == "__main__":
    main()
