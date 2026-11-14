#!/usr/bin/env python3
"""v2 E -> WJR2: judge records from the systematic contrast pairs of training/contrast_job.py (training shops only).

A contrast pair = (question', program', answer') minimally different from (question, program, answer) on one slot of the program,
both run on the shop's data with different answers and told apart by a blind back-check. Two list records per pair, in the judge's
inference prompt (solver prefix of the shop + judge.question_block): question' with both programs as candidates (label: program')
and question with the same two candidates (label: program); candidate order random. Same label format as WJR1 (judge.label_text,
plan = grammar.auto_plan of the right program). Point records: the wrong program alone for each question (label 0).
Usage: wj_contrast.py <contrast.jsonl>[,...] <out.jsonl> [--mix Process/train_data/p3/v3a_mix.jsonl.gz] [--point-share 0.3] [--seed 11]"""
import os, sys, json, random, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge, grammar, w_judge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs"); ap.add_argument("out")
    ap.add_argument("--mix", default=os.path.join(ROOT, "Process", "train_data", "p3", "v3a_mix.jsonl.gz"))
    ap.add_argument("--point-share", type=float, default=0.3); ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--max-chars", type=int, default=13000)
    a = ap.parse_args(); rnd = random.Random(a.seed)
    P = w_judge.prefixes(a.mix)
    pairs = []
    for f in a.pairs.split(","):
        pairs += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    recs, st = [], {"pairs": 0, "no_prefix": 0, "style": 0, "val": 0}
    for c in pairs:
        if c["slot"][0] == "style": st["style"] += 1; continue
        if c["business_id"] not in P: st["no_prefix"] += 1; continue
        if w_judge.is_val(c["id"].split("~")[0]): st["val"] += 1; continue  # same validation split as WJR1's data
        st["pairs"] += 1
        sysm, biz = P[c["business_id"]]
        A = {"code": c["program"], "answer": c["answer"], "n": 1, "votes": 1}
        B = {"code": c["orig_program"], "answer": c["orig_answer"], "n": 1, "votes": 1}
        for q, right, wrong in ((c["question"], A, B), (c["orig_question"], B, A)):
            cs = [right, wrong]; rnd.shuffle(cs); choice = cs.index(right) + 1
            u = biz + "\n\n" + judge.question_block(q, c["as_of_date"], cs, "list", 0, None)
            if len(sysm) + len(u) > a.max_chars: continue
            recs.append({"id": c["id"], "business_id": c["business_id"], "source": "WJ_list_contrast", "choice": choice,
                         "messages": [{"role": "system", "content": sysm}, {"role": "user", "content": u},
                                      {"role": "assistant", "content": judge.label_text(grammar.auto_plan(right["code"]), choice)}]})
            if rnd.random() < a.point_share:                             # the wrong reading alone: rejected
                u = biz + "\n\n" + judge.question_block(q, c["as_of_date"], [wrong], "point", 0, None)
                recs.append({"id": c["id"], "business_id": c["business_id"], "source": "WJ_point_contrast", "choice": 0,
                             "messages": [{"role": "system", "content": sysm}, {"role": "user", "content": u},
                                          {"role": "assistant", "content": judge.label_text(grammar.auto_plan(right["code"]), 0)}]})
    rnd.shuffle(recs)
    with open(a.out, "w", encoding="utf-8") as w:
        for r in recs: w.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({**st, "records": len(recs)}))


if __name__ == "__main__":
    main()
