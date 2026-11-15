#!/usr/bin/env python3
"""v2 data augmentation (handoff/07 U1): keep the model-written unit questions whose generated program S agrees with.
S (V3A x 0.70, mode H) answered every question of the uqchk sets with the teacher's functions in # SHOP FUNCTIONS (configs/flow_uqchk.json);
a pair is kept when S's final answer equals the answer of the generated program (src/score.judge, the competition's comparison).
Disagreement means the question's period, the function's meaning or the question itself is off (15 % of a 30-pair sample had a wrong
period, e.g. 'Chủ nhật tuần trước' -> last_week). Two model paths agreeing is the label check; no hand-written rule decides.
Usage: uq_filter.py <uqchk submission.jsonl>[,<more>] <uq dir> <out dir>"""
import os, sys, json, glob
from collections import Counter
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
import score


def main():
    subs = {}
    for f in sys.argv[1].split(","):
        for l in open(f, encoding="utf-8"):
            if l.strip(): r = json.loads(l); subs[r["id"]] = r
    Q = [json.loads(l) for f in sorted(glob.glob(os.path.join(sys.argv[2], "cards_unitq_*.jsonl"))) for l in open(f, encoding="utf-8") if l.strip()]
    os.makedirs(sys.argv[3], exist_ok=True)
    st, keep = Counter(), []
    for i, q in enumerate(Q):
        a = subs.get(f"uqchk~{i:04d}")
        if a is None: st["no answer"] += 1; continue
        if a.get("type") == "abstain": st["S abstained"] += 1; continue
        try: ok = score.judge(a, q["answer"]) == "correct"
        except Exception: ok = False
        st["agree" if ok else "disagree"] += 1
        if ok: keep.append(q)
    with open(os.path.join(sys.argv[3], "cards_unitq_f.jsonl"), "w", encoding="utf-8") as w:
        for q in keep: w.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(json.dumps({"pairs": len(Q), "kept": len(keep), **st}, ensure_ascii=False))


if __name__ == "__main__":
    main()
