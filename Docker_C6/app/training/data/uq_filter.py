#!/usr/bin/env python3
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
