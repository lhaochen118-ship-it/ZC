#!/usr/bin/env python3
"""Build the retrieval library for agent.py --shots from agent runs on the TRAINING split (Rule 6: train only):
for every training question and path (A = canonical tables C, B = raw tables T), keep one program whose executed answer
is correct against answers.jsonl (the shortest one: usually the cleanest). Output rows: {business_id, question, type, path, code}.

Usage: build_shots.py <train_split_dir> <out.jsonl> <logs.jsonl> [<logs.jsonl> ...]"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import judge


def main():
    split, out, logs = sys.argv[1], sys.argv[2], sys.argv[3:]
    refs = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(split, "answers.jsonl"), encoding="utf-8")}
    best = {}
    for f in logs:
        for l in open(f, encoding="utf-8"):
            L = json.loads(l); ref = refs.get(L["id"])
            if ref is None: continue
            for key, atts in (L.get("samples") or {}).items():
                ok = [a for a in atts if a.get("ok")]
                if not ok or judge(ok[-1]["answer"], ref) != "correct": continue
                code = ok[-1]["code"].strip(); k = (L["id"], key[-1])
                if k not in best or len(code) < len(best[k]["code"]):
                    best[k] = {"business_id": L["business_id"], "question": L["question"], "type": ref["type"], "path": key[-1], "code": code}
    with open(out, "w", encoding="utf-8") as w:
        for i in sorted(best): w.write(json.dumps(best[i], ensure_ascii=False) + "\n")
    print(f"{len(best)} examples ({sum(1 for k in best if k[1] == 'A')} path A) covering {len({k[0] for k in best})} / {len(refs)} questions -> {out}")


if __name__ == "__main__":
    main()
