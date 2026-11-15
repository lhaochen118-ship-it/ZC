#!/usr/bin/env python3
"""combine New7 item files into one GPU-pass file with mode+tag-prefixed ids: new7_combine.py <out.jsonl> <tag> [<tag> ...]"""
import sys, json, os
D = "/home/user/ZC/training/data/out/p5n7"; out = []
for tag in sys.argv[2:]:
    for m in ("n7a", "n3a", "n6a"):
        p = os.path.join(D, f"{m}_{tag}.jsonl")
        if not os.path.isfile(p): continue
        for l in open(p, encoding="utf-8"):
            it = json.loads(l); it["id"] = f"{m}_{tag}|{it['id']}"; it.pop("order", None); out.append(it)
with open(sys.argv[1], "w", encoding="utf-8") as w:
    for it in out: w.write(json.dumps(it, ensure_ascii=False) + "\n")
print(len(out), "items ->", sys.argv[1])
