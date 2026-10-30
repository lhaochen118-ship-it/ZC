#!/usr/bin/env python3
"""Build an SFT dataset from agent runs on the TRAINING split (rejection sampling / STaR): for every training question,
keep the programs whose executed answer is judged correct against answers.jsonl, as chat examples
(system, user prompt, assistant = the correct program). Repaired programs are kept as their final, clean version
(the conversation is rebuilt without the failed turns), so the model learns to write the right program first time.
Only the training split is used (Rule 6).

Usage: harvest.py <split_dir> <out.jsonl> <logs.jsonl> [<logs.jsonl> ...] [--compile-logs <compile_logs.jsonl>] [--max-per-q 2]"""
import os, sys, json, pickle
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent
from score import judge


def main():
    args = sys.argv[1:]
    maxq = int(args[args.index("--max-per-q") + 1]) if "--max-per-q" in args else 2
    clogs = [args[args.index("--compile-logs") + 1]] if "--compile-logs" in args else []
    pos = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] not in ("--max-per-q", "--compile-logs"))]
    split, out, logs = pos[0], pos[1], pos[2:]
    refs = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(split, "answers.jsonl"), encoding="utf-8")}
    qs = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(split, "questions.jsonl"), encoding="utf-8")}
    rulebook = open(os.path.join(split, "rulebook.md"), encoding="utf-8").read()
    cache = {}
    def card(b):
        if b not in cache: cache[b] = agent.load_biz("", split, b)
        return cache[b]
    # canonical cards per business from compile logs (path A examples need them)
    canon = {}
    for f in clogs:
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            for atts in r["attempts"].values():
                ok = [a for a in atts if a.get("ok")]
                if ok and r.get("chosen") and ok[-1].get("path") == r["chosen"]: canon[r["business_id"]] = ok[-1]
    seen = defaultdict(set); n_q = set(); rows = []
    for f in logs:
        for l in open(f, encoding="utf-8"):
            L = json.loads(l); q = qs[L["id"]]; ref = refs[L["id"]]
            for key, atts in (L.get("samples") or {}).items():
                ok = [a for a in atts if a.get("ok")]
                if not ok or judge(ok[-1]["answer"], ref) != "correct": continue
                code = ok[-1]["code"].strip(); path = key[-1]
                if code in seen[L["id"]] or len(seen[L["id"]]) >= maxq: continue
                if path == "A" and q["business_id"] not in canon: continue
                z = card(q["business_id"])
                if path == "A":
                    c = canon[q["business_id"]]
                    msgs = [{"role": "system", "content": agent.SYSTEM_CANON},
                            {"role": "user", "content": agent.user_msg(rulebook, z.readme, z.card(), q, c["card"], c["notes"])}]
                else:
                    msgs = [{"role": "system", "content": agent.SYSTEM_RAW}, {"role": "user", "content": agent.user_msg(rulebook, z.readme, z.card(), q)}]
                msgs.append({"role": "assistant", "content": f"```python\n{code}\n```"})
                rows.append({"id": L["id"], "business_id": q["business_id"], "type": ref["type"], "path": path, "messages": msgs})
                seen[L["id"]].add(code); n_q.add(L["id"])
    with open(out, "w", encoding="utf-8") as w:
        for r in rows: w.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} examples covering {len(n_q)} questions / {len(refs)} training questions -> {out}")


if __name__ == "__main__":
    main()
