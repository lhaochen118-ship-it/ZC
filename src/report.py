#!/usr/bin/env python3
"""Summary of a v3 run: PA on dev (biz_101-120) / holdout (biz_121-130) / all, per answer type, per business with the
mapping that was chosen, abstain rate, and how often the LLM samples agree with the rule-based program.
Usage: report.py <split_dir> <run_dir> [--errors N]   (run_dir holds logs.jsonl, compile_logs.jsonl)"""
import os, re, sys, json
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score


def load(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.isfile(p) else []


def main():
    split, run = sys.argv[1], sys.argv[2]
    nerr = int(sys.argv[sys.argv.index("--errors") + 1]) if "--errors" in sys.argv else 0
    logs = load(os.path.join(run, "logs.jsonl"))
    refs = score.load(os.path.join(split, "answers.jsonl")) if os.path.isfile(os.path.join(split, "answers.jsonl")) else {}
    chosen = {r["business_id"]: r["chosen"] for r in load(os.path.join(run, "compile_logs.jsonl")) if "chosen" in r}
    def verdict(r):
        if r["final"].get("type") == "abstain": return "abstain"
        return "correct" if score.judge(r["final"], refs[r["id"]]) == "correct" else "wrong"
    def pa(c): n = sum(c.values()); return (c["correct"] - c["wrong"]) / max(n, 1)
    groups = defaultdict(Counter); per_b = defaultdict(Counter); per_t = defaultdict(Counter); agree = Counter(); errs = []
    for r in logs:
        b = r["business_id"]
        if refs:
            v = verdict(r); per_b[b][v] += 1; per_t[refs[r["id"]]["type"]][v] += 1
            dev = bool(re.fullmatch(r"biz_1(0\d|1\d|20)", b))
            groups["dev" if dev else "holdout"][v] += 1; groups["all"][v] += 1
            if v == "wrong": errs.append(r)
        else: per_b[b]["abstain" if r["final"].get("type") == "abstain" else "answered"] += 1
        rule = r["samples"].get("rule") or {}
        if rule.get("ok"):
            top = r["votes"][0][0] if r["votes"] else None
            agree["rule_ok"] += 1; agree["rule_is_top"] += int(top is not None and json.dumps(rule.get("answer"), ensure_ascii=False, sort_keys=True) in top or False)
    if refs:
        for g in ("dev", "holdout", "all"):
            c = groups[g]; print(f"{g:8s} PA {pa(c):+.3f}  n={sum(c.values())} correct {c['correct']} wrong {c['wrong']} abstain {c['abstain']}")
        print("type     " + "  ".join(f"{t}:{pa(c):+.2f}(n{sum(c.values())})" for t, c in sorted(per_t.items(), key=lambda x: -sum(x[1].values()))))
    print("business (chosen mapping):")
    for b in sorted(per_b):
        c = per_b[b]; s = f"PA {pa(c):+.2f} " if refs else ""
        print(f"  {b} {chosen.get(b, 'NONE'):22s} {s}{dict(c)}")
    t = [r.get("elapsed_s", 0) for r in logs]
    print(f"questions {len(logs)} | mapped {sum(1 for b in per_b if b in chosen)}/{len(per_b)} businesses | rule ok {agree['rule_ok']}")
    for r in errs[:nerr]:
        ref = refs[r["id"]]
        print(f"- {r['business_id']} {r['as_of_date']} | {r['question'][:110]}\n    ref {ref['type']} {json.dumps(ref['answer'], ensure_ascii=False)[:80]}"
              f" | got {json.dumps(r['final'], ensure_ascii=False)[:90]} conf {r['conf']:.2f} votes {str(r['votes'])[:120]}")


if __name__ == "__main__":
    main()
