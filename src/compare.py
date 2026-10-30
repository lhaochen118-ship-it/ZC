#!/usr/bin/env python3
"""Compare agent runs on the training split with the fixed validation protocol: choose tau (and the vote variant) on
the DEV businesses (biz_101-biz_120), report PA on the HOLDOUT businesses (biz_121-biz_130) with the dev choice.
Vote variants are re-computed offline from the per-sample answers in logs.jsonl (no LLM):
  all   : every sample, conf = top votes / K
  B / A : samples of one path only
  agree : like all, but the answer is kept only if both paths (when present) contain the top answer

Usage: compare.py <answers.jsonl> <name>=<logs.jsonl>[,<logs.jsonl>...] [...]"""
import os, sys, json, re
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import judge
import agent

DEV = re.compile(r"biz_1(0[1-9]|1\d|20)$")
S = {"correct": 1, "abstain": 0, "wrong": -1}


def last_ok(atts):
    ok = [a for a in atts if a.get("ok")]
    return ok[-1]["answer"] if ok else None


def vote(L, variant):
    samples = L.get("samples") or {}
    keys = [k for k in samples if variant in ("all", "agree") or k.endswith(variant)]
    if not keys: return None, 0.0
    ans = {k: last_ok(samples[k]) for k in keys}
    oks = {k: a for k, a in ans.items() if a}
    if not oks: return None, 0.0
    votes = Counter(agent.canon_of(a) for a in oks.values()); rep = {agent.canon_of(a): a for a in oks.values()}
    top, n = votes.most_common(1)[0]; conf = n / len(keys)
    if variant == "agree":
        paths = {k[-1] for k in keys}
        if len(paths) > 1 and not all(any(agent.canon_of(a) == top for k, a in oks.items() if k.endswith(p)) for p in paths): conf = 0.0
    return rep[top], conf


def pa(rows, tau):
    return sum(S[v] for v, c in rows if c >= tau and v != "none") / max(len(rows), 1)


def main():
    refs = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[1], encoding="utf-8")}
    for spec in sys.argv[2:]:
        name, files = spec.split("=", 1)
        logs = [json.loads(l) for f in files.split(",") for l in open(f, encoding="utf-8") if l.strip()]
        print(f"=== {name}: {len(logs)} questions ({sum(1 for L in logs if DEV.match(L['business_id']))} dev)")
        base = {"dev": [], "hold": []}
        for L in logs: base["dev" if DEV.match(L["business_id"]) else "hold"].append(judge(L["final"], refs[L["id"]]))
        for sp, r in base.items():
            c = Counter(r); n = max(len(r), 1)
            print(f"  as run {sp:4s}: PA={(c['correct'] - c['wrong']) / n:+.4f}  {dict(c)}")
        for variant in ("all", "agree", "A", "B"):
            rows = {"dev": [], "hold": []}
            for L in logs:
                a, conf = vote(L, variant)
                rows["dev" if DEV.match(L["business_id"]) else "hold"].append((judge(a, refs[L["id"]]) if a else "none", conf))
            if not any(c for _, c in rows["dev"]): continue
            taus = sorted({round(c, 3) for _, c in rows["dev"] if c > 0})
            best = max(taus, key=lambda t: (pa(rows["dev"], t), -t))
            acc_any = Counter(v for v, c in rows["dev"] + rows["hold"])
            print(f"  vote={variant:5s} tau*={best:.2f}  dev PA={pa(rows['dev'], best):+.4f}  holdout PA={pa(rows['hold'], best):+.4f}"
                  f"  (holdout@0.5={pa(rows['hold'], 0.5):+.4f}; top-answer correct {acc_any['correct']}/{len(logs)})")
        by_type = defaultdict(Counter)
        for L in logs: by_type[refs[L["id"]]["type"]][judge(L["final"], refs[L["id"]])] += 1
        print("  by type: " + "  ".join(f"{t}:{(c['correct'] - c['wrong']) / sum(c.values()):+.2f}/{sum(c.values())}" for t, c in
                                       sorted(by_type.items(), key=lambda x: -sum(x[1].values()))))


if __name__ == "__main__":
    main()
