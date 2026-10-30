#!/usr/bin/env python3
"""Offline analysis of an agent run (no LLM needed): logs.jsonl + answers.jsonl ->
PA by type, failure categories, per-sample / per-path accuracy, agreement-vs-accuracy calibration, best abstain threshold,
re-voting with another tau, sample failures and the most frequent program errors.
Reads both log formats (v0: "attempts" list with "s"; v1: "samples" {"<idx><path>": [attempts]}).
Usage: analyze.py <logs.jsonl> <answers.jsonl> [--show N] [--revote out_submission.jsonl --tau T]"""
import os, sys, json
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import judge
import agent


def samples_of(L):
    if "samples" in L: return L["samples"]
    d = defaultdict(list)
    for a in L.get("attempts", []): d[str(a.get("s", 0)) + "B"].append(a)
    return d


def last_ok(atts):
    ok = [a for a in atts if a.get("ok")]
    return ok[-1]["answer"] if ok else None


def main():
    logs = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
    refs = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[2], encoding="utf-8") if l.strip()}
    show = int(sys.argv[sys.argv.index("--show") + 1]) if "--show" in sys.argv else 8
    S = {"correct": 1, "abstain": 0, "wrong": -1}
    by_type = defaultdict(Counter); cats = Counter(); calib = defaultdict(Counter); per_sample = defaultdict(Counter); rows = []
    k_max = max(len(samples_of(L)) for L in logs)
    for L in logs:
        ref = refs[L["id"]]; v = judge(L["final"], ref); by_type[ref["type"]][v] += 1
        answers = {k: last_ok(atts) for k, atts in samples_of(L).items()}
        for k, ans in answers.items():
            per_sample[k][judge(ans, ref) if ans else "failed"] += 1
        oks = [x for x in answers.values() if x]
        votes = Counter(agent.canon_of(x) for x in oks); rep = {agent.canon_of(x): x for x in oks}
        top = votes.most_common(1)[0] if votes else None
        conf = top[1] / k_max if top else 0.0
        top_v = judge(rep[top[0]], ref) if top else "none"
        calib[round(conf, 2)][top_v] += 1
        if v == "wrong":
            ft = L["final"].get("type")
            cats[f"wrong type (pred {ft}, ref {ref['type']})" if ft != ref["type"] else f"wrong value ({ref['type']})"] += 1
        elif v == "abstain":
            cats["abstain: no successful program" if not oks else "abstain: low agreement"] += 1
        rows.append((L, ref, v, conf, rep.get(top[0]) if top else None))
    n = len(logs); pa = sum(S[r[2]] for r in rows) / n
    print(f"N={n} PA={pa:.4f} correct={sum(r[2] == 'correct' for r in rows)} wrong={sum(r[2] == 'wrong' for r in rows)} "
          f"abstain={sum(r[2] == 'abstain' for r in rows)}")
    print("\nby reference type:")
    for t, c in sorted(by_type.items(), key=lambda x: -sum(x[1].values())):
        m = sum(c.values()); print(f"  {t:13s} n={m:4d} PA={(c['correct'] - c['wrong']) / m:+.3f}  {dict(c)}")
    print("\nper sample (index+path; greedy = first of each path): accuracy of that sample alone")
    for k in sorted(per_sample):
        c = per_sample[k]; m = sum(c.values()); print(f"  {k:4s} acc={c['correct'] / m:.3f} failed={c['failed'] / m:.3f}  {dict(c)}")
    print("\nfailure categories:"); [print(f"  {k}: {c}") for k, c in cats.most_common() if c]
    print("\nagreement of the top answer -> its outcome (answer only where accuracy > 50%):")
    for conf in sorted(calib):
        c = calib[conf]; m = sum(c.values())
        print(f"  conf={conf:.2f} n={m:4d} correct={c['correct']} wrong={c['wrong']} acc={c['correct'] / max(m, 1):.2f}")
    best = None
    for tau in sorted(calib):
        p = sum(calib[cf]["correct"] - calib[cf]["wrong"] for cf in calib if cf >= tau) / n
        if best is None or p > best[0]: best = (p, tau)
    print(f"\nbest tau on this run: {best[1]} -> PA {best[0]:.4f} (in-sample: confirm on held-out businesses)")
    if "--revote" in sys.argv:
        tau = float(sys.argv[sys.argv.index("--tau") + 1]) if "--tau" in sys.argv else best[1]
        with open(sys.argv[sys.argv.index("--revote") + 1], "w", encoding="utf-8") as f:
            for L, ref, v, conf, top in rows:
                f.write(json.dumps({"id": L["id"], **(top if top and conf >= tau else {"type": "abstain"})}, ensure_ascii=False) + "\n")
        print(f"re-voted submission written with tau={tau}")
    print("\nsample wrong answers:")
    for L, ref, v, conf, _ in [r for r in rows if r[2] == "wrong"][:show]:
        print(f"- {L['id']} [{L['business_id']}] {L['question']}\n    ref={json.dumps(ref['answer'], ensure_ascii=False)[:120]} ({ref['type']})"
              f"  pred={json.dumps(L['final'].get('answer'), ensure_ascii=False)[:120]} ({L['final'].get('type')}) conf={conf:.2f}")
    print("\nmost frequent program errors (last line):")
    errs = Counter(a["error"].strip().splitlines()[-1][:140] for L in logs for atts in samples_of(L).values() for a in atts
                   if not a.get("ok") and a.get("error"))
    for e, c in errs.most_common(show): print(f"  {c:4d} x {e}")


if __name__ == "__main__":
    main()
