#!/usr/bin/env python3
"""Offline analysis of an agent run (no LLM needed): logs.jsonl + answers.jsonl ->
PA by type, error categories, agreement-vs-accuracy calibration, best abstain threshold, and sample failures.
Re-deciding with a new tau / voting rule happens here in seconds instead of re-running Kaggle.
Usage: analyze.py <logs.jsonl> <answers.jsonl> [--show N] [--json out.json]"""
import sys, json, ast
from collections import Counter, defaultdict
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from score import judge


def main():
    logs = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
    refs = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[2], encoding="utf-8") if l.strip()}
    show = int(sys.argv[sys.argv.index("--show") + 1]) if "--show" in sys.argv else 8
    S = {"correct": 1, "abstain": 0, "wrong": -1}
    by_type = defaultdict(Counter); cats = Counter(); calib = defaultdict(Counter); rows = []
    for L in logs:
        ref = refs[L["id"]]; fin = L["final"]; v = judge(fin, ref)
        by_type[ref["type"]][v] += 1
        att = L["attempts"]; ok = [a for a in att if a.get("ok")]
        top = None
        if L["votes"]:
            # top-voted candidate regardless of tau: is it right?
            cand = next((a["answer"] for a in ok if str(canon_key(a["answer"])) == L["votes"][0][0]), None)
            top = judge(cand, ref) if cand else "abstain"
        conf = L["conf"]; calib[round(conf, 2)][top or "none"] += 1
        if v == "wrong":
            ft = fin.get("type")
            cats["wrong type (pred %s, ref %s)" % (ft, ref["type"])] += ft != ref["type"]
            cats["wrong value"] += ft == ref["type"]
        elif v == "abstain":
            cats["abstain: no successful program" if not ok else "abstain: low agreement"] += 1
        rows.append((L, ref, v, conf))
    n = len(logs); pa = sum(S[v] for *_, v, _ in rows) / n
    print(f"N={n} PA={pa:.4f} correct={sum(v == 'correct' for *_, v, _ in rows)} wrong={sum(v == 'wrong' for *_, v, _ in rows)} abstain={sum(v == 'abstain' for *_, v, _ in rows)}")
    print("\nby reference type:")
    for t, c in sorted(by_type.items(), key=lambda x: -sum(x[1].values())):
        m = sum(c.values()); print(f"  {t:13s} n={m:4d} PA={(c['correct'] - c['wrong']) / m:+.3f}  {dict(c)}")
    print("\nfailure categories:"); [print(f"  {k}: {c}") for k, c in cats.most_common() if c]
    print("\nagreement -> top-candidate outcome (choose tau where accuracy > 50%):")
    best = (pa, None)
    for conf in sorted(calib):
        c = calib[conf]; m = sum(c.values())
        print(f"  conf={conf:.2f} n={m:4d} correct={c['correct']} wrong={c['wrong']} acc={c['correct'] / max(m, 1):.2f}")
    for tau in sorted(calib):                                          # PA if we answer only when conf >= tau
        p = sum(calib[cf]["correct"] - calib[cf]["wrong"] for cf in calib if cf >= tau) / n
        if p > best[0]: best = (p, tau)
    print(f"\nbest tau on this run: {best[1]} -> PA {best[0]:.4f} (in-sample, verify on held-out businesses)")
    print("\nsample wrong answers:")
    for L, ref, v, conf in [r for r in rows if r[2] == "wrong"][:show]:
        last = next((a for a in reversed(L["attempts"]) if a.get("ok")), {})
        print(f"- {L['id']} [{L['business_id']}] {L['question']}\n    ref={json.dumps(ref['answer'], ensure_ascii=False)[:120]} ({ref['type']})"
              f"  pred={json.dumps(L['final'].get('answer'), ensure_ascii=False)[:120]} ({L['final'].get('type')}) conf={conf}")
    print("\nsample program errors:")
    errs = Counter(a["error"].strip().splitlines()[-1][:120] for L in logs for a in L["attempts"] if not a.get("ok") and a.get("error"))
    for e, c in errs.most_common(show): print(f"  {c:4d} x {e}")


def canon_key(ans):
    import agent
    return agent.canon(ans)


if __name__ == "__main__":
    main()
