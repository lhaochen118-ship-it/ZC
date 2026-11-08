#!/usr/bin/env python3
"""Error analysis of a hybrid-flow run (docs/08 WP9; docs/06 §5 protocol): the wrong / abstained questions of the DEV sets only
(pub_full, pub_k0, training shops) grouped by how they failed and by question kind, never the held-out sets (pub_ho, tr_ho:
counted, not shown). For public questions the rule reader's program (src/rules_first.py log, 1000 / 1000 on public) is shown
next to the model's as a reference for the analyst (analysis only: nothing here goes to src/ or a prompt).

Failure kinds: no_valid (no sample ran), disagree (samples split, the top group wrong or under the threshold), agree_wrong (all
valid samples agree on a wrong answer), abstain_right (the top answer was right but conf < theta).
Usage: analyze_run.py <run_dir> [--sets pub_full,pub_k0] [--rules-log F] [--show 30] [--json out.json]"""
import os, sys, json, re
from collections import Counter, defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import score


def main():
    run = sys.argv[1]
    sets = sys.argv[sys.argv.index("--sets") + 1].split(",") if "--sets" in sys.argv else ["pub_full", "pub_k0"]
    show = int(sys.argv[sys.argv.index("--show") + 1]) if "--show" in sys.argv else 30
    rl = sys.argv[sys.argv.index("--rules-log") + 1] if "--rules-log" in sys.argv else ""
    held = set()
    for s in ("pub_ho", "tr_ho"):
        p = os.path.join(ROOT, "tests", "splits", s + ".txt")
        if os.path.isfile(p): held |= {l.strip() for l in open(p) if l.strip()}
    ids = set()
    for s in sets:
        ids |= {l.strip() for l in open(os.path.join(ROOT, "tests", "splits", s + ".txt")) if l.strip()}
    ids -= held
    R = {}
    for p in (os.path.join(ROOT, "data", "raw", "public_gt.jsonl"), os.path.join(ROOT, "data", "raw", "train", "train", "answers.jsonl")):
        if os.path.isfile(p): R.update(score.load(p))
    rules = {}
    if rl and os.path.isfile(rl):
        for l in open(rl, encoding="utf-8"):
            r = json.loads(l); rules[r["id"]] = r
    logs = {}
    for l in open(os.path.join(run, "logs.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        if r["id"] in ids: logs[r["id"]] = r
    kinds = Counter(); by_tpl = defaultdict(Counter); by_biz = defaultdict(Counter); rows = []
    import agent
    for i, r in logs.items():
        ref = R.get(i.split("~")[0])
        if ref is None: continue
        fin = r["final"]; j = score.judge(fin, ref) if fin.get("type") != "abstain" else "abstain"
        lg = r.get("samples") if isinstance(r.get("samples"), dict) else {}
        S = lg.get("samples") or []; valid = [s for s in S if s.get("ok")]
        groups = Counter(agent.canon_of(s["answer"]) for s in valid)
        top_ok = False
        if valid:
            best = lg.get("best") or valid[0]["answer"]
            try: top_ok = score.judge(best, ref) == "correct"
            except Exception: top_ok = False
        if j == "correct": k = "right"
        elif not valid: k = "no_valid"
        elif j == "abstain" and top_ok: k = "abstain_right"
        elif len(groups) == 1: k = "agree_wrong"
        else: k = "disagree"
        kinds[k] += 1
        tpl = (rules.get(i.split("~")[0]) or {}).get("tpl", "?")
        by_tpl[tpl][k] += 1; by_biz[r["business_id"]][k] += 1
        if k != "right": rows.append((k, i, r, ref, tpl, valid, S))
    n = sum(kinds.values())
    print(f"{n} dev questions: " + ", ".join(f"{k} {v} ({v / max(1, n):.1%})" for k, v in kinds.most_common()))
    print("by rule template (wrong kinds):")
    for tpl, c in sorted(by_tpl.items(), key=lambda x: -(sum(x[1].values()) - x[1]["right"]))[:25]:
        bad = sum(c.values()) - c["right"]
        if bad: print(f"  {tpl:24s} {bad:4d} / {sum(c.values()):4d}  {dict(c)}")
    print("by shop:", {b: sum(c.values()) - c["right"] for b, c in sorted(by_biz.items())})
    for k, i, r, ref, tpl, valid, S in rows[:show]:
        print(f"\n[{k}] {i} {r['business_id']} {tpl} | {r['question'][:150]}")
        print(f"   ref {json.dumps({x: ref.get(x) for x in ('type', 'answer')}, ensure_ascii=False)[:160]}")
        rr = rules.get(i.split("~")[0])
        if rr: print(f"   rule  {rr.get('code', '')[:200]}")
        for s in S[:3]:
            t = (s.get("turns") or [{}])[-1]
            print(f"   model[{s.get('variant')}] {str(t.get('code'))[:200]} -> {json.dumps(s.get('answer'), ensure_ascii=False)[:120] if s.get('ok') else (t.get('exec') or {}).get('error') or t.get('precheck')}")
    if "--json" in sys.argv:
        json.dump({"kinds": kinds, "by_tpl": {k: dict(v) for k, v in by_tpl.items()}}, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
