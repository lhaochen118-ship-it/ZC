#!/usr/bin/env python3
"""P5 plan H01: probe items from solver logs. One item per distinct (question, answer) candidate: text = question + the first
program giving that answer + the answer; votes = samples giving it. Training logs get labels from the training answers.
Usage: probe_items.py <out.jsonl> <logs.jsonl[.gz]>:<split>[:<answers.jsonl>] ..."""
import sys, json, gzip, hashlib, os
sys.path.insert(0, "/home/user/ZC/src")
import score


def rd(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        for l in f:
            if l.strip(): yield json.loads(l)


def items_of(path, split, answers=None):
    refs = score.load(answers) if answers else {}
    out = []
    for r in rd(path):
        smp = (r.get("samples") or {}).get("samples") or []
        groups = {}
        for s in smp:
            ans = s.get("answer")
            if not s.get("ok") or not isinstance(ans, dict) or ans.get("type") in (None, "abstain"): continue
            k = json.dumps(ans, ensure_ascii=False, sort_keys=True)
            code = next((t.get("code") for t in reversed(s.get("turns") or []) if t.get("code")), "") or ""
            g = groups.setdefault(k, {"ans": ans, "code": code, "votes": 0}); g["votes"] += 1
        tot = sum(g["votes"] for g in groups.values()) or 1
        for k, g in groups.items():
            it = {"uid": f"{split}|{r['id']}|{hashlib.md5(k.encode()).hexdigest()[:10]}", "split": split, "qid": r["id"],
                  "group": r.get("business_id"), "answer": g["ans"], "votes": g["votes"], "share": g["votes"] / tot, "n_groups": len(groups),
                  "text": f"Câu hỏi: {r.get('question')}\nChương trình:\n{g['code'][:3000]}\nKết quả: {k[:1500]}"}
            if refs:
                ref = refs.get(r["id"])
                if ref is None: continue
                it["label"] = int(score.judge(g["ans"], ref) == "correct")
            out.append(it)
    return out


if __name__ == "__main__":
    allit = []
    for spec in sys.argv[2:]:
        p = spec.split(":"); allit += items_of(p[0], p[1], p[2] if len(p) > 2 else None)
    with open(sys.argv[1], "w", encoding="utf-8") as w:
        for it in allit: w.write(json.dumps(it, ensure_ascii=False) + "\n")
    import collections
    c = collections.Counter(it["split"] for it in allit); lab = [it["label"] for it in allit if "label" in it]
    print(f"{len(allit)} items {dict(c)}; train positive rate {sum(lab) / max(1, len(lab)):.3f}")
