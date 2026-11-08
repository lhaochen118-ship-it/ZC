#!/usr/bin/env python3
"""Training data of training/data/* (docs/08 WP4 acceptance), on the files given:
- no public test id, no public shop (dev_*), no held-out training shop (tr_ho) anywhere in a record (ids, prompts, labels);
- the label program of sampled records runs on its shop / day to the record's answer (D1: the training answer);
- answer types of each set within 5 points (%) of D1's (D3 is allowed its own mix: reported only);
- duplicates (same shop + question) under 1 %.
Usage: test_data_factory.py <train_split> <scratch> <file.jsonl>[,<file2>...] [--n 150]"""
import os, sys, json, random, re
from collections import Counter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "training", "data"))
TR_HO = ("biz_105", "biz_109", "biz_114", "biz_123", "biz_127", "biz_130")


def main():
    split, S, files = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 150
    import common, dm, agent, tools
    pub_ids = {json.loads(l)["id"] for l in open(os.path.join(ROOT, "data", "raw", "public_test", "public_test", "questions.jsonl"), encoding="utf-8")}
    cache, canon = os.environ.get("CACHE") or os.path.join(S, "cache"), os.path.join(S, "canon")
    rng = random.Random(20261108); bad = []; types = {}; dup = {}
    for f in files:
        rows = common.jl(f); name = os.path.basename(f)
        leak = [r["id"] for r in rows if r["id"].split("~")[0] in pub_ids or str(r.get("business_id", "")).startswith("dev_") or r.get("business_id") in TR_HO]
        txt_leak = [r["id"] for r in rows[:3000] if re.search(r"\bdev_0\d\d\b|" + "|".join(TR_HO), json.dumps(r.get("messages", ""), ensure_ascii=False))]
        if leak or txt_leak: bad.append((name, "leak", (leak + txt_leak)[:5]))
        types[name] = Counter((r.get("answer") or {}).get("type") for r in rows)
        keys = Counter((r["business_id"], r.get("question")) for r in rows); dup[name] = sum(c - 1 for c in keys.values()) / max(1, len(rows))
        for r in rng.sample(rows, min(n, len(rows))):
            c = common.ctx(split, cache, canon, r["business_id"], "compact"); shop = c.shop(r["as_of_date"])
            try: got = agent.fmt_answer(dm.run_program(r["program"], shop, r["as_of_date"], extra=tools.bind_tools(shop, r["as_of_date"])))
            except dm.Unanswerable: got = {"type": "unanswerable", "answer": None}
            except Exception as e: got = {"error": str(e)[:100]}
            if r.get("answer") and r["answer"].get("type") and agent.canon_of(got) != agent.canon_of(r["answer"]) if "error" not in got else True:
                bad.append((name, "label", r["id"], r["program"][:120], got, r.get("answer")))
    base = types.get("d1.jsonl") or next(iter(types.values()))
    tot = sum(base.values())
    for name, cnt in types.items():
        t = sum(cnt.values())
        dev = max(abs(cnt.get(k, 0) / t - base.get(k, 0) / tot) for k in set(cnt) | set(base)) * 100
        print(f"{name}: {t} records, types {dict(cnt)}, max type share gap vs D1 {dev:.1f} points, duplicates {dup[name]:.2%}")
        if not name.startswith("d3") and dev > 5: bad.append((name, "types", round(dev, 1)))
        if dup[name] >= 0.01: bad.append((name, "duplicates", dup[name]))
    for b in bad[:15]: print("PROBLEM", b)
    print("test_data_factory " + ("PASS" if not bad else "FAIL")); sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
