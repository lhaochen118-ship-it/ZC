#!/usr/bin/env python3
import os, sys, json, glob, gzip, random, argparse
from collections import Counter, defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--unitq", required=True); ap.add_argument("--units", required=True)
    ap.add_argument("--mix", default=os.path.join(ROOT, "Process", "train_data", "p3", "v3a_mix.jsonl.gz"))
    ap.add_argument("--with-s2", default=""); ap.add_argument("--replay", type=int, default=1000); ap.add_argument("--u2", type=int, default=2500)
    ap.add_argument("--max-units", type=int, default=8); ap.add_argument("--seed", type=int, default=9); ap.add_argument("--name", default="s3_mix.jsonl")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rnd = random.Random(a.seed)
    import units as unitsmod, lrl, s2_build, wt_newterm_in
    ho = set(wt_newterm_in.holdout_shops())
    mix = [r for r in map(json.loads, gzip.open(a.mix, "rt", encoding="utf-8")) if r["business_id"] not in ho]
    by_bid = defaultdict(list)
    for r in mix: by_bid[r["business_id"]].append(r)
    U = defaultdict(list)
    for l in open(a.units, encoding="utf-8"):
        if l.strip():
            u = json.loads(l)
            if u["bid"] not in ho: U[u["bid"]].append(u)
    Q = []
    for f in glob.glob(os.path.join(a.unitq, "**", "cards_unitq*.jsonl"), recursive=True):
        Q += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    recs, st = [], Counter()

    def rec(base, user, target, source):
        m = base["messages"]
        recs.append({"id": f"{base['id']}~{source}~{len(recs)}", "business_id": base["business_id"], "source": "S3:" + source,
                     "messages": [m[0], {"role": "user", "content": user}, {"role": "assistant", "content": target}]})
        st[source] += 1

    for q in Q:
        b = q["bid"]
        if b in ho or not by_bid.get(b): continue
        base = rnd.choice(by_bid[b])
        fn = next((u for u in U[b] if u["name"] == q["unit_name"] and u["term"] == q["term"]), None)
        if fn is None: fn = {"name": q["unit_name"], "signature": q.get("signature") or q["unit_name"], "term": q["term"], "definition": q.get("definition"), "code": q["unit_code"]}
        others = [u for u in U[b] if u["name"] != fn["name"]]
        blk = unitsmod.block([fn] + rnd.sample(others, min(len(others), a.max_units - 1)))
        label = s2_build.label(q["program"])
        u0 = base["messages"][1]["content"]
        if rnd.random() < 0.6:
            rec(base, s2_build.with_parts(u0, block=blk, question=q["question"], as_of=q["as_of_date"]), label, "U1")
        else:
            row = {"content": f"{fn['term']}: {(fn.get('definition') or '')[:300]}"}
            dis = [{"content": f"{x['term']}: {(x.get('definition') or '')[:300]}"} for x in rnd.sample(others, min(1, len(others)))]
            rows = [row] + dis; rnd.shuffle(rows)
            rec(base, s2_build.with_parts(u0, block=blk, rules=lrl.render(rows), question=q["question"], as_of=q["as_of_date"]), label, "U3")
    pool = list(mix); rnd.shuffle(pool)
    for r in pool[:a.u2]:
        us = [u for u in U.get(r["business_id"], []) if u["term"].lower() not in r["question"].lower()]
        if not us: continue
        blk = unitsmod.block(rnd.sample(us, min(len(us), a.max_units)))
        rec(r, s2_build.with_parts(r["messages"][1]["content"], block=blk), r["messages"][2]["content"], "U2")
    for r in pool[a.u2:a.u2 + a.replay]:
        rec(r, r["messages"][1]["content"], r["messages"][2]["content"], "R")
    if a.with_s2:
        for l in open(a.with_s2, encoding="utf-8"):
            if not l.strip(): continue
            r = json.loads(l)
            if r.get("source") in ("S2:term", "S2:contrast", "S2:style"):
                recs.append(r); st[r["source"]] += 1
    rnd.shuffle(recs)
    with open(os.path.join(a.out, a.name), "w", encoding="utf-8") as w:
        for r in recs: w.write(json.dumps(r, ensure_ascii=False) + "\n")
    chars = sorted(sum(len(m["content"]) for m in r["messages"]) for r in recs) or [0]
    summ = {"records": len(recs), "by_source": dict(st), "chars_p99": chars[int(0.99 * (len(chars) - 1))], "chars_max": chars[-1],
            "est_tokens_M": round(sum(chars) * 0.36 / 1e6, 1)}
    json.dump(summ, open(os.path.join(a.out, a.name.replace(".jsonl", "_summary.json")), "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
