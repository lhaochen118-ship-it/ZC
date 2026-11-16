#!/usr/bin/env python3
import os, sys, json, glob, zlib, argparse
from collections import Counter, defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def target(u):
    lines = u["code"].split("\n")
    if len(lines) > 40 or any(len(l) > 200 for l in lines): return None
    return json.dumps({"plan": (u.get("plan") or "")[:700], "name": u["name"], "signature": u.get("signature") or u["name"],
                       "doc": (u.get("doc") or "")[:300], "triggers": (u.get("triggers") or [])[:8], "code": lines}, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tg1"); ap.add_argument("out"); ap.add_argument("--split", default=os.path.join(ROOT, "data", "raw", "train", "train"))
    ap.add_argument("--null-share", type=float, default=0.08); ap.add_argument("--cache", default="")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    import lrl_gen, wt_newterm_in
    ho = set(wt_newterm_in.holdout_shops())
    units, raws = [], []
    for d in [x for x in a.tg1.split(",") if x]:
        for f in glob.glob(os.path.join(d, "**", "units_teacher.jsonl"), recursive=True):
            units += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        for f in glob.glob(os.path.join(d, "**", "B", "raw_unit*.jsonl"), recursive=True):
            raws += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    kept = {(u["bid"], u["term"]): u for u in units}
    nulls, n_samples, plan_of = Counter(), Counter(), {}
    for r in raws:
        m = r.get("meta") or {}; k = (m.get("bid"), m.get("term"))
        if k[0] is None: continue
        n_samples[k] += 1
        j = lrl_gen.first_json(r.get("text") or "")
        if isinstance(j, dict) and j.get("name") in (None, "", "null") and not j.get("code"):
            nulls[k] += 1; plan_of.setdefault(k, str(j.get("plan") or "")[:300])
    meta_of = {}
    for r in raws:
        m = r.get("meta") or {}
        if m.get("bid"): meta_of.setdefault((m["bid"], m["term"]), m)
    cards = {}
    def card(b):
        if b not in cards:
            sh = lrl_gen._shop_load(a.split, a.cache, b)
            cards[b] = lrl_gen._tables_card(sh[0], sh[2]) if sh else None
        return cards[b]
    recs, evals, stats = [], [], Counter()
    for (b, term), u in sorted(kept.items()):
        t = target(u)
        if t is None: stats["skip_schema"] += 1; continue
        if card(b) is None: stats["skip_shop"] += 1; continue
        prompt = lrl_gen.unit_prompt(b, card(b), term, u.get("definition") or "", u.get("section") or "rulebook")
        if b in ho:
            evals.append({"bid": b, "term": term, "definition": u.get("definition"), "section": u.get("section"), "src": u.get("src"),
                          "teacher": {"name": u["name"], "code": u["code"], "agree": (u.get("verified") or {}).get("agree")}})
            stats["eval"] += 1; continue
        recs.append({"id": u.get("id"), "business_id": b, "source": "WT:" + str(u.get("src")),
                     "messages": [{"role": "system", "content": lrl_gen.SYS}, {"role": "user", "content": prompt}, {"role": "assistant", "content": t}]})
        stats["train_" + str(u.get("src"))] += 1
    null_keys = [k for k, c in nulls.items() if k not in kept and c * 2 > n_samples[k] and k[0] not in ho]
    null_keys.sort(key=lambda k: zlib.crc32(f"{k[0]}|{k[1]}".encode()))
    cap = int(a.null_share * len(recs) / max(1e-9, 1 - a.null_share))
    for b, term in null_keys[:cap]:
        m = meta_of.get((b, term)) or {}
        if card(b) is None: continue
        prompt = lrl_gen.unit_prompt(b, card(b), term, m.get("definition") or "", m.get("section") or "rulebook")
        t = json.dumps({"plan": plan_of.get((b, term)) or "không phải đại lượng tính được", "name": None, "signature": "", "doc": "",
                        "triggers": [], "code": []}, ensure_ascii=False)
        recs.append({"id": f"null_{zlib.crc32((b + term).encode())}", "business_id": b, "source": "WT:null",
                     "messages": [{"role": "system", "content": lrl_gen.SYS}, {"role": "user", "content": prompt}, {"role": "assistant", "content": t}]})
        stats["train_null"] += 1
    recs.sort(key=lambda r: zlib.crc32(str(r["id"]).encode()))
    with open(os.path.join(a.out, "wt_train.jsonl"), "w", encoding="utf-8") as w:
        for r in recs: w.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(a.out, "wt_eval_rows.jsonl"), "w", encoding="utf-8") as w:
        for r in evals: w.write(json.dumps(r, ensure_ascii=False) + "\n")
    chars = sorted(sum(len(m["content"]) for m in r["messages"]) for r in recs) or [0]
    summ = {"train": len(recs), "eval": len(evals), "stats": dict(stats), "teacher_units": len(units), "raw_samples": len(raws),
            "chars_p50": chars[len(chars) // 2], "chars_p99": chars[int(0.99 * (len(chars) - 1))], "chars_max": chars[-1],
            "est_tokens_M": round(sum(chars) * 0.33 / 1e6, 2)}
    json.dump(summ, open(os.path.join(a.out, "wt_summary.json"), "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
