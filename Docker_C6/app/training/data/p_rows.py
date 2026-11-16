#!/usr/bin/env python3
import os, sys, json, glob, gzip, pickle, argparse
from collections import Counter
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
TRAIN = os.path.join(ROOT, "data", "raw", "train", "train")


def shop_job(args):
    b, qs, cache, hints_by_term = args
    import ingest, automap, shopctx, hints as hintsmod, link
    rb = open(os.path.join(TRAIN, "rulebook.md"), encoding="utf-8").read()
    bp = os.path.join(cache, b + ".pkl")
    biz = pickle.load(open(bp, "rb")) if os.path.isfile(bp) else ingest.load_business(os.path.join(TRAIN, "data", b))
    C, notes, _ = automap.build(biz)
    ctx = shopctx.ShopCtx(TRAIN, b, biz, C, notes, automap.parse_readme(biz.readme).get("data_end"), rb, mode="full")
    out, st = [], Counter()
    for q in qs:
        try: h = ctx.hints(q["question"], q["as_of_date"])
        except Exception: st["hint_error"] += 1; continue
        cards = []
        for rt in (h.get("rule_terms") or [])[:3]:
            t, d = rt.get("term", ""), rt.get("definition", "")
            cards.append({"content": f"{t}: {d[:400]}", "code_hint": hints_by_term.get(t), "source": "p:" + t})
        st["with_rows" if cards else "none"] += 1
        out.append({"id": q["id"], "cards": cards})
    return b, out, dict(st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="c1a,c1b,rq5_tr_dev_para_a,rq5_tr_dev_para_b,v2v4s1a,v2v4s1b,v2v4s2a,v2v4s2b,rq5_tr_dev_all")
    ap.add_argument("--out", default=os.path.join(ROOT, "training", "data", "out", "wj", "p_rows.jsonl"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--cache", default=os.path.join(ROOT, "training", "data", "out", "_ingest_train"))
    a = ap.parse_args()
    qs = {}
    for run in a.runs.split(","):
        for p in glob.glob(os.path.join(ROOT, "Process", "runs", run, "**", "logs.jsonl.gz"), recursive=True):
            for l in gzip.open(p, "rt", encoding="utf-8"):
                r = json.loads(l)
                if r.get("business_id", "").startswith("biz_"): qs.setdefault(r["id"], {k: r[k] for k in ("id", "business_id", "question", "as_of_date")})
    hb = {}
    for l in open(os.path.join(ROOT, "src", "refs", "lrl", "compile_train.jsonl"), encoding="utf-8"):
        c = json.loads(l); src = str(c.get("source") or "")
        h = c.get("code_hint")
        if src.startswith("rulebook:") and isinstance(h, str) and h.strip() and h.strip().lower() not in ("null", "none"): hb.setdefault(src[9:], h)
    by = {}
    for q in qs.values(): by.setdefault(q["business_id"], []).append(q)
    from concurrent.futures import ProcessPoolExecutor
    rows, st = [], Counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for b, out, s in ex.map(shop_job, [(b, by[b], a.cache, hb) for b in sorted(by)]):
            rows += out; st.update(s); print(f"  {b}: {len(out)} {s}", flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"p_rows: {len(rows)} questions {dict(st)} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
