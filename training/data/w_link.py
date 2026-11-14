#!/usr/bin/env python3
"""Plan 5 W2 training data (docs/18 §2, DW2): rule-linking records on the TRAINING shops.

Prompt = src/link.py prompt (the shop's numbered rulebook rows + the question), the inference prompt of the linker.
Label  = {"ids": [...]}: the rows the training question uses, read by P's hand-written term matcher (src/hints.py rule_terms,
user 14/11: allowed ONLY to label training questions; nothing hand-written runs at inference) on the original question and
on each paraphrase, plus the rows whose compiled toolkit hint names the function of the question's checked program.
Questions: training originals (d1_all) + paraphrases (d2_all); no public / private question. One worker process per shop.

To make the linker read the definitions and not memorise row numbers, each record shows the rows in a shuffled order, with a
few rows of the PUBLIC rulebook's industry sections added as distractors (rulebook text = input of the public split; no
question / answer of it is read).
Usage: w_link.py [--out training/data/out/wl] [--workers 4] [--max-per-shop 400]"""
import os, sys, json, gzip, random, pickle, argparse, re
from collections import Counter
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

TRAIN = os.path.join(ROOT, "data", "raw", "train", "train")
PUB_RB = os.path.join(ROOT, "src", "refs", "public_rulebook.md")


def jl(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def shop_job(args):
    b, qs, cache, seed, hints_of_program = args
    import ingest, automap, shopctx, hints as hintsmod, link, agent
    rb = open(os.path.join(TRAIN, "rulebook.md"), encoding="utf-8").read()
    bp = os.path.join(cache, b + ".pkl")
    if os.path.isfile(bp): biz = pickle.load(open(bp, "rb"))
    else:
        biz = ingest.load_business(os.path.join(TRAIN, "data", b)); os.makedirs(cache, exist_ok=True); pickle.dump(biz, open(bp, "wb"), protocol=4)
    C, notes, _ = automap.build(biz)
    ctx = shopctx.ShopCtx(TRAIN, b, biz, C, notes, automap.parse_readme(biz.readme).get("data_end"), rb, mode="full")
    sec = ctx.ind[0] if ctx.ind else None
    rows = link.rows_for(rb, sec)
    key = {hintsmod._k(t): i for i, (t, d, s) in enumerate(rows)}
    try: pub = open(PUB_RB, encoding="utf-8").read()
    except OSError: pub = ""
    distract = []
    for title, body in agent.rulebook_sections(pub).items():
        distract += [(t, d, title) for t, d in hintsmod.rulebook_rows(body).values()]
    rnd = random.Random(seed); out = []; st = Counter()
    for q in qs:
        try: h = ctx.hints(q["question"], q["as_of_date"])
        except Exception as e: st["hint_error"] += 1; continue
        lab = []
        for rt in h.get("rule_terms") or []:
            i = key.get(hintsmod._k(rt.get("term", "")))
            if i is not None and i not in lab: lab.append(i)
        fn = re.findall(r"\b([a-z_][a-z0-9_]*)\(", q.get("program") or "")
        for i in hints_of_program.get(tuple(sorted(set(fn))), []):
            if i < len(rows) and i not in lab and len(lab) < 3: lab.append(i)
        lab = lab[:3]; st["linked" if lab else "none"] += 1
        order = list(range(len(rows))) + [-1 - j for j in rnd.sample(range(len(distract)), min(4, len(distract)))]
        rnd.shuffle(order)
        shown = [rows[i] if i >= 0 else distract[-1 - i] for i in order]
        ids = sorted(order.index(i) + 1 for i in lab)
        conv = link.prompt(shown, q["question"], q["as_of_date"])
        out.append({"id": q["id"], "business_id": b, "source": "WL", "ids": ids,
                    "messages": conv + [{"role": "assistant", "content": json.dumps({"ids": ids})}]})
    return b, out, dict(st), sec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "training", "data", "out", "wl")); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-per-shop", type=int, default=400); ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--cache", default=os.path.join(ROOT, "training", "data", "out", "_ingest_train"))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rnd = random.Random(a.seed)
    qs = jl(os.path.join(ROOT, "Process", "train_data", "p3", "d1_all.jsonl.gz")) + jl(os.path.join(ROOT, "Process", "train_data", "p3", "d2_all.jsonl.gz"))
    by = {}
    for q in qs:
        if q.get("question") and q.get("business_id"): by.setdefault(q["business_id"], []).append(q)
    for b in by:
        rnd.shuffle(by[b]); by[b] = by[b][:a.max_per_shop]
    from concurrent.futures import ProcessPoolExecutor
    recs, st = [], Counter()
    jobs = [(b, by[b], a.cache, a.seed + i, {}) for i, b in enumerate(sorted(by))]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for b, out, s, sec in ex.map(shop_job, jobs):
            recs += out; st.update(s); print(f"  {b}: {len(out)} records {s} (section {sec})", flush=True)
    rnd.shuffle(recs)
    with open(os.path.join(a.out, "wl.jsonl"), "w", encoding="utf-8") as f:
        for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    chars = sum(len(m["content"]) for r in recs for m in r["messages"])
    print(f"wl.jsonl: {len(recs)} records, ~{chars / 3.1 / 1e6:.1f}M tokens; {dict(st)}; ids per record "
          f"{dict(Counter(len(r['ids']) for r in recs))}", flush=True)


if __name__ == "__main__":
    main()
