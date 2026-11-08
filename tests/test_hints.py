#!/usr/bin/env python3
"""src/hints.py on labelled training questions (src/refs/answer_examples.jsonl: question -> checked program):
- period: the hint's period (from / to) equals the first period expression of the checked program (questions with one);
- entities: every entity literal of the program (item= / customer= / staff= / group= / supplier= ..., positional item) is the
  data value of some hint entity (accent-insensitive);
- no hint contains a toolkit metric name (period codes are allowed, docs/08 §5.2).
Acceptance (docs/08 §8 WP2): period >= 99 %, entities >= 98 % on 200 questions (seed 20261108; --all for every question).
Usage: test_hints.py <train_split> <scratch> [--all] [--n 200]"""
import os, sys, json, ast, random, pickle, time
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("ZC_ADJ_ROWS", "move")
import pandas as pd
import dm, hints, qnorm, automap, ingest, api_spec

ENT_KW = {"item": "item", "customer": "customer", "staff": "staff", "group": "group", "supplier": "supplier", "customer_group": "customer_group"}
POS_ITEM = {"price_on", "avg_purchase_price", "stock_at", "stock_split", "import_minus_sales", "nearest_expiry", "price_change_date",
            "price_effect", "factor", "purchased_qty_base"}


def program_period(code, as_of):
    """(from, to) of the first period expression of a program, else None."""
    t = ast.parse(code); ns = dict(dm.PERIOD_API); ns["TODAY"] = pd.Timestamp(as_of)
    for n in ast.walk(t):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in dm.PERIOD_API and n.func.id not in ("lunar_to_solar", "month_seq"):
            try: p = eval(compile(ast.Expression(n), "<p>", "eval"), ns)
            except Exception: return None
            if isinstance(p, dm.Period): return str(p.start.date()), str(p.end.date())
    return None


def program_entities(code):
    out = []
    for n in ast.walk(ast.parse(code)):
        if not isinstance(n, ast.Call): continue
        for k in n.keywords:
            if k.arg in ENT_KW and isinstance(k.value, ast.Constant) and isinstance(k.value.value, str): out.append((ENT_KW[k.arg], k.value.value))
        if isinstance(n.func, ast.Name) and n.func.id in POS_ITEM and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
            out.append(("item", n.args[0].value))
    return out


def main():
    split, scratch = sys.argv[1], sys.argv[2]
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 200
    os.makedirs(scratch, exist_ok=True)
    ex = [json.loads(l) for l in open(os.path.join(ROOT, "src", "refs", "answer_examples.jsonl"), encoding="utf-8")]
    if "--all" not in sys.argv: ex = sorted(random.Random(20261108).sample(ex, n), key=lambda r: (r["business_id"], r["id"]))
    rb = hints.rulebook_rows(open(os.path.join(split, "rulebook.md"), encoding="utf-8").read())
    names = {e["name"] for e in api_spec.build_api_spec() if e["group"] in ("metric", "industry", "tool")}
    by = defaultdict(list)
    for r in ex: by[r["business_id"]].append(r)
    sys.path.insert(0, os.path.join(ROOT, "training", "data")); import common
    cache, canon_dir = os.environ.get("CACHE") or os.path.join(scratch, "cache"), os.path.join(scratch, "canon")
    pn = pok = en = eok = 0; leaks = []; misses = []; t0 = time.time()
    for b in sorted(by):
        biz, C, _, de = common.canon(split, cache, canon_dir, b)
        nz = qnorm.Normalizer.for_split(split, exclude_business=b, extra_texts=qnorm.Normalizer.entity_texts(biz))
        voc = None
        for r in by[b]:
            shop = dm.Shop(C, r["as_of_date"], {"data_end": de})
            if voc is None: voc = hints.ShopVocab(shop)
            h = hints.build_hints(r["question"], r["as_of_date"], shop, nz, rb, voc)
            txt = json.dumps(h, ensure_ascii=False)
            bad = [x for x in names if f"{x}(" in txt]
            if bad: leaks.append((r["id"], bad))
            pp = program_period(r["code"], r["as_of_date"])
            if pp is not None:
                pn += 1; got = (h["period"]["from"], h["period"]["to"]) if h["period"] else None
                if got == pp: pok += 1
                else: misses.append(("period", r["question"], pp, got))
            for kind, v in program_entities(r["code"]):
                en += 1
                if any(dm.key(e["data_value"]) == dm.key(v) for e in h["entities"]): eok += 1
                else: misses.append(("entity", r["question"], (kind, v), [(e["kind"], e["data_value"], e["how"]) for e in h["entities"]]))
    for m in misses[:40]: print("MISS", m)
    pr, er = pok / max(pn, 1), eok / max(en, 1)
    print(f"period {pok}/{pn} = {pr:.3f} | entities {eok}/{en} = {er:.3f} | leaks {len(leaks)} {leaks[:3]} | {time.time() - t0:.0f}s")
    ok = pr >= 0.99 and er >= 0.98 and not leaks
    print("test_hints " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
