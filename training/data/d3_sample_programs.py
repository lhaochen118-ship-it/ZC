#!/usr/bin/env python3
"""D3 step 1, "programs first" (docs/08 §7, OmniSQL-style synthesis): the checked training programs (held-out shops excluded)
re-instantiated on every non-held-out training shop, with another as-of day and other periods / entities / k / thresholds of
the same kind; each program runs with the toolkit on that shop and is kept when it gives a non-degenerate answer (not an error,
not empty; few zeros / unanswerable). A teacher then writes the questions (d3_teacher_prompts.py + a GPU job), and a round-trip
filter keeps the questions the student reads back to the same answer.
Logic only: programs are built from checked programs by code; questions come from a model.
Usage: d3_sample_programs.py <train_split> <ingest_cache> <canon_dir> <out.jsonl> [--n 15000] [--workers 3] [--seed 20261108]"""
import os, sys, ast, json, math, time, random, argparse, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

PERIOD_TODAY = {"yesterday", "today_p", "this_month", "last_week", "this_week", "ytd", "prev_month_same_days", "tet_of"}
DATE_ARG = {"debts": [0], "debt_total": [0], "debt_by_customer": [0], "n_debtors": [0], "top_debtors": [0], "debt_aging": [0],
            "debt_older_than": [0], "stock_at": [1], "stock_split": [1], "stock_value": [0], "lot_stock": [0], "lots_expiring": [0, 1],
            "items_expiring": [0, 1], "out_of_stock": [0], "nearest_expiry": [1], "price_on": [1], "day": [0], "until": [0]}
ITEM_POS = {"price_on", "avg_purchase_price", "stock_at", "stock_split", "import_minus_sales", "nearest_expiry", "price_change_date",
            "price_effect", "factor", "purchased_qty_base"}
ENT_KW = {"item": "item", "group": "group", "customer": "customer", "staff": "staff", "channel": "channel", "payment": "payment",
          "customer_group": "customer_group"}
TOPK = {"top_items", "top_customers", "top_debtors", "top_lines", "top_vehicles", "top_staff", "top_brands", "top_plates", "top_warranty_lines"}


def double_quotes(code):
    """the program with every plain string literal written with double quotes (the style of the checked programs)."""
    import io, tokenize
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(code).readline):
        t = tok
        if tok.type == tokenize.STRING and tok.string[:1] in ("'", '"') and tok.string[:3] not in ("'" * 3, '"' * 3):
            try:
                v = ast.literal_eval(tok.string)
                if isinstance(v, str): t = tok._replace(string=json.dumps(v, ensure_ascii=False))
            except Exception: pass
        out.append(t)
    return tokenize.untokenize(out)


class Mutator(ast.NodeTransformer):
    def __init__(self, rng, ctx):
        self.rng, self.c = rng, ctx; self.ok = True

    def rand_day(self, lo=None, hi=None):
        import pandas as pd
        lo = lo or self.c["first"]; hi = hi or self.c["as_of"]
        n = max(0, (hi - lo).days); return lo + pd.Timedelta(days=self.rng.randint(0, n))

    def visit_Call(self, n):
        self.generic_visit(n)
        f = n.func.id if isinstance(n.func, ast.Name) else None
        rng, c = self.rng, self.c
        if f is None: return n
        C = lambda v: ast.Constant(v)
        if f == "month" and len(n.args) == 2:
            d = self.rand_day(); n.args = [C(d.year), C(d.month)]
        elif f == "quarter" and len(n.args) == 2:
            d = self.rand_day(); n.args = [C(d.year), C((d.month - 1) // 3 + 1)]
        elif f == "year" and n.args:
            n.args = [C(self.rand_day().year)]
        elif f == "between" and len(n.args) == 2:
            import pandas as pd
            a = self.rand_day(); b = min(a + pd.Timedelta(days=rng.choice([2, 6, 9, 13, 20, 30, 45, 60])), c["as_of"])
            n.args = [C(a.strftime("%Y-%m-%d")), C(b.strftime("%Y-%m-%d"))]
        elif f == "last_n_days" and n.args:
            n.args[0] = C(rng.choice([7, 10, 14, 15, 30, 60, 90]))
        elif f == "month_noyear" and n.args:
            n.args[0] = C(self.rand_day().month)
        elif f == "quarter_noyear" and n.args:
            n.args[0] = C(rng.randint(1, 4))
        elif f == "month_seq" and len(n.args) == 4:
            import pandas as pd
            d = self.rand_day(); e = d + pd.DateOffset(months=rng.choice([2, 3, 4]))
            if e > c["as_of"]: e, d = c["as_of"], c["as_of"] - pd.DateOffset(months=2)
            n.args = [C(d.year), C(d.month), C(e.year), C(e.month)]
        elif f == "months_back" and len(n.args) == 2:
            n.args[1] = C(rng.randint(2, 6))
        elif f == "tet" and n.args:
            n.args = [C(rng.choice(sorted({self.rand_day().year for _ in range(5)})))]
        for i in DATE_ARG.get(f, []):
            if i < len(n.args) and isinstance(n.args[i], ast.Constant) and isinstance(n.args[i].value, str):
                n.args[i] = C(self.rand_day().strftime("%Y-%m-%d"))
        if f in ITEM_POS and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
            v = self.pick("item"); n.args[0] = C(v) if v else n.args[0]
        if f in TOPK and len(n.args) >= 2 and isinstance(n.args[1], ast.Constant) and isinstance(n.args[1].value, int):
            n.args[1] = C(rng.choice([1, 2, 3, 3, 5, 5, 10]))
        if f in ("n_days_revenue_at_least", "n_orders_revenue_at_least") and len(n.args) >= 2 and c.get("daily_q"):
            n.args[1] = C(int(round(rng.choice(c["daily_q"]), -4)))
        for kw in n.keywords:
            if kw.arg in ENT_KW and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                v = self.pick(ENT_KW[kw.arg])
                if v is None: self.ok = False
                else: kw.value = C(v)
        if f == "key" and n.args and isinstance(n.args[0], ast.Constant) and c.get("key_col"):
            vals = c["cols"].get(c["key_col"]) or []
            if vals: n.args[0] = C(rng.choice(vals))
            else: self.ok = False
        return n

    def pick(self, kind):
        vals = self.c["E"].get(kind) or []
        return self.rng.choice(vals) if vals else None


def shop_ctx(split, cache, canon_dir, bid):
    import pandas as pd, grammar, dm
    biz, C, notes, de = common.canon(split, cache, canon_dir, bid)
    shop = dm.Shop(C, None, {"data_end": de})
    E = grammar.entity_values(C)
    E["item"] = sorted({str(x) for x in C["lines"]["item_name"].dropna().unique()}) if "item_name" in C["lines"].columns else E["item"]
    E["customer"] = [x for x in E["customer"] if not dm.is_walkin(x)][:3000]
    cols = {}
    for t in ("lines", "items"):
        df = C.get(t)
        if df is None: continue
        for col in df.columns:
            if str(col) in ("active_ingredient", "brand", "size", "item_group"):
                cols[str(col)] = sorted({str(x) for x in df[col].dropna().unique()})[:500]
    o = C["orders"]; v = o[o["valid"]] if "valid" in o.columns else o
    daily = v.groupby(v["order_date"].dt.normalize())["revenue"].sum() if "revenue" in v.columns else None
    q = [float(daily.quantile(x)) for x in (0.3, 0.5, 0.7, 0.85)] if daily is not None and len(daily) else []
    return {"C": C, "de": de, "E": E, "cols": cols, "first": shop.first, "last": shop.last, "daily_q": q}


_SHOPS = {}


def run_one(args):
    """(record or None) for one mutated program."""
    split, cache, canon_dir, bid, tpl, as_of, seed = args
    import pandas as pd, signal, dm, agent, tools
    if bid not in _SHOPS:
        cal = pd.read_csv(os.path.join(split, "calendar.csv")); cal["date"] = pd.to_datetime(cal["date"]); dm.set_calendar(cal)
        _SHOPS.clear(); _SHOPS[bid] = shop_ctx(split, cache, canon_dir, bid)
    s = _SHOPS[bid]; rng = random.Random(seed)
    ctx = dict(s, as_of=pd.Timestamp(as_of))
    m = Mutator(rng, ctx)
    kc = None
    if "key(" in tpl["code"]:
        import re
        mm = re.search(r"d\[['\"](\w+)['\"]\]\.map\(key\)", tpl["code"])
        kc = mm.group(1) if mm else None
        if kc not in s["cols"]: return None
    ctx["key_col"] = kc
    tree = m.visit(ast.parse(tpl["code"]))
    if not m.ok: return None
    code = double_quotes(ast.unparse(tree))
    shop = dm.Shop(s["C"], as_of, {"data_end": s["de"]})
    def alarm(*_): raise TimeoutError()
    signal.signal(signal.SIGALRM, alarm); signal.alarm(20)
    try:
        r = dm.run_program(code, shop, as_of, extra=tools.bind_tools(shop, as_of))
        ans = agent.fmt_answer(r)
    except Exception: return None
    finally: signal.alarm(0)
    t, v = ans["type"], ans.get("answer")
    if t != "unanswerable":
        if v is None or (isinstance(v, (list, dict)) and not v): return None
        if t == "table" and not v.get("rows"): return None
    return {"business_id": bid, "as_of_date": as_of, "code": code, "answer": ans, "src_id": tpl["id"], "family": tpl["family"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("cache"); ap.add_argument("canon_dir"); ap.add_argument("out")
    ap.add_argument("--n", type=int, default=15000); ap.add_argument("--workers", type=int, default=3); ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--shops", default="")
    a = ap.parse_args()
    import retrieval, pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    rng = random.Random(a.seed); t0 = time.time()
    ex = [r for r in common.jl(os.path.join(common.SRC, "refs", "answer_examples.jsonl")) if r["business_id"] not in common.TR_HO]
    for r in ex: r["family"] = retrieval.family(r["code"])
    fam = {}
    for r in ex: fam.setdefault(r["family"], []).append(r)
    bids = sorted(d for d in os.listdir(os.path.join(a.split, "data")) if d not in common.TR_HO and os.path.isdir(os.path.join(a.split, "data", d)))
    if a.shops: bids = [b for b in bids if b in a.shops.split(",")]
    rng_dates = {}
    for b in bids:
        _, C, _, _ = common.canon(a.split, a.cache, a.canon_dir, b)
        d = C["orders"]["order_date"].dropna(); rng_dates[b] = (d.min(), d.max())
    per = math.ceil(a.n * 1.8 / len(bids))                               # candidates (about half survive)
    tasks = []
    fams = sorted(fam)
    for b in bids:
        lo, hi = rng_dates[b]
        for i in range(per):
            f = rng.choice(fams); tpl = rng.choice(fam[f])              # families balanced, then a checked program of the family
            span = max(1, (hi - lo).days - 60)
            as_of = (lo + pd.Timedelta(days=60 + rng.randint(0, span + 20))).strftime("%Y-%m-%d")
            tasks.append((a.split, a.cache, a.canon_dir, b, tpl, as_of, rng.randint(0, 10**9)))
    tasks.sort(key=lambda t: t[3])                                       # one shop per worker at a time (its tables loaded once)
    out, seen = [], set()
    with ProcessPoolExecutor(a.workers) as px:
        for r in px.map(run_one, tasks, chunksize=64):
            if r is None: continue
            k = (r["business_id"], r["as_of_date"], r["code"])
            if k in seen: continue
            seen.add(k); out.append(r)
    zero = [r for r in out if r["answer"]["type"] in ("number", "percent") and r["answer"]["answer"] == 0]
    unans = [r for r in out if r["answer"]["type"] == "unanswerable"]
    keep = {id(r) for r in rng.sample(zero, min(len(zero), int(0.05 * len(out))))} | {id(r) for r in rng.sample(unans, min(len(unans), int(0.05 * len(out))))}
    drop = {id(r) for r in zero + unans} - keep                         # few zeros / unanswerable (degenerate questions)
    out = [r for r in out if id(r) not in drop]
    rng.shuffle(out); out = out[:a.n]
    for i, r in enumerate(out): r["id"] = f"d3_{i:05d}"
    common.write_jl(a.out, out)
    from collections import Counter
    print(f"D3 programs: {len(out)} of {len(tasks)} candidates on {len(bids)} shops ({time.time() - t0:.0f}s); "
          f"types {dict(Counter(r['answer']['type'] for r in out))}; families {len(set(r['family'] for r in out))}", flush=True)


if __name__ == "__main__":
    main()
