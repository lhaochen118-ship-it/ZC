#!/usr/bin/env python3
import os, sys, re, json, ast, time, random, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

KW_WRONG = {"item": ["product", "mat_hang", "item_name"], "customer": ["khach", "customer_name"], "group": ["category", "item_group"],
            "staff": ["employee", "nhan_vien"], "channel": ["kenh", "sales_channel"], "payment": ["payment_method", "method"]}


def typo(s, rng):
    import dm
    k = dm.key(s); w = k.split()
    if len(w) >= 2 and rng.random() < 0.5: w = w[:-1] if rng.random() < 0.5 else w
    w = " ".join(w)
    if len(w) > 4:
        i = rng.randrange(1, len(w) - 1)
        w = w[:i] + rng.choice("aeiounhtc") + w[i + 1:] if rng.random() < 0.6 else w[:i] + w[i + 1:]
    return w


def corrupt(code, rng, counts=None):
    import dm
    t = ast.parse(code); cands = []
    for n in ast.walk(t):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            for kw in n.keywords:
                if kw.arg in ("item", "customer", "staff", "group") and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    cands.append(("entity", kw)); cands.append(("kwarg", kw))
                if kw.arg == "by" and isinstance(kw.value, ast.Constant): cands.append(("by", kw))
            cands.append(("fname", n))
    if not cands: return None
    by_kind = {}
    for k, nd in cands: by_kind.setdefault(k, []).append(nd)
    ks = sorted(by_kind, key=lambda k: ((counts or {}).get(k, 0), rng.random()))
    kind = ks[0]; node = rng.choice(by_kind[kind])
    if kind == "entity":
        v = node.value.value
        reg = [(b, a) for a, b in dm.REGIONAL if dm.key(b) in dm.key(v)]
        if reg and rng.random() < 0.5:
            b, a = reg[0]; nv = re.sub(re.escape(b), a, v, flags=re.I) if re.search(re.escape(b), v, re.I) else typo(v, rng)
        else: nv = typo(v, rng)
        if dm.key(nv) == dm.key(v): return None
        node.value = ast.Constant(nv)
    elif kind == "kwarg":
        node.arg = rng.choice(KW_WRONG.get(node.arg, ["value"]))
    elif kind == "by":
        node.value = ast.Constant(rng.choice(["sales", "amount", "quantity", "money", "count"]))
    elif kind == "fname":
        f = node.func.id
        if len(f) < 4: return None
        node.func = ast.Name(rng.choice([f[:-1], f + "s", f.replace("_", ""), "get_" + f]), ast.Load())
    from d3_sample_programs import double_quotes
    return double_quotes(ast.unparse(t)), kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("cache"); ap.add_argument("canon_dir"); ap.add_argument("out"); ap.add_argument("inputs")
    ap.add_argument("--n", type=int, default=2000); ap.add_argument("--seed", type=int, default=20261108)
    a = ap.parse_args()
    import precheck as pre, grammar
    rng = random.Random(a.seed); t0 = time.time()
    src = []
    for p in a.inputs.split(","): src += common.jl(p)
    rng.shuffle(src); rows = []; kinds = {}
    for r in src:
        if len(rows) >= a.n: break
        c = common.ctx(a.split, a.cache, a.canon_dir, r["business_id"], "compact")
        x = corrupt(r["program"], rng, kinds)
        if x is None: continue
        bad, kind = x
        if kind == "fname" and kinds.get("fname", 0) >= a.n // 3: continue
        m = r["messages"]; user = m[1]["content"]
        h = c.hints(r["question"], r["as_of_date"])
        iss = pre.precheck(bad, c.spec, h, c.entity_values, None, r["as_of_date"])
        if not pre.errors(iss): continue
        bad_out = grammar.wrap(bad, grammar.auto_plan(r["program"]))
        repair = "The program has problems:\n" + pre.message(iss) + "\n\nWrite the corrected output in the same format (<plan> ... </plan> then <code> ... </code>)."
        rows.append(dict(r, id=r["id"] + "~r", source="D4T", kind=kind,
                         messages=[m[0], {"role": "user", "content": user}, {"role": "assistant", "content": bad_out},
                                   {"role": "user", "content": repair}, m[-1]]))
        kinds[kind] = kinds.get(kind, 0) + 1
    common.write_jl(a.out, rows)
    print(f"D4T: {len(rows)} trajectories {kinds} -> {a.out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
