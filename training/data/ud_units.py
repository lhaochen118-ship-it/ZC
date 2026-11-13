#!/usr/bin/env python3
"""UD — 'use the shop functions' records for the solver S (plan 4, docs/17 §4 S1): at inference the rule writer W adds a
# SHOP FUNCTIONS block (src/units.py) to the business prefix; S must call such a function when the question uses its term, and
ignore the ones that do not apply. Built from checked training records (no new labels):
  call      a record whose program is one metric call over a period, ANSWER = {"type": T, "answer": F(P, args)}: the call
            becomes a unit `def <name>(p): return F(p, args)` (its term / definition from the rulebook card whose toolkit hint
            uses F, else the toolkit's own doc line), shown in the block with 1-2 distractor units of other records; the label
            calls the unit: ANSWER = {"type": T, "answer": <name>(P)} (spec / plan renamed alike);
  ignore    the same prompt with only distractor units: the label is unchanged.
Usage: ud_units.py <records.jsonl> <out.jsonl> [--n-call 1200] [--n-ignore 600] [--seed 20261113]"""
import os, re, sys, ast, json, random, argparse, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

PERIODS = {"day", "today_p", "yesterday", "between", "month", "month_noyear", "quarter", "quarter_noyear", "year", "this_month",
           "prev_month_same_days", "prev_month", "same_month_last_year", "this_week", "last_week", "last_n_days", "ytd", "until",
           "months_back", "month_seq", "tet", "before_tet", "tet_of"}
MARKS = ("\n\n# EXAMPLES", "\n\n# RULES", "\n\n# HINTS", "\n\n# QUESTION")


def key(s):
    import dm
    return re.sub(r"[^a-z0-9]+", "_", dm.key(s)).strip("_")[:40]


def cards_by_fn():
    """toolkit function -> [(term, definition, by)] from the compiled rulebook cards (src/refs/lrl/compile_*.jsonl)."""
    out = {}
    for f in ("compile_train.jsonl", "compile_public.jsonl"):
        p = os.path.join(ROOT, "src", "refs", "lrl", f)
        if not os.path.isfile(p): continue
        for c in map(json.loads, open(p, encoding="utf-8")):
            h = c.get("code_hint") or ""
            m = re.match(r"\s*([a-z_]+)\(", h)
            if not m: continue
            by = re.search(r'by\s*=\s*"([^"]+)"', h)
            term = (c.get("content") or "").split(":")[0].strip()
            txt = (c.get("rule") or "") + " " + (c.get("content") or "")
            if re.search(r"JSON|\btype\b|làm tròn|ghi `|định dạng", txt) or term.lower() in ("percent", "number", "list", "table", "date",
                                                                                                "string", "boolean", "unanswerable"): continue
            out.setdefault(m.group(1), []).append((term, (c.get("rule") or c.get("content") or "")[:300], by.group(1) if by else None))
    return out


def doc_line(fn):
    import dm
    for l in dm.API_DOC.split("\n"):
        if re.search(r"\b" + re.escape(fn) + r"\(", l): return l.strip("- ").strip()[:250]
    return fn


def single_call(code):
    """(T, call node, period node) for ANSWER = {"type": T, "answer": F(P, ...)} with F a metric and P a period call, else None."""
    try: tree = ast.parse(code)
    except SyntaxError: return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign): return None
    v = tree.body[0].value
    if not isinstance(v, ast.Dict): return None
    d = {k.value: x for k, x in zip(v.keys, v.values) if isinstance(k, ast.Constant)}
    t, call = d.get("type"), d.get("answer")
    if not (isinstance(t, ast.Constant) and isinstance(call, ast.Call) and isinstance(call.func, ast.Name)): return None
    if call.func.id in PERIODS or not call.args: return None
    p = call.args[0]
    if not (isinstance(p, ast.Call) and isinstance(p.func, ast.Name) and p.func.id in PERIODS): return None
    return t.value, call, p


def make_unit(call, cards):
    fn = call.func.id
    by = next((kw.value.value for kw in call.keywords if kw.arg == "by" and isinstance(kw.value, ast.Constant)), None)
    plain = len(call.args) == 1 and all(kw.arg == "by" for kw in call.keywords)
    cand = [c for c in cards.get(fn, []) if c[2] == by] if plain else []   # the rulebook card of exactly this metric (same by=, no filter)
    rest = ast.Call(func=ast.Name(fn), args=[ast.Name("p")] + call.args[1:], keywords=call.keywords)
    body = ast.unparse(rest)
    if cand:
        term, definition, _ = cand[0]
    else:
        term, definition = fn.replace("_", " ") + (f" theo {by}" if by else ""), doc_line(fn)
    fixed = [ast.unparse(a) for a in call.args[1:]] + [f"{kw.arg}={ast.unparse(kw.value)}" for kw in call.keywords]
    if fixed: definition = f"{definition} (ở cửa hàng này: {', '.join(fixed)})"
    name = key(term) or fn
    name = (name if not name[0].isdigit() else "u_" + name) + "_" + hashlib.sha1(body.encode()).hexdigest()[:4]
    return {"name": name, "signature": f"{name}(p)", "term": term, "definition": definition[:300], "code": f"def {name}(p):\n    return {body}"}


def block(us):
    lines = ["# SHOP FUNCTIONS (written from this shop's rulebook; already defined: call them like toolkit metrics when the question uses the term)"]
    for u in us: lines.append(f"- {u['signature']}: {u['term']} = {u['definition']}")
    return "\n".join(lines)


def insert_block(user, blk):
    cut = min([i for i in (user.find(m) for m in MARKS) if i > 0] or [len(user)])
    return user[:cut] + "\n\n" + blk + user[cut:]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("records"); ap.add_argument("out")
    ap.add_argument("--n-call", type=int, default=1200); ap.add_argument("--n-ignore", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20261113)
    a = ap.parse_args(); rnd = random.Random(a.seed)
    cards = cards_by_fn()
    recs = [json.loads(l) for l in open(a.records, encoding="utf-8") if l.strip()]
    pool = []
    for r in recs:
        if not str(r.get("source", "")).startswith(("D1", "D2", "RFT")) or not r.get("program"): continue
        sc = single_call(r["program"])
        if sc: pool.append((r, sc, make_unit(sc[1], cards)))
    rnd.shuffle(pool)
    units_all = [u for _, _, u in pool]
    out, n_call, n_ign = [], 0, 0
    from collections import Counter
    per_fn = Counter(); cap = max(40, a.n_call // 12)                  # no metric dominates (revenue is 1 in 3 checked programs)
    for r, (t, call, per), u in pool:
        want_call = n_call < a.n_call and per_fn[call.func.id] < cap
        if not want_call and n_ign >= a.n_ignore: continue
        msgs = [dict(m) for m in r["messages"]]
        if len(msgs) != 3 or msgs[2]["role"] != "assistant": continue
        others = [x for x in rnd.sample(units_all, min(6, len(units_all))) if x["name"] != u["name"] and x["term"] != u["term"]][:2]
        if want_call:
            per_fn[call.func.id] += 1
            us = others[:rnd.randint(1, 2)] + [u]; rnd.shuffle(us)
            msgs[1]["content"] = insert_block(msgs[1]["content"], block(us))
            old = ast.unparse(call); new = f"{u['name']}({ast.unparse(per)})"
            lab = msgs[2]["content"]
            if old not in lab: continue
            lab = lab.replace(old, new).replace(f'"{call.func.id}"', f'"{u["name"]}"', 1)
            lab = re.sub(r"<plan>\s*" + re.escape(call.func.id) + r"\b", f"<plan>{u['name']}", lab)
            msgs[2]["content"] = lab
            out.append(dict(r, id=r["id"] + "~ud", source="UDcall", messages=msgs, program=r["program"].replace(old, new), unit=u)); n_call += 1
        elif n_ign < a.n_ignore and others:
            msgs[1]["content"] = insert_block(msgs[1]["content"], block(others))
            out.append(dict(r, id=r["id"] + "~udi", source="UDignore", messages=msgs)); n_ign += 1
        if n_call >= a.n_call and n_ign >= a.n_ignore: break
    with open(a.out, "w", encoding="utf-8") as f:
        for x in out: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"ud_units: {len(pool)} single-call records; wrote {n_call} call + {n_ign} ignore -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
