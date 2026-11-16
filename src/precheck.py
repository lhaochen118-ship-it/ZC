#!/usr/bin/env python3
"""Checks of a generated program BEFORE it runs (docs/08_MASTER_PLAN.md §5.6, module M9). Logic only: it reports problems
for the model to fix (a repair turn); it never edits or writes a program.

    precheck(code, spec, hints=None, entity_values=None, escape=None, today=None) -> [{"level": "error"|"warn", "rule", "msg", "suggestion"}]
Rules: (1) parses; (2) only names of the API spec / tools / allowed builtins / its own variables; (3) keyword names exist and
not too many positional arguments; (4) entity strings exist in the data (accent-insensitive) - else an error with the closest
values; (5) the period differs from the hint period (warn, not for multi-period questions); (6) the answer type differs from
the hint answer type (warn); (7) no import / file / network / exec; plus columns subscripted with a string that no table has
(warn)."""
import ast, re, difflib
import dm

FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "input", "breakpoint", "exit", "quit",
                   "getattr", "setattr", "delattr"}
FORBIDDEN_MODS = {"os", "sys", "subprocess", "socket", "shutil", "pathlib", "requests", "urllib", "http", "pickle", "importlib", "builtins"}
ENT_KW = {"item": "item", "group": "group", "customer": "customer", "staff": "staff", "supplier": "supplier", "channel": "channel",
          "payment": "payment", "customer_group": "customer_group", "status": "status"}
POS_ITEM = {"price_on", "avg_purchase_price", "stock_at", "stock_split", "import_minus_sales", "nearest_expiry", "price_change_date",
            "price_effect", "factor", "purchased_qty_base"}
SOFT_KINDS = {"channel", "payment"}                                      # the toolkit canonicalises synonyms ('shopee', 'cash')


def _issue(level, rule, msg, suggestion=""): return {"level": level, "rule": rule, "msg": msg, "suggestion": suggestion}


def _ent_index(entity_values):
    idx = {}
    for kind, vals in (entity_values or {}).items():
        idx[kind] = {dm.key(v): v for v in vals}
    return idx


def first_period(tree, today):
    """(from, to) of the first period expression of the program, else None."""
    import pandas as pd
    ns = dict(dm.PERIOD_API); ns["TODAY"] = pd.Timestamp(today)
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in dm.PERIOD_API and n.func.id not in ("lunar_to_solar", "month_seq"):
            try: p = eval(compile(ast.Expression(n), "<p>", "eval"), ns)
            except Exception: return None
            if isinstance(p, dm.Period): return str(p.start.date()), str(p.end.date())
    return None


_SIZE_FUNCS = {"size_qty"}


def _size_args(tree):
    """string sizes given to size filters: size_qty(p, "M"), any call's size="XL"."""
    out = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call): continue
        if isinstance(n.func, ast.Name) and n.func.id in _SIZE_FUNCS and len(n.args) >= 2 and isinstance(n.args[1], ast.Constant) and isinstance(n.args[1].value, str):
            out.append(n.args[1].value)
        out += [kw.value.value for kw in n.keywords if kw.arg == "size" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str)]
    return out


_MASK_METHODS = {"ge", "gt", "le", "lt", "eq", "ne", "notna", "isna", "notnull", "isnull", "where", "between", "isin"}


def _len_of_mask(tree):
    """len(<comparison>) or len(<series>.ge(...) / .notna() / .where(...) ...), also through list() / .tolist() / .values: the
    length of a boolean mask (all rows); 38 / 38 such P2 samples were wrong."""
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "len" and n.args:
            a = n.args[0]
            for _ in range(3):                                          # len(list(m)), len(m.tolist()), len(m.values): still the mask
                if isinstance(a, ast.Call) and isinstance(a.func, ast.Name) and a.func.id in ("list", "tuple") and a.args: a = a.args[0]
                elif isinstance(a, ast.Call) and isinstance(a.func, ast.Attribute) and a.func.attr in ("tolist", "to_list", "to_numpy"): a = a.func.value
                elif isinstance(a, ast.Attribute) and a.attr == "values": a = a.value
                else: break
            if isinstance(a, ast.Compare): return True
            if isinstance(a, ast.Call) and isinstance(a.func, ast.Attribute) and a.func.attr in _MASK_METHODS: return True
    return False


def _bare_calls(tree, name):
    """calls of name(...) with no keyword filter and only the period argument."""
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name
            and len(n.args) <= 1 and not n.keywords]


def _uses_number(code, v):
    """the program writes the number v (187, 9100000, 9_100_000, 9.1e6, 9100000.0, 10 * 1e6 ...): any numeric literal or
    constant product of literals equal to v."""
    try: tree = ast.parse(code)
    except SyntaxError: return str(int(v)) in code
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool) and abs(n.value - v) < 1e-6: return True
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
            try:
                x = eval(compile(ast.Expression(n), "<n>", "eval"), {"__builtins__": {}})
                if isinstance(x, (int, float)) and abs(x - v) < 1e-6: return True
            except Exception: pass
    return False


MONEY_Q = r"\b(?:tien|doanh thu|dthu|doanh so|gia tri|von|loi nhuan|lai|lo|hoa hong|chi phi|thu nhap|gia)\b"
MONEY_BY = {"revenue", "amount", "aov", "profit", "commission"}


def _money_by(tree, by):
    """calls of a toolkit function with a by= metric whose metric is money: by='revenue' / 'amount' ... or no by= where the
    default is money (weekday_avg(p, 2) averages revenue). -> [(function, metric, non-money options)]"""
    out = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call) or not isinstance(n.func, ast.Name) or n.func.id not in by: continue
        e = by[n.func.id]; opts = list(e.get("by") or [])
        if not opts: continue
        dflt = next((p.get("default") for p in e["params"] if p["name"] == "by"), None)
        kw = next((k for k in n.keywords if k.arg == "by"), None)
        pos = [p["name"] for p in e["params"]].index("by") if any(p["name"] == "by" for p in e["params"]) else None
        if kw is not None: metric = kw.value.value if isinstance(kw.value, ast.Constant) else None
        elif pos is not None and len(n.args) > pos: metric = n.args[pos].value if isinstance(n.args[pos], ast.Constant) else None
        else: metric = dflt
        alts = [o for o in opts if o not in MONEY_BY]
        if metric in MONEY_BY and alts: out.append((n.func.id, metric, alts))
    return out


def _size_used(tree, v):
    """the program counts a size: a size_qty(...) call, or a string constant naming the size ('Số 4', '4' after 'size' / 'số')."""
    k = " ".join(dm.key(v).split()); bare = re.sub(r"^(?:size|so|co)\s+", "", k).strip()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "size_qty": return True
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            c = " ".join(dm.key(n.value).split())
            if c in (k, bare) or re.search(r"\b(?:size|so|co)\s*" + re.escape(bare) + r"\b", c): return True
    return False


def all_periods(src, today):
    """{(from, to)} of every period expression of a program (source text or parsed tree); None if it does not parse."""
    import pandas as pd
    try: tree = ast.parse(src) if isinstance(src, str) else src
    except SyntaxError: return None
    ns = dict(dm.PERIOD_API); ns["TODAY"] = pd.Timestamp(today); out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in dm.PERIOD_API and n.func.id not in ("lunar_to_solar", "month_seq"):
            try: p = eval(compile(ast.Expression(n), "<p>", "eval"), ns)
            except Exception: continue
            if isinstance(p, dm.Period): out.add((str(p.start.date()), str(p.end.date())))
    return out


def precheck(code, spec, hints=None, entity_values=None, escape=None, today=None, question=None, unit_check=False, mask_check=False,
             count_check=False, size_use_check=False, industry_doc=None):
    # question / unit_check / count_check / size_use_check / industry_doc: accepted for call compatibility; the domain-specific
    # checks they enabled were removed (16/11: only general question <-> program consistency checks remain)
    import api_spec
    out = []
    if code is None: return [_issue("error", "format", "no program block (<code> ... </code>) in the output")]
    try: tree = ast.parse(code)
    except SyntaxError as e: return [_issue("error", "syntax", f"SyntaxError: {e.msg} (line {e.lineno})")]
    by = {e["name"]: e for e in spec}
    known = set(by) | set(api_spec.BUILTINS) | set(api_spec.CONSTANTS) | {"d", "x"}
    assigned = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            mods = [a.name.split(".")[0] for a in n.names] if isinstance(n, ast.Import) else [(n.module or "").split(".")[0]]
            if any(m in FORBIDDEN_MODS for m in mods) or escape is None:
                out.append(_issue("error", "import", f"import of {', '.join(mods)} is not allowed: every function is already available"))
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store): assigned.add(n.id)
        if isinstance(n, ast.arg): assigned.add(n.arg)
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)): assigned.add(n.name)
        if isinstance(n, ast.ExceptHandler) and n.name: assigned.add(n.name)
    # escape blocks (<python>) may use pandas freely: only the safety rules
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            if n.id in FORBIDDEN_NAMES or n.id in FORBIDDEN_MODS:
                out.append(_issue("error", "forbidden", f"{n.id} is not allowed (no files, no network, no dynamic code)"))
            elif escape is None and n.id not in known and n.id not in assigned:
                near = difflib.get_close_matches(n.id, sorted(known), n=3, cutoff=0.6)
                out.append(_issue("error", "name", f"unknown name {n.id!r}", f"did you mean {', '.join(near)}?" if near else "use help(name) for the toolkit"))
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in FORBIDDEN_MODS:
            out.append(_issue("error", "forbidden", f"{n.value.id}.{n.attr} is not allowed"))
    if escape is not None: return out
    eidx = _ent_index(entity_values)
    def ent_check(kind, s, where):
        vals = eidx.get(kind)
        if not vals or not isinstance(s, str) or kind in SOFT_KINDS: return
        if kind == "status" and s in ("hoàn thành", "huỷ", "trả hàng", "đang giao"): return
        k = dm.key(s)
        if k in vals: return
        if kind == "item" and any(v.startswith(k + " ") for v in vals): return            # a model line / name prefix ('iPhone 16')
        near = [vals[x] for x in difflib.get_close_matches(k, list(vals), n=5, cutoff=0.5)]
        out.append(_issue("error", "entity", f"{where}: {s!r} is not a {kind} of this shop's data",
                          f"closest values: {near}" if near else f"find({kind!r}, {s!r}) lists matching values; if none exists the question may be unanswerable"))
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call) or not isinstance(n.func, ast.Name) or n.func.id not in by: continue
        e = by[n.func.id]; params = [p["name"] for p in e["params"]]
        if len(n.args) > len(params) and not any(isinstance(a, ast.Starred) for a in n.args):
            out.append(_issue("error", "args", f"{n.func.id}() takes {len(params)} positional arguments ({', '.join(params)}), got {len(n.args)}",
                              api_spec.help_text(spec, n.func.id)[:300]))
        for kw in n.keywords:
            if kw.arg is None: continue
            if kw.arg not in params and not (e["filters"] and kw.arg in api_spec.FILTERS):
                out.append(_issue("error", "kwarg", f"{n.func.id}() has no parameter {kw.arg}=", api_spec.help_text(spec, n.func.id)[:300]))
            if kw.arg in ENT_KW and isinstance(kw.value, ast.Constant): ent_check(ENT_KW[kw.arg], kw.value.value, f"{n.func.id}({kw.arg}=...)")
            if kw.arg in ENT_KW and isinstance(kw.value, (ast.List, ast.Tuple)):
                for x in kw.value.elts:
                    if isinstance(x, ast.Constant): ent_check(ENT_KW[kw.arg], x.value, f"{n.func.id}({kw.arg}=[...])")
            if kw.arg == "by" and e.get("by") and isinstance(kw.value, ast.Constant) and kw.value.value not in e["by"]:
                out.append(_issue("error", "by", f"{n.func.id}(by={kw.value.value!r}): by is one of {e['by']}"))
        if n.func.id in POS_ITEM and n.args and isinstance(n.args[0], ast.Constant): ent_check("item", n.args[0].value, f"{n.func.id}(item, ...)")
    # columns: d["col"] / x["col"] with a column no canonical table has
    cols = set((entity_values or {}).get("column") or [])
    if cols:
        for n in ast.walk(tree):
            if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str) and isinstance(n.value, ast.Name) \
                    and n.value.id in ("d", "x", "df", "l", "o") and n.slice.value not in cols:
                near = difflib.get_close_matches(n.slice.value, sorted(cols), n=4, cutoff=0.5)
                out.append(_issue("warn", "column", f"no canonical table has a column {n.slice.value!r}", f"closest: {near}" if near else ""))
    h = hints or {}
    if h.get("period") and not h.get("periods") and today is not None:
        fp = first_period(tree, today)
        hp = (h["period"].get("from"), h["period"].get("to"))
        if fp and fp != hp:
            out.append(_issue("warn", "period", f"the program's period {fp[0]}..{fp[1]} differs from the period read in the question {hp[0]}..{hp[1]}"))
    if len(h.get("periods") or []) >= 2 and today is not None:          # P2: every period the question names should be used
        hp, pp = all_periods("\n".join(h["periods"]), today), all_periods(tree, today)
        if hp and pp is not None and not hp <= pp:
            miss = sorted(hp - pp)
            out.append(_issue("warn", "named_periods", "the question names the periods " + ", ".join(h["periods"]) + " but the program does "
                              f"not use {miss[0][0]}..{miss[0][1]}"))
    if mask_check and _len_of_mask(tree):                               # P2: len() of a True / False Series counts every row
        out.append(_issue("error", "len_mask", "len(...) of a comparison / mask (x >= n, .ge(n), .notna(), .where(...)) counts every row, not the matches",
                          "count the matches with (s >= n).sum(), or filter first: len(s[s >= n])"))
    for th in h.get("thresholds") or []:                                 # P2: a threshold the question sets ('ít nhất 187 đơn')
        v = th.get("value")                                             # should appear in the program
        if isinstance(v, (int, float)) and not _uses_number(code, v):
            out.append(_issue("warn", "threshold", f"the question sets {th.get('op')} {v:g} {th.get('of') or ''}".rstrip() + " but the program does not use that number"))
    at = h.get("answer_type")
    m = re.search(r'ANSWER\s*=\s*\{\s*["\']type["\']\s*:\s*["\'](\w+)["\']', code)
    if at and m and m.group(1) != at and m.group(1) != "unanswerable" and not (at == "number" and m.group(1) == "percent"):
        out.append(_issue("warn", "type", f"answer type {m.group(1)} but the question seems to ask a {at}"))
    if not m and "raise Unanswerable" not in code and "ANSWER" not in code:
        out.append(_issue("error", "answer", "the program does not assign ANSWER"))
    return out


def errors(issues): return [i for i in issues if i["level"] == "error"]


def message(issues):
    """the repair message listing the issues (errors first)."""
    lines = []
    for i in sorted(issues, key=lambda x: x["level"] != "error"):
        lines.append(f"- [{i['level']}] {i['msg']}" + (f" ({i['suggestion']})" if i.get("suggestion") else ""))
    return "\n".join(lines)
