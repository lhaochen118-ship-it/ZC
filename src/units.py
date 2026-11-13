#!/usr/bin/env python3
"""Executable rule units written by the model (plan 4 W, docs/17 §1): a rulebook row of a NEW split (an industry the toolkit does
not know, a term the training rulebook does not have) -> a Python function over the shop's tables that the solver S can call like
a toolkit metric. Written by src/lrl_gen.py --mode unit from the split's INPUTS only (rulebook row + the shop's table card); kept
only when it runs on the shop's own data and gives a non-degenerate result. No answer is ever read.

Unit (one JSON line of cards_unit*.jsonl):
  {"kind": "unit", "id", "bid", "term", "definition", "name", "signature", "doc", "code", "triggers", "rulebook", "section"}

  load(paths) -> [unit]                         every unit of the files (one per (bid, term), the first that verified)
  bind(g, units)                                exec each unit's def in the program namespace g (toolkit + C + T already in g)
  verify(unit, C, data_end, T=None) -> (ok, why, sample)   exec + call on the shop's data (the last 12 months of data, the whole data)
  block(units) -> str                           '# SHOP FUNCTIONS' prompt block (signature + definition)
  spec_entries(units) -> [dict]                 api_spec-like entries so that precheck knows the names and their parameters"""
import ast, json, math, os, re, glob

MAX_UNITS = 12
SAFE_MODS = {"re", "math", "datetime", "collections", "itertools", "statistics", "functools", "decimal", "calendar", "pandas", "numpy"}


def load(paths):
    out, seen = [], set()
    for p in [x for x in (paths.split(",") if isinstance(paths, str) else paths) if x]:
        for f in (sorted(glob.glob(os.path.join(p, "cards_unit*.jsonl"))) if os.path.isdir(p) else [p]):
            if not os.path.isfile(f): continue
            for l in open(f, encoding="utf-8"):
                if not l.strip(): continue
                u = json.loads(l)
                if u.get("kind") != "unit" or not u.get("code") or not u.get("name"): continue
                k = (u.get("bid"), u.get("name"))
                if k in seen: continue
                seen.add(k); out.append(u)
    return out


def for_shop(units, bid):
    return [u for u in units if u.get("bid") in (None, "", "*", bid)]


def safe_code(code):
    """a unit is one top-level def (helper statements inside it), no import / open / exec / dunder access."""
    try: tree = ast.parse(code)
    except SyntaxError as e: return False, f"syntax: {e.msg}"
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(defs) != 1 or any(not isinstance(n, (ast.FunctionDef, ast.Expr)) for n in tree.body):
        return False, "one top-level function definition expected"
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            mods = [a.name.split(".")[0] for a in n.names] if isinstance(n, ast.Import) else [(n.module or "").split(".")[0]]
            if any(m not in SAFE_MODS for m in mods): return False, f"import of {mods} not allowed"
        if isinstance(n, (ast.Global, ast.Nonlocal)): return False, "global not allowed"
        if isinstance(n, ast.Name) and n.id in ("open", "exec", "eval", "compile", "__import__", "globals", "locals", "input", "breakpoint"):
            return False, f"{n.id} not allowed"
        if isinstance(n, ast.Attribute) and n.attr.startswith("__"): return False, "dunder access not allowed"
    return True, defs[0].name


def bind(g, units):
    """define every unit function in namespace g (its globals are g: toolkit metrics, C, T, pd ... are visible)."""
    ok = []
    for u in units:
        try:
            exec(compile(u["code"], f"<unit {u['name']}>", "exec"), g)
            if callable(g.get(u["name"])): ok.append(u["name"])
        except Exception: g.pop(u["name"], None)
    return ok


def _degenerate(v):
    try:
        import numpy as np, pandas as pd
        if v is None: return True
        if isinstance(v, bool): return False
        if isinstance(v, (int, float, np.integer, np.floating)): return not math.isfinite(float(v))
        if isinstance(v, (pd.Series, pd.DataFrame)): return len(v) == 0
        if isinstance(v, (list, tuple, dict, set, str)): return len(v) == 0
        return False
    except Exception: return True


def _named_parts(v):
    """a dict of several named quantities ({'tien_thu_may_cu': .., 'khach_phai_tra': ..}: G0e) instead of one value; a dict keyed by
    data values (names from the shop's tables) is a ranking and stays valid."""
    return isinstance(v, dict) and len(v) >= 2 and all(isinstance(k, str) and re.fullmatch(r"[a-z_][a-z0-9_]*", k) for k in v)


def _zero(v):
    try:
        import numpy as np
        return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) and float(v) == 0.0
    except Exception: return False


def _params(code):
    try: fn = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef))
    except Exception: return []
    a = fn.args; nd = len(a.defaults); out = []
    for i, x in enumerate(a.args):
        out.append({"name": x.arg, "required": i < len(a.args) - nd})
    return out


def verify(unit, C, data_end, T=None, timeout=20, months=12, budget=40.0):
    """run the unit on the shop's data: (ok, why, sample result). Periods tried: each of the last `months` months of the data and the
    whole data (from ~3 years back): more periods make the self-consistency signature sharper and catch a unit that only fails on
    some months (G0e: TypeError on 14 / 24 months). Unanswerable on a period is a valid outcome; any other exception on more than a
    quarter of the periods rejects the unit. A required parameter other than the period makes the unit unverifiable here."""
    import dm, pandas as pd, time as _t
    ok, why = safe_code(unit.get("code") or "")
    if not ok: return False, why, None
    end = pd.Timestamp(data_end) if data_end else None
    shop = dm.Shop(C, end, {"data_end": data_end})
    if end is None: end = shop.last
    g = dm.api(shop, end); g["T"] = T or {}
    if bind(g, [unit]) != [unit["name"]]: return False, "does not define its function", None
    f = g[unit["name"]]; ps = _params(unit["code"])
    req = [p["name"] for p in ps if p["required"]]
    if len(req) > 1: return False, f"needs arguments {req[1:]} besides the period: not checkable", None
    periods = [dm.between(pd.Timestamp(end) - pd.Timedelta(days=365 * 3), end)]
    if req:
        for i in range(months):
            k = end.year * 12 + end.month - 1 - i
            periods.insert(i, dm.month(k // 12, k % 12 + 1))
    results, errs, t0 = [], [], _t.time()
    import signal
    def _alarm(*_): raise TimeoutError("unit ran too long")
    for P in periods:
        if _t.time() - t0 > budget and results: break                  # a slow unit: what ran so far decides
        try:
            old = signal.signal(signal.SIGALRM, _alarm); signal.alarm(timeout)
            try: v = f(P) if req else f()
            finally: signal.alarm(0); signal.signal(signal.SIGALRM, old)
            results.append(v)
        except dm.Unanswerable as e: results.append(None)
        except Exception as e: errs.append(f"{type(e).__name__}: {str(e)[:160]}"); results.append(("E", type(e).__name__))
    n = len(results)
    if errs and (len(errs) * 4 > n or len(errs) == n): return False, f"errors on {len(errs)} / {n} periods: {errs[0]}"[:300], None
    vals = [v for v in results if not (isinstance(v, tuple) and v[:1] == ("E",))]
    if any(_named_parts(v) for v in vals): return False, "returns a dict of several named quantities: one value expected (a number, or a table / list for a ranking)", None
    good = [v for v in vals if not _degenerate(v)]
    if not good: return False, "degenerate result (None / NaN / empty) on every period", None
    if all(_zero(v) for v in good): return False, "zero on every test period", None
    s = good[-1]
    try: s = s if isinstance(s, (int, float, str, bool)) else (s.head(3).to_dict() if hasattr(s, "head") else list(s)[:3])
    except Exception: s = str(s)[:120]
    unit["_sig"] = signature(results)                                  # what the unit computes on the test periods (self-consistency)
    unit["_nerr"] = len(errs)
    return True, "ok", s


def signature(results):
    """a hashable summary of a unit's results on the test periods: numbers rounded, lists / tables by their first items."""
    out = []
    for v in results:
        try:
            if v is None or isinstance(v, tuple) and v[:1] == ("E",): out.append(("-",))
            elif isinstance(v, bool): out.append(("b", v))
            elif isinstance(v, (int, float)) or type(v).__module__.startswith("numpy") and hasattr(v, "__float__"): out.append(("n", round(float(v), 2)))
            elif hasattr(v, "to_dict") and hasattr(v, "head"): out.append(("t", str(list(v.head(5).to_dict().items()))[:200]))
            elif isinstance(v, (list, tuple, set)): out.append(("l", tuple(str(x)[:30] for x in list(v)[:5]), len(v)))
            else: out.append(("s", str(v)[:60]))
        except Exception: out.append(("?", str(v)[:60]))
    return repr(out)


def block(units):
    if not units: return ""
    lines = ["# SHOP FUNCTIONS (written from this shop's rulebook; already defined: call them like toolkit metrics when the question uses the term)"]
    for u in units[:MAX_UNITS]:
        lines.append(f"- {u['signature']}: {u.get('term')} = {str(u.get('definition') or u.get('doc') or '')[:300]}")
    return "\n".join(lines)


def spec_entries(units):
    out = []
    for u in units[:MAX_UNITS]:
        ps = [{"name": p["name"], "kind": "period" if p["name"] in ("p", "period") else "any", "required": p["required"], "default": None}
              for p in _params(u["code"])]
        out.append({"name": u["name"], "signature": u.get("signature") or u["name"], "params": ps, "filters": False, "returns": "any",
                    "doc": str(u.get("definition") or "")[:200], "example": "", "group": "unit", "by": []})
    return out


def env_units():
    """units of the current run (env ZC_UNITS: comma list of files / dirs), loaded once per process (CPU workers)."""
    global _ENV
    try: return _ENV
    except NameError: pass
    _ENV = load(os.environ.get("ZC_UNITS", "")) if os.environ.get("ZC_UNITS") else []
    return _ENV
