#!/usr/bin/env python3
"""API spec of what an answer program may call (docs/08_MASTER_PLAN.md §5.1): one entry per name of the program namespace
(periods, number helpers, industry helpers, the shop's metrics, the agent tools T1-T9) with its parameters and their kinds.
Used by the grammar (src/grammar.py: one rule per function, right parameter names, values typed by kind), the precheck
(src/precheck.py: names / parameters exist) and the tool help (T8). Logic only: nothing here answers a question.

    build_api_spec(shop=None, today=None, hide=()) -> [{"name", "signature", "params": [{"name", "kind", "required",
                                                        "default"}], "filters": bool, "returns", "doc", "example", "group"}]
hide: names to leave out; "industry" hides every industry metric and helper (the pub_noind evaluation set)."""
import os, re, json, inspect
import dm, industry

HERE = os.path.dirname(os.path.abspath(__file__))
EX_PATH = os.path.join(HERE, "refs", "answer_examples.jsonl")
TR_HO = ("biz_105", "biz_109", "biz_114", "biz_123", "biz_127", "biz_130")   # held-out shops: never an example (tests/make_splits.py)

# keyword filters of every metric taking **f (dm.Shop._filter)
FILTERS = {"item": "item", "group": "group", "customer": "customer", "customer_group": "customer_group", "staff": "staff",
           "channel": "channel", "payment": "payment", "shift": "shift", "status": "status", "where": "lambda"}

# kind of a parameter by its name (positional or keyword)
KIND = {"p": "period", "ordered": "period", "sold": "period", "p_or_today": "period_or_today", "periods": "periods", "months": "periods",
        "d": "date", "date": "date", "until": "date", "a": "date", "b": "date", "start": "date", "today": "today",
        "y": "int", "m": "int", "q": "int", "y1": "int", "m1": "int", "y2": "int", "m2": "int", "ly": "int", "lm": "int", "ld": "int",
        "k": "int", "n": "int", "min_orders": "int", "min_trips": "int", "min_paid": "int", "nd": "int", "ndigits": "int",
        "x": "number", "new_price": "number", "part": "number", "whole": "number", "new": "number", "old": "number",
        "item": "item", "group": "group", "customer": "customer", "staff": "staff", "supplier": "supplier", "channel": "channel",
        "payment": "payment", "status": "status", "customer_group": "customer_group", "vehicle": "vehicle", "brand": "brand",
        "size": "size", "line": "line", "unit": "unit", "big_unit": "unit", "base_unit": "unit", "weekday": "weekday",
        "by": "by", "level": "text", "kind": "text", "shift": "shift", "job": "text", "where": "lambda",
        "valid": "bool_or_none", "after": "bool", "strict": "bool", "paid": "bool_or_none", "warranty": "bool_or_none", "leap": "bool",
        "ascending": "bool", "diapers": "bool_or_none",
        "values": "any", "codes": "any", "series": "any", "overrides": "any", "label": "text", "default": "text", "s": "text",
        "cond": "any", "why": "text", "name": "text", "text": "text", "task": "text", "bins": "any"}

# allowed values of the 'by' / text parameters (from the docstrings and dm.API_DOC); absent -> any string
BY = {"best_day": ("revenue", "orders"), "item_sales": ("qty", "revenue", "n_orders", "profit"), "top_items": ("qty", "revenue", "n_orders", "profit"),
      "customer_sales": ("revenue", "n_orders", "n_days", "aov", "n_cancelled"), "top_customers": ("revenue", "n_orders", "n_days", "aov", "n_cancelled"),
      "staff_sales": ("revenue", "orders", "n_days"), "weekday_avg": ("revenue", "orders", "qty"), "supplier_ranking": ("amount", "receipts"),
      "top_supplier": ("amount", "receipts"), "device_sales": ("qty", "revenue"), "top_lines": ("qty", "revenue"), "device_share": ("revenue", "qty"),
      "warranty_sales": ("visits",), "vehicle_sales": ("revenue", "tickets"), "top_vehicles": ("revenue", "tickets"),
      "staff_stats": ("revenue", "tickets", "commission", "n_days", "orders"), "top_staff": ("revenue", "tickets", "commission", "n_days", "orders"),
      "brand_sales": ("qty", "revenue"), "top_brands": ("qty", "revenue")}
TEXT_VALUES = {"shift": ("sáng", "chiều"), "level": ("line", "item", "brand"), "kind": ("máy mới", "máy cũ", "điện thoại", "máy tính bảng")}

# what a name returns (prompt docs, answer-type checks)
RETURNS = {"revenue": "money", "n_orders": "count", "aov": "money", "cancel_rate": "percent", "share": "percent", "growth": "percent",
           "pct": "percent", "pct_customers": "percent", "best_day": "date", "price_change_date": "date", "nearest_expiry": "date",
           "peak_hour": "int", "item_qty": "number", "n_customers": "count", "n_debtors": "count", "debt_total": "money",
           "debt_older_than": "money", "purchase_total": "money", "avg_purchase_price": "money", "gross_profit": "money",
           "net_profit": "money", "stock_at": "number", "stock_value": "money", "price_on": "money", "price_effect": "percent",
           "n_days_revenue_at_least": "count", "n_orders_revenue_at_least": "count", "n_delivery_trips": "count",
           "n_orders_by_trips": "count", "n_due_orders": "count", "delivered_qty": "number", "import_minus_sales": "number",
           "purchased_qty_base": "number", "size_qty": "number", "factor": "number", "increasing": "boolean",
           "top_items": "list", "top_customers": "list", "unsold_items": "list", "items_expiring": "list", "out_of_stock": "list",
           "top_supplier": "string", "best_month": "period", "monthly_revenue": "list", "top_debtors": "table", "debt_aging": "list",
           "debt_flow": "list", "stock_split": "list", "weekday_avg": "money", "orders": "DataFrame", "lines": "DataFrame",
           "item_sales": "Series", "customer_sales": "Series", "staff_sales": "Series", "supplier_ranking": "Series",
           "daily_revenue": "Series", "debts": "Series", "debt_by_customer": "Series", "lot_stock": "DataFrame",
           "lots_expiring": "DataFrame", "money": "money", "round_half_up": "number", "rank": "list", "key": "text"}
for _n in ("day", "today_p", "yesterday", "between", "month", "month_noyear", "quarter", "quarter_noyear", "year", "this_month",
           "prev_month_same_days", "prev_month", "same_month_last_year", "this_week", "last_week", "last_n_days", "next_n_days", "ytd", "until",
           "months_back", "tet", "before_tet", "tet_of"):
    RETURNS[_n] = "period"
RETURNS["month_seq"] = "periods"; RETURNS["lunar_to_solar"] = "date"

GROUP_OF = {**{n: "period" for n in dm.PERIOD_API}, **{n: "number" for n in dm.NUM_API}, **{n: "industry_helper" for n in industry.HELPERS}}

# agent tools T1-T9 (src/tools.py binds them per shop); signature, kinds and docs here so that the grammar / precheck / help know them
TOOLS = {
    "find": ("find(kind, text, k=5)", [("kind", "tool_kind", True), ("text", "text", True), ("k", "int", False)], "list",
             "T1: data values of a kind (item / customer / staff / supplier / group / channel / payment / vehicle) matching text: exact, "
             "accent-free, regional word, abbreviation / model line, typo; find(...)[0] is the best value (a string), printing it shows how each matched"),
    "values": ("values(table, column, contains=None, k=20)", [("table", "text", True), ("column", "text", True), ("contains", "text", False), ("k", "int", False)],
               "list", "T2: distinct values of a column of C with their counts (accent-free substring filter)"),
    "period": ("period(text, today=TODAY)", [("text", "text", True), ("today", "today", False)], "period",
               "T3: the Period a Vietnamese time phrase means under the rulebook ('tuần trước', 'quý 3/2025', 'Tết năm ngoái')"),
    "rule": ("rule(term)", [("term", "text", True)], "text", "T4: the rulebook definition of a term and the toolkit function implementing it"),
    "schema": ("schema(table=None)", [("table", "text", False)], "text", "T5: columns of the canonical tables (type, % empty, examples)"),
    "peek": ("peek(table, n=3, where=None)", [("table", "text", True), ("n", "int", False), ("where", "lambda", False)], "text", "T5: a few rows of a table"),
    "explain": ("explain(code)", [("code", "text", True)], "text", "T6: run a program with filter tracing: rows left after each filter, filters matching nothing"),
    "check": ("check(answer, expected_type=None)", [("answer", "any", True), ("expected_type", "text", False)], "list",
              "T7: format / sanity checks of an ANSWER dict (counts >= 0, shares in [0, 100], top-N size, dates inside the data)"),
    "help": ("help(name)", [("name", "text", True)], "text", "T8: signature, description and one checked example of a toolkit function"),
    "has_data": ("has_data(concept)", [("concept", "text", True)], "text",
                 "T9: whether the shop records a kind of information (tồn kho, giá vốn, nợ, nhân viên, ca, kênh, khách hàng, nhà cung cấp, lô / hạn dùng, ...) and where"),
}

SHOP_NAMES = ["orders", "lines", "revenue", "n_orders", "cancel_rate", "aov", "daily_revenue", "best_day", "n_days_revenue_at_least",
              "n_orders_revenue_at_least", "monthly_revenue", "best_month", "item_qty", "item_sales", "top_items", "customer_sales",
              "top_customers", "n_customers", "staff_sales", "debts", "debt_total", "debt_by_customer", "n_debtors", "top_debtors",
              "debt_aging", "debt_older_than", "debt_flow", "share", "increasing", "weekday_avg", "peak_hour", "size_qty", "unsold_items",
              "purchase_total", "avg_purchase_price", "supplier_ranking", "top_supplier", "import_minus_sales", "purchased_qty_base", "stock_at",
              "stock_split", "stock_value", "lot_stock", "lots_expiring", "items_expiring", "out_of_stock", "nearest_expiry", "price_on",
              "price_change_date", "price_effect", "gross_profit", "net_profit", "n_delivery_trips", "pct_customers", "factor", "delivered_qty",
              "n_orders_by_trips", "n_due_orders"]
class _Names:                                                           # dm.shop_api on this lists the metric names it binds
    def __getattr__(self, n): return n


assert set(dm.shop_api(_Names())) == set(SHOP_NAMES) | set(industry.INDUSTRY_NAMES), "api_spec.SHOP_NAMES out of date with dm.shop_api"

# builtins a program may use (docs/08 §5.6 (2))
BUILTINS = ("round", "len", "sum", "min", "max", "sorted", "abs", "int", "float", "str", "list", "dict", "set", "enumerate", "zip", "range",
            "bool", "tuple", "any", "all", "print", "isinstance", "reversed")
CONSTANTS = ("TODAY", "D", "C", "T", "pd", "np", "math", "re", "Period", "Unanswerable", "None", "True", "False", "ANSWER")


def _doc_lines():
    """name -> the line of dm.API_DOC / industry.INDUSTRY_API that documents it (first mention)."""
    out = {}
    texts = [dm.API_DOC] + list(industry.INDUSTRY_API.values())
    for t in texts:
        for line in re.split(r"\n(?=- )|; ", t):
            for n in re.findall(r"\b([a-z_][a-z0-9_]*)\(", line):
                out.setdefault(n, " ".join(line.split())[:300])
    return out


_EXAMPLES = None


def examples():
    """name -> the shortest checked program (src/refs/answer_examples.jsonl, held-out shops excluded) calling it."""
    global _EXAMPLES
    if _EXAMPLES is None:
        _EXAMPLES = {}
        try:
            for l in open(EX_PATH, encoding="utf-8"):
                r = json.loads(l)
                if r.get("business_id") in TR_HO: continue
                for n in set(re.findall(r"\b([a-z_][a-z0-9_]*)\(", r["code"])):
                    if n not in _EXAMPLES or len(r["code"]) < len(_EXAMPLES[n]["code"]):
                        _EXAMPLES[n] = {"question": r["question"], "code": r["code"]}
        except OSError: pass
    return _EXAMPLES


def _params(fn, name):
    try: sig = inspect.signature(fn)
    except (TypeError, ValueError): return [], False, f"{name}(...)"
    ps, filt = [], False
    for p in sig.parameters.values():
        if p.name == "self": continue
        if p.kind == p.VAR_KEYWORD: filt = True; continue
        if p.kind == p.VAR_POSITIONAL: continue
        req = p.default is p.empty
        ps.append({"name": p.name, "kind": KIND.get(p.name, "any"), "required": req,
                   "default": None if req else (p.default if isinstance(p.default, (int, float, str, bool, type(None))) else repr(p.default))})
    s = str(sig).replace("(self, ", "(").replace("(self)", "()")
    return ps, filt, f"{name}{s}"


_CACHE = {}


def build_api_spec(shop=None, today=None, hide=()):
    """the spec (shop / today do not change it: kept for the per-shop signature of docs/08 §5.1). hide: names or 'industry'."""
    hide = set(hide or ())
    if "industry" in hide: hide |= set(industry.INDUSTRY_NAMES) | set(industry.HELPERS)
    ck = tuple(sorted(hide))
    if ck in _CACHE: return _CACHE[ck]
    docs, exs = _doc_lines(), examples()
    spec = []
    def add(name, fn, group):
        if name in hide: return
        ps, filt, sig = _params(fn, name)
        doc = ((fn.__doc__ or "").strip().split("\n")[0] if fn is not None else "") or docs.get(name, "")
        spec.append({"name": name, "signature": sig + (" (+ filters)" if filt else ""), "params": ps, "filters": filt,
                     "returns": RETURNS.get(name, "any"), "doc": " ".join(doc.split())[:300], "example": (exs.get(name) or {}).get("code", ""),
                     "group": group, "by": list(BY.get(name, ()))})
    for n, f in dm.PERIOD_API.items(): add(n, f, "period")
    for n, f in dm.NUM_API.items(): add(n, f, "number")
    for n, f in industry.HELPERS.items(): add(n, f, "industry_helper")
    for n in SHOP_NAMES: add(n, getattr(dm.Shop, n), "metric")
    for n in industry.INDUSTRY_NAMES: add(n, getattr(dm.Shop, n), "industry")
    for n, (sig, ps, ret, doc) in TOOLS.items():
        if n in hide: continue
        spec.append({"name": n, "signature": sig, "params": [{"name": a, "kind": k, "required": r, "default": None} for a, k, r in ps],
                     "filters": False, "returns": ret, "doc": doc, "example": "", "group": "tool", "by": []})
    _CACHE[ck] = spec
    return spec


def by_name(spec): return {e["name"]: e for e in spec}


def compact_doc(spec, groups=("tool",)):
    """one line per entry of the given groups (the tools; the toolkit itself is documented by dm.API_DOC)."""
    return "\n".join(f"- {e['signature']}: {e['doc']}" for e in spec if e["group"] in groups)


def help_text(spec, name):
    """T8: signature + description + one checked example (<= ~300 tokens)."""
    e = by_name(spec).get(name)
    if e is None:
        import difflib
        near = difflib.get_close_matches(name, [x["name"] for x in spec], n=5, cutoff=0.5)
        return f"no function {name!r}" + (f"; did you mean {', '.join(near)}?" if near else "")
    out = f"{e['signature']} -> {e['returns']}: {e['doc']}"
    if e["by"]: out += f"\n  by= one of {', '.join(repr(b) for b in e['by'])}"
    if e["filters"]: out += "\n  filters: " + ", ".join(f"{k}=" for k in FILTERS)
    if e["example"]: out += f"\n  example: {e['example']}"
    return out[:1200]


if __name__ == "__main__":
    sp = build_api_spec()
    print(len(sp), "entries;", sum(1 for e in sp if e["group"] == "metric"), "metrics,", sum(1 for e in sp if e["group"] == "industry"), "industry")
    print(help_text(sp, "top_items")); print(help_text(sp, "revenu"))
