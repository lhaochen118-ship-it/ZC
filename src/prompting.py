#!/usr/bin/env python3
"""Compact prompt of the hybrid flow and the parser of the model's output (docs/08_MASTER_PLAN.md §5.3).

Order (shared parts first, for vLLM's prefix cache; agent.Engine.prefix_of cuts at '# EXAMPLES' / '# QUESTION' / '# TASK'):
  system = # ROLE, # API (dm.API_DOC + tools; hidden names left out), # OUTPUT FORMAT          -> same for every shop
  user   = # BUSINESS (compact schema card <= ~1000 tokens, data card, rulebook changes, industry rules)   -> same per shop
           # EXAMPLES (2-4 checked programs), # HINTS (JSON), # QUESTION (+ as-of date)           -> per question
The model writes '<plan>one line</plan>' then '<code> ... ANSWER = {...} </code>' (or the escapes <python> / <sql>).
Variants: direct (default), rephrase (restate the question inside <plan> first), explore (may call the tools first).
Logic only: nothing here decides an answer."""
import re, json
import dm

ROLE = """# ROLE
You are DataMate, a data analyst for a Vietnamese small business. You answer ONE question by writing a short Python program
that calls the toolkit below on the business's canonical tables. The program computes the answer; never compute numbers
yourself and never copy numbers from the hints into the answer."""

OUTPUT = """# OUTPUT FORMAT
<plan>quantity | period | filters, one line</plan>
<code>
ANSWER = {"type": "<number|percent|string|date|boolean|list|table|unanswerable>", "answer": <expression>}
</code>
- One program, usually one line; helper assignments before ANSWER are allowed. Use the exact names, codes and labels of the
  data (the HINTS give the data values matching the question's words; find(kind, text)[0] looks one up).
- number: money in whole VND, counts as integers; percent: 12.5 for 12.5 %; date: "YYYY-MM-DD"; list: in the order asked;
  table: {"columns": [...], "rows": [[...], ...]}; string: the name as written in the data.
- When the data cannot answer (not recorded, outside the data's dates, a quality no column records), write
  `raise Unanswerable("<reason>")` instead of ANSWER.
- If the toolkit cannot express the question, write <python> ... </python> instead of <code>: pandas over C (canonical
  tables) and T (raw tables) with the toolkit, ending with ANSWER = {...}."""

OUTPUT_SQL = """- Or <sql> ... </sql>: one DuckDB SELECT over the canonical tables of # BUSINESS (orders, lines, ...; valid = completed
  orders) and the raw tables (raw_<name>); its first column is the answer: name it after the answer type (AS number, AS percent,
  AS date, AS string, AS list); several columns = a table."""

VARIANT = {
    "direct": "",
    "think": "",                                                       # same prompt; the model reasons first (thinking mode)
    "rephrase": "\n# TASK\nFirst restate the question in plain standard Vietnamese inside <plan> (quantity | period | filters), then write the program.",
    "explore": "\n# TASK\nYou may first explore the data: write <code> with print(...) calls of the tools (find, values, schema, peek, has_data, help, "
               "period, rule); you will see their output, then write the final program.",
}


OUTPUT_COMPACT = """# OUTPUT FORMAT
<plan>quantity | period | filters</plan>
<code>
ANSWER = {"type": "<number|percent|string|date|boolean|list|table|unanswerable>", "answer": <expression>}
</code>
Data values exactly as in the data (HINTS give them). percent: 12.5; date: "YYYY-MM-DD"; table: {"columns": [...], "rows": [...]}.
Not answerable from the data: raise Unanswerable("<reason>"). Toolkit not enough: <python> pandas over C / T ending with ANSWER </python>."""


COMPACT_SKIP = {"canon_status", "map_status", "canon_payment", "canon_channel", "require", "lunar_to_solar", "debts", "debt_by_customer",
                "rule", "schema", "peek", "explain", "check", "has_data", "purchased_qty_base"}   # mapping helpers / exploration tools


def api_short(spec, hide=()):
    """the toolkit as signatures only, grouped (compact prompt): names, parameters, 'by' values, return kinds."""
    from api_spec import FILTERS
    groups = [("Periods (TODAY = today; whole days)", ("period",)), ("Numbers", ("number",)),
              ("Metrics; filters " + ", ".join(k + "=" for k in FILTERS) + " (where=lambda d: <bool Series over the lines>)", ("metric",)),
              ("Tools", ("tool",))]
    out = []
    for title, gs in groups:
        items = []
        for e in spec:
            if e["group"] not in gs or e["name"] in hide or e["name"] in COMPACT_SKIP: continue
            ps = ", ".join(p["name"] + ("" if p["required"] else "=" + (repr(p["default"]) if not isinstance(p["default"], str) or not p["default"].startswith("(") else "..."))
                           for p in e["params"])
            sig = f"{e['name']}({ps})"
            if e.get("by"): sig = sig.replace("by='" + e["by"][0] + "'", "by=" + "|".join(e["by"]))
            r = e.get("returns") or "any"
            if r not in ("any", "period", "text", "number") and e["group"] not in ("period",): sig += f"->{r}"
            items.append(sig)
        if items: out.append(f"- {title}: " + "; ".join(items))
    return "\n".join(out)


def system_block(api_spec=None, hide=(), escapes=("python",), tools_doc="", mode="full"):
    """# ROLE + # API + # OUTPUT FORMAT: identical for every shop of a run (one prefix-cache entry).
    mode full: dm.API_DOC (the base model needs the semantics); compact: signatures only (fine-tuned models, SFT <= 2k tokens)."""
    if mode == "compact":
        if api_spec is None:
            import api_spec as _as
            api_spec = _as.build_api_spec(hide=hide)
        api = api_short(api_spec, hide)
        out = OUTPUT_COMPACT + ("\n" + OUTPUT_SQL if "sql" in escapes else "")
        return f"{ROLE}\n\n# API\n{api}\n\n{out}"
    api = dm.API_DOC
    if tools_doc: api += "\n- Tools (deterministic lookups, callable inside the program):\n" + tools_doc
    out = OUTPUT + ("\n" + OUTPUT_SQL if "sql" in escapes else "")
    return f"{ROLE}\n\n# API\n{api}\n\n{out}"


def _fmt_val(x):
    s = str(x)
    if hasattr(x, "strftime"):
        try: s = x.strftime("%Y-%m-%d")
        except Exception: pass
    s = " ".join(s.split())
    return s[:40] + ("…" if len(s) > 40 else "")


def compact_card(C, notes="", max_chars=3600, examples=2):
    """the canonical tables in one line per table: columns with their kind and 1-2 example values (or the full value set
    when it is small), date ranges, fill rates under 90 %. Columns repeated from orders are not repeated in lines."""
    import pandas as pd
    out, seen = [], set()
    order = ["orders", "lines", "items", "customers", "purchases", "payments", "stock", "lots", "prices", "expenses", "staff"]
    names = sorted(C, key=lambda t: (order.index(t) if t in order else len(order), t))
    for t in names:
        df = C.get(t)
        if not hasattr(df, "columns") or not len(df.columns): continue
        cols = []
        for c in df.columns:
            if str(c).startswith("_") or str(c).endswith("_raw") or (t == "lines" and c in seen and c not in ("order_id",)): continue
            s = df[c]; nn = s.notna().mean() if len(s) else 0.0
            if nn == 0: continue
            fill = f" {nn:.0%}" if nn < 0.9 else ""
            if pd.api.types.is_bool_dtype(s): desc = "bool"
            elif pd.api.types.is_datetime64_any_dtype(s):
                d = s.dropna(); desc = f"{_fmt_val(d.min())}..{_fmt_val(d.max())}" if len(d) else "date"
            elif pd.api.types.is_numeric_dtype(s):
                d = s.dropna(); desc = f"num {d.min():g}..{d.max():g}" if len(d) else "num"
            else:
                u = s.dropna().astype(str).str.strip(); u = u[u != ""].unique()
                if len(u) <= 6: desc = "|".join(_fmt_val(x) for x in u)
                else: desc = f"{len(u)} values e.g. " + ", ".join(_fmt_val(x) for x in u[:examples])
            cols.append(f"{c} ({desc}{fill})")
        if t == "orders": seen = set(df.columns)
        out.append(f"{t} [{len(df)} rows]: " + "; ".join(cols))
    card = "\n".join(out)
    if len(card) > max_chars: card = card[:max_chars].rsplit("\n", 1)[0] + "\n…"
    if notes and str(notes).strip(): card += "\nNOTES: " + " ".join(str(notes).split())[:600]
    return card


def business_block(schema_card, datacard="", rule_changes="", industry_doc="", readme=""):
    """# BUSINESS: the same for every question of a shop (second prefix-cache level)."""
    parts = ["# BUSINESS", "Canonical tables C (the toolkit reads them):", schema_card]
    if readme: parts += ["README (shop description):", readme.strip()[:1500]]
    if datacard: parts += ["Data card:", datacard.strip()]
    if rule_changes: parts += ["Rulebook of this test (definitions that differ from the usual ones; they take precedence):", rule_changes.strip()]
    if industry_doc: parts += [industry_doc.strip()]
    return "\n".join(parts)


def render_example(ex):
    """one checked example: question (+ as-of date) and its labelled output."""
    from grammar import wrap, auto_plan
    return f"Q ({ex.get('as_of_date', '')}): {ex['question']}\n{wrap(ex['code'], ex.get('plan') or auto_plan(ex['code']))}"


def compact_hints(h):
    """the hint dict without empty fields, as one JSON line."""
    if not h: return ""
    keep = {k: v for k, v in h.items() if v not in (None, [], {}, "")}
    return json.dumps(keep, ensure_ascii=False, separators=(", ", ": "), default=str)


def question_block(examples, hints, question, as_of, variant="direct"):
    """# EXAMPLES + # HINTS + # QUESTION (+ # TASK of the variant): per question, after the shared prefix."""
    parts = []
    if examples: parts.append("# EXAMPLES (other shops: their names differ; use them for the method)\n" + "\n\n".join(render_example(e) for e in examples))
    hj = compact_hints(hints)
    if hj: parts.append("# HINTS (read by rules from the question and the data; may be incomplete or wrong)\n" + hj)
    parts.append(f"# QUESTION (today = {as_of})\n{question.strip()}")
    return "\n\n".join(parts) + VARIANT.get(variant, "")


def messages(system, business, qblock):
    """chat messages of one request (system shared by all shops; user = business prefix + per-question part)."""
    return [{"role": "system", "content": system}, {"role": "user", "content": business + "\n\n" + qblock}]


_TAG = re.compile(r"<(plan|code|python|sql)>\s*\n?(.*?)\n?\s*</\1>", re.S)


def parse_output(text):
    """the model's output -> {"plan", "code", "escape": None | "python" | "sql", "raw_code"}; code None when no block.
    Tolerant of a missing closing tag at the very end (max_tokens) and of a ```python fence (models without the grammar)."""
    t = text or ""
    if "</think>" in t: t = t.split("</think>", 1)[1]
    blocks = {m.group(1): m.group(2) for m in _TAG.finditer(t)}
    plan = blocks.get("plan", "")
    for tag in ("code", "python", "sql"):
        if tag in blocks: return {"plan": plan.strip(), "code": blocks[tag].strip("\n"), "escape": None if tag == "code" else tag}
    m = re.search(r"<(code|python|sql)>\s*\n?(.*)$", t, re.S)                # unterminated last block
    if m: return {"plan": plan.strip(), "code": m.group(2).strip("\n"), "escape": None if m.group(1) == "code" else m.group(1), "unterminated": True}
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", t, re.S)
    if m: return {"plan": plan.strip(), "code": m.group(1).strip("\n"), "escape": None, "fenced": True}
    return {"plan": plan.strip(), "code": None, "escape": None}
