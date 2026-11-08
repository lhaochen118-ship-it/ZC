#!/usr/bin/env python3
"""The <sql> escape of the hybrid flow (docs/08 WP12): one DuckDB SELECT written by the model over the shop's canonical tables
C (orders, lines, items, purchases, ...: the names of the # BUSINESS card) and its raw tables (raw_<name>), in process and
read-only (no file / network / extension access). The first row / column of the result is the answer; its type comes from the
answer column's name when it is a type name (AS number, AS percent, AS date, AS string, AS boolean, AS list), else from the
expected type of the question hints, else from the result's shape.

    sql_answer = make_sql_answer(C, T)  ;  sql_answer(query, expected_type=None) -> {"type", "answer"}"""
import re, math, datetime

TYPES = ("number", "percent", "date", "string", "boolean", "list", "table")
_FIRST = re.compile(r"^\s*(--[^\n]*\n\s*|/\*.*?\*/\s*)*(select|with|from|values)\b", re.I | re.S)
_DENY = re.compile(r"\b(copy|attach|detach|install|load|pragma|export|import|create|insert|update|delete|drop|alter|set|call|"
                   r"read_csv\w*|read_parquet|read_json\w*|read_text|read_blob|glob|httpfs|sniff_csv)\b", re.I)


def _safe(name):
    return re.sub(r"\W+", "_", str(name)).strip("_").lower() or "t"


def _scalar(v):
    if v is None: return None
    try:
        import pandas as pd
        if v is pd.NaT: return None
        if isinstance(v, pd.Timestamp): return v.strftime("%Y-%m-%d")
    except ImportError: pass
    if isinstance(v, (datetime.date, datetime.datetime)): return v.strftime("%Y-%m-%d")
    if hasattr(v, "item"): v = v.item()                                # numpy scalars
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
    return v


def _typed(df, expected=None):
    cols = [str(c) for c in df.columns]
    named = cols[0].lower() if cols and cols[0].lower() in TYPES else None
    t = named or (expected if expected in TYPES else None)
    if len(df.columns) > 1 and t not in ("list", "number", "percent", "date", "string", "boolean"):
        t = "table"
    if t == "table" or (t is None and len(df.columns) > 1):
        return {"type": "table", "answer": {"columns": cols, "rows": [[_scalar(v) for v in r] for r in df.itertuples(index=False)]}}
    first = [_scalar(v) for v in df.iloc[:, 0].tolist()] if len(df.columns) else []
    if t == "list" or (t is None and len(first) > 1):
        return {"type": "list", "answer": first}
    if not first: raise ValueError("the query returned no row")
    v = first[0]
    if t is None:
        if isinstance(v, bool): t = "boolean"
        elif isinstance(v, (int, float)): t = "number"
        elif isinstance(v, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", v): t = "date"
        else: t = "string"
    if t == "number" and isinstance(v, float) and v.is_integer(): v = int(v)
    if t == "boolean": v = bool(v)
    return {"type": t, "answer": v}


def make_sql_answer(C, T=None, threads=1):
    """sql_answer bound to one shop's tables (pandas DataFrames; never modified: DuckDB reads them)."""
    def sql_answer(query, expected_type=None):
        import duckdb
        q = str(query).strip().rstrip(";")
        if ";" in q: raise ValueError("one SELECT only")
        if not _FIRST.match(q): raise ValueError("the <sql> block must be one SELECT (or WITH ... SELECT) query")
        if _DENY.search(re.sub(r"'(?:[^']|'')*'", "''", q)): raise ValueError("only reading queries over the shop's tables are allowed")
        con = duckdb.connect(":memory:", config={"threads": threads})
        try:
            for k, v in (C or {}).items(): con.register(_safe(k), v)
            for k, v in (T or {}).items(): con.register("raw_" + _safe(k), v)
            con.execute("SET enable_external_access = false"); con.execute("SET lock_configuration = true")
            df = con.execute(q).fetchdf()
        finally:
            con.close()
        return _typed(df, expected_type)
    return sql_answer


def schema_lines(C, T=None, max_cols=14):
    """'orders(order_id, order_date, ...)' lines naming the tables a query may read (the prompt's # BUSINESS has the meanings)."""
    out = [f"{_safe(k)}({', '.join(map(str, list(v.columns)[:max_cols]))})" for k, v in (C or {}).items()]
    out += [f"raw_{_safe(k)}({', '.join(map(str, list(v.columns)[:max_cols]))})" for k, v in (T or {}).items()]
    return out
