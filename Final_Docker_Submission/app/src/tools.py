#!/usr/bin/env python3
"""Agent tools T1-T9 (docs/08_MASTER_PLAN.md §5.5): deterministic lookups the model calls INSIDE its program (CodeAct style)
or that the flow calls for a repair message. Each output is cut to about 300 tokens. Logic only: a tool never answers.

    bind_tools(shop, today, rulebook="", data_end=None, spec=None) -> {name: function}   (added to the program namespace)
T1 find(kind, text, k=5)      data values matching text: exact, accent-free, regional word, abbreviation / model line, typo
T2 values(table, column, contains=None, k=20)      distinct values with counts
T3 period(text, today=TODAY)  the Period a time phrase means (qparse.parse_period); error with valid examples otherwise
T4 rule(term)                 rulebook definition of a term
T5 schema(table=None), peek(table, n=3, where=None)
T6 explain(code)              the program run with filter tracing (rows left after each filter)
T7 check(answer, expected_type=None)   postcheck flags (§5.6)
T8 help(name)                 signature + description + one checked example
T9 has_data(concept)          whether the shop records a kind of information, and where"""
import re, difflib, json
import pandas as pd
import dm, industry

MAX_CHARS = 900


def _cut(s, n=MAX_CHARS):
    s = str(s)
    return s if len(s) <= n else s[:n - 1] + "…"


class Matches(list):
    """find() result: a list of data values (best first; find(...)[0] is a string usable as a filter value) whose printed
    form also shows how each matched, its score and its number of rows."""
    def __init__(self, info):
        super().__init__([v for v, _, _, _ in info]); self.info = info
    def __repr__(self):
        if not self.info: return "[] (no match: the data has no such value; check the spelling or raise Unanswerable)"
        return _cut("[" + ", ".join(f"({v!r}, {how}, {sc:.2f}, {n} rows)" for v, how, sc, n in self.info) + "]")
    __str__ = __repr__


class Text(str):
    """a tool's text output (prints as is)."""
    def __repr__(self): return str(self)


KIND_COLS = {"item": [("lines", ("item_name", "item_code")), ("items", ("item_name", "item_code")), ("purchases", ("item_name", "item_code"))],
             "group": [("lines", ("item_group",)), ("items", ("item_group",))],
             "customer": [("orders", ("customer_name", "customer_id")), ("customers", ("customer_name", "customer_id"))],
             "staff": [("orders", ("staff",)), ("staff", ("staff", "staff_name"))],
             "supplier": [("purchases", ("supplier",))], "channel": [("orders", ("channel",))], "payment": [("orders", ("payment_method",))],
             "customer_group": [("orders", ("customer_group",)), ("customers", ("customer_group",))],
             "unit": [("lines", ("unit", "base_unit")), ("items", ("base_unit",)), ("conversions", ("unit", "base_unit"))],
             "vehicle": [("orders", ("vehicle", "vehicle_line"))], "ingredient": [("lines", ("active_ingredient",)), ("items", ("active_ingredient",))]}
CONCEPTS = [("tồn kho", ("stock", "lots", "purchases"), ()), ("giá vốn", (), ("unit_cost", "cost")), ("nợ", ("payments",), ("debt_end", "paid_total", "due_date")),
            ("nhân viên", ("staff",), ("staff",)), ("ca", (), ("order_time",)), ("kênh", (), ("channel",)), ("khách hàng", ("customers",), ("customer_name", "customer_id")),
            ("nhà cung cấp", (), ("supplier",)), ("nhập hàng", ("purchases",), ()), ("lô", ("lots",), ("expiry",)), ("hạn dùng", ("lots",), ("expiry",)),
            ("bảo hành", ("warranty",), ("warranty",)), ("trả góp", ("instalments",), ()), ("thu cũ đổi mới", ("tradeins",), ("tradein",)),
            ("phiếu sửa", (), ("work", "plate")), ("chi phí", ("expenses",), ()), ("giao hàng", ("deliveries",), ("delivery_date",)),
            ("giá bán", ("prices",), ("unit_price", "price")), ("nhóm hàng", (), ("item_group",)), ("hoạt chất", (), ("active_ingredient",)),
            ("thành viên", ("customers",), ("registered",)), ("voucher", (), ("order_discount", "voucher")), ("phí ship", (), ("shipping_fee",)),
            ("giới tính", (), ("gender", "customer_gender")), ("địa chỉ", (), ("address", "district")), ("lý do", (), ("reason", "cancel_reason", "return_reason"))]


def _tables(shop):
    C = dict(shop.C); C.setdefault("orders", shop.O)
    if shop.L is not None: C.setdefault("lines", shop.L)
    return C


def bind_tools(shop, today=None, rulebook="", data_end=None, spec=None):
    """the tool functions bound to one shop and day."""
    import api_spec, hints as hintsmod
    C = _tables(shop)
    _vals = {}
    def vals(kind):
        if kind not in _vals:
            cnt = {}
            for t, cols in KIND_COLS.get(kind, []):
                df = C.get(t)
                if df is None: continue
                for c in cols:
                    if c in df.columns:
                        for v, n in df[c].dropna().astype(str).value_counts().items():
                            if v.strip() and v.lower() != "nan": cnt[v] = max(cnt.get(v, 0), int(n))
            _vals[kind] = cnt
        return _vals[kind]

    def find(kind, text, k=5):
        """T1: data values of kind matching text, best first."""
        kind = {"product": "item", "hang": "item", "mat hang": "item", "khach": "customer", "nhan vien": "staff", "ncc": "supplier",
                "nha cung cap": "supplier", "kenh": "channel", "thanh toan": "payment", "nhom": "group"}.get(dm.key(kind), kind)
        if kind not in KIND_COLS: raise ValueError(f"find(kind, ...): kind is one of {', '.join(KIND_COLS)}")
        V = vals(kind); t = str(text).strip(); tk = dm.key(t); out = {}
        def add(v, how, sc):
            if v not in out or out[v][1] < sc: out[v] = (how, sc)
        for v in V:
            vk = dm.key(v)
            if v.lower() == t.lower(): add(v, "exact", 1.0)
            elif vk == tk: add(v, "no_accent", 0.98)
            elif dm.regional_key(v) == dm.regional_key(t) or dm.regional_key_plain(v) == dm.regional_key_plain(t): add(v, "regional", 0.95)
            elif tk and (vk.startswith(tk + " ") or (kind == "item" and dm.key(industry.model_line(v) or "") == tk)): add(v, "abbrev", 0.9)
            elif tk and len(tk) >= 3 and re.search(r"(^|\s)" + re.escape(tk) + r"(\s|$)", vk): add(v, "contains", 0.8)
        if not out:
            keys = {dm.key(v): v for v in V}
            for kk in difflib.get_close_matches(tk, list(keys), n=k, cutoff=0.6):
                add(keys[kk], "typo", round(difflib.SequenceMatcher(None, tk, kk).ratio(), 2))
        ranked = sorted(out.items(), key=lambda x: (-x[1][1], -V.get(x[0], 0), x[0]))[:k]
        return Matches([(v, how, sc, V.get(v, 0)) for v, (how, sc) in ranked])

    def values(table, column, contains=None, k=20):
        """T2: distinct values of C[table][column] with their counts."""
        df = C.get(table)
        if df is None: return Text(f"no table {table!r}; tables: {', '.join(sorted(C))}")
        if column not in df.columns:
            near = difflib.get_close_matches(column, list(map(str, df.columns)), n=5, cutoff=0.4)
            return Text(f"no column {column!r} in {table}; columns: {', '.join(map(str, df.columns))[:500]}" + (f"; closest {near}" if near else ""))
        s = df[column].dropna().astype(str).value_counts()
        if contains: s = s[[dm.key(contains) in dm.key(v) for v in s.index]]
        return Text(_cut(f"{table}.{column}: {len(s)} values: " + ", ".join(f"{v!r} x{n}" for v, n in s.head(k).items())))

    def period(text, today=None):
        """T3: the Period of a Vietnamese time phrase."""
        import qparse
        td = pd.Timestamp(today if today is not None else (shop.today if shop.today is not None else pd.Timestamp.now())).normalize()
        p, code = qparse.parse_period(str(text), td)
        if p is None:
            raise ValueError(f"period(): cannot read {text!r}; use e.g. 'tháng 7/2025', 'quý 3/2025', 'tuần trước', '7 ngày qua', 'hôm qua', "
                             "'từ 01/07/2025 đến 15/07/2025', 'năm 2025', or the period functions (month(2025, 7), last_week(TODAY), ...)")
        return p

    rows = hintsmod.rulebook_rows(rulebook) if rulebook else {}
    def rule(term):
        """T4: rulebook definition(s) of a term."""
        tk = " ".join(dm.key(term).split())
        hits = [(shown, d) for k, (shown, d) in rows.items() if tk and (tk in k or k in tk)]
        if not hits:
            near = difflib.get_close_matches(tk, list(rows), n=3, cutoff=0.6)
            hits = [rows[k] for k in near]
        return Text(_cut("\n".join(f"{s}: {d}" for s, d in hits[:4]) or f"the rulebook defines no term like {term!r}"))

    def schema(table=None):
        """T5: columns of the canonical tables (or one table)."""
        import prompting
        sub = {table: C[table]} if table in C else C
        if table and table not in C: return Text(f"no table {table!r}; tables: {', '.join(sorted(C))}")
        return Text(_cut(prompting.compact_card(sub, "", max_chars=MAX_CHARS * 2), MAX_CHARS * 2))

    def peek(table, n=3, where=None):
        """T5: a few rows of a table (where: a function of the frame -> bool Series)."""
        df = C.get(table)
        if df is None: return Text(f"no table {table!r}; tables: {', '.join(sorted(C))}")
        if where is not None: df = df[where(df)]
        cols = [c for c in df.columns if not str(c).startswith("_") and not str(c).endswith("_raw")][:14]
        return Text(_cut(df[cols].head(int(n)).to_string(max_colwidth=30)))

    def explain(code):
        """T6: run a program with filter tracing (rows after each filter)."""
        shop.trace = []
        try:
            r = dm.run_program(code, shop, today if today is not None else shop.today, extra=bound)
            out = [f"ANSWER = {r!r}"[:300]]
        except Exception as e: out = [f"{type(e).__name__}: {e}"[:300]]
        finally:
            tr = list(getattr(shop, "trace", None) or []); shop.trace = None
        for t in tr[:20]: out.append(f"- {t}")
        zero = [t for t in tr if " -> 0 rows" in t]
        if zero: out.append("WARNING: a filter matched nothing: " + "; ".join(zero[:3]))
        return Text(_cut("\n".join(out), MAX_CHARS * 2))

    def check(answer, expected_type=None):
        """T7: postcheck flags of an ANSWER dict (empty list = nothing suspicious)."""
        return postcheck(answer, expected_type, shop)

    sp = spec or api_spec.build_api_spec()
    def help(name):
        """T8: signature, description and one checked example of a function."""
        return Text(api_spec.help_text(sp, getattr(name, "__name__", str(name))))

    def has_data(concept):
        """T9: does the shop record this kind of information, and where."""
        ck = dm.key(concept)
        for name, tabs, cols in CONCEPTS:
            if dm.key(name) in ck or ck in dm.key(name):
                where = [t for t in tabs if C.get(t) is not None and len(C[t])]
                where += [f"{t}.{c}" for t, df in C.items() for c in cols if hasattr(df, "columns") and c in df.columns and df[c].notna().any()]
                if name == "ca" and "orders.order_time" in where:
                    ot = pd.to_datetime(C["orders"]["order_time"], errors="coerce").dropna()
                    if not len(ot) or ((ot.dt.hour == 0) & (ot.dt.minute == 0)).mean() > 0.98: where.remove("orders.order_time")
                return Text(f"{name}: " + (f"recorded in {', '.join(sorted(set(where)))}" if where else "NOT recorded in this shop's data (questions needing it are unanswerable)"))
        return Text(f"unknown concept {concept!r}; known: {', '.join(n for n, _, _ in CONCEPTS)}")

    bound = {"find": find, "values": values, "period": period, "rule": rule, "schema": schema, "peek": peek, "explain": explain,
             "check": check, "help": help, "has_data": has_data}
    return bound


_COUNT_Q = re.compile(r"\b(?:bao nhieu|may|bn|so) (?:don|bill|hoa don|lan|khach|luot|phieu|lo|mat hang|san pham|dot|chiec|xe|ngay|nguoi)\b")


def postcheck(answer, expected_type=None, shop=None, question_key=None, k_asked=None):
    """T7 / §5.6: sanity flags of an executed answer (list of short strings; empty = nothing suspicious)."""
    flags = []
    if not isinstance(answer, dict): return ["ANSWER is not a dict"]
    t, v = answer.get("type"), answer.get("answer")
    benign = (expected_type == "number" and t in ("percent", "list")) or (expected_type == "string" and t == "number")
    if expected_type and t != expected_type and not benign and t != "unanswerable":
        flags.append(f"type {t} but the question asks {expected_type}")
    if t == "unanswerable": return flags
    if v is None: return flags + ["empty answer"]
    if t in ("number", "percent") and isinstance(v, (int, float)) and not isinstance(v, bool):
        if t == "percent" and not (-1000 <= v <= 1000): flags.append(f"percentage {v} out of range")
        if v == 0: flags.append("zero result (a filter may match nothing)")
    if t == "list":
        if not v: flags.append("empty list")
        elif all(isinstance(x, str) for x in v) and len(v) != len(set(v)): flags.append("duplicates in list")
        if k_asked and isinstance(v, list) and len(v) > k_asked: flags.append(f"list of {len(v)}, question asks {k_asked}")
    if t == "table" and isinstance(v, dict) and not v.get("rows"): flags.append("empty table")
    if t == "date" and shop is not None and isinstance(v, str) and len(v) == 10 and shop.first is not None:
        try:
            d = pd.Timestamp(v)
            if d < shop.first - pd.Timedelta(days=400) or d > shop.last + pd.Timedelta(days=400): flags.append(f"date {v} far outside the data")
        except Exception: flags.append("date not parseable")
    if t == "string" and isinstance(v, str) and not v.strip(): flags.append("empty string")
    return flags
