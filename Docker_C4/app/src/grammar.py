#!/usr/bin/env python3
import re, json, hashlib
import dm, industry

ATYPES = ("number", "percent", "string", "date", "boolean", "list", "table", "unanswerable")
METHODS = ("sum", "mean", "median", "max", "min", "idxmax", "idxmin", "head", "tail", "tolist", "to_list", "astype", "fillna", "map", "isin",
           "nunique", "count", "unique", "sort_values", "sort_index", "round", "abs", "value_counts", "reset_index", "groupby", "agg", "apply",
           "between", "contains", "startswith", "endswith", "lower", "upper", "strip", "replace", "items", "keys", "values", "get", "notna",
           "isna", "notnull", "isnull", "any", "all", "cumsum", "diff", "nlargest", "nsmallest", "drop_duplicates", "copy", "to_dict", "strftime",
           "normalize", "weekday", "mask", "where", "dropna", "first", "last", "size", "std", "prod", "eq", "ne", "gt", "ge", "lt", "le",
           "isoformat", "date", "split", "join", "index", "pct_change", "clip", "quantile", "to_period", "to_timestamp", "loc", "query")
ATTRS = ("index", "values", "columns", "str", "dt", "iloc", "loc", "empty", "shape", "start", "end", "days", "year", "month", "day",
         "hour", "dayofweek", "name", "size", "is_unique", "T")
CANON_CHANNELS = ("tại quầy", "điện thoại", "zalo", "sàn TMĐT")
CANON_PAYMENTS = ("tiền mặt", "chuyển khoản", "ví điện tử", "cod", "ghi nợ", "thẻ")
STATUSES = ("hoàn thành", "huỷ", "trả hàng", "đang giao")
TOOL_KINDS = ("item", "customer", "staff", "supplier", "group", "channel", "payment", "vehicle", "customer_group", "unit")
ENT_KINDS = ("item", "group", "customer", "staff", "supplier", "channel", "payment", "status", "customer_group", "unit")


def _lit(s):
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t") + '"'


def _pystr_alts(vals, variants=True):
    out = []
    for v in vals:
        s = str(v)
        if not s or "\n" in s or "\r" in s: continue
        forms = [s] + ([dm.key(s)] if variants and dm.key(s) and dm.key(s) != s else [])
        for f in forms:
            out.append(_lit('"' + f.replace("\\", "\\\\").replace('"', '\\"') + '"'))
            out.append(_lit("'" + f.replace("\\", "\\\\").replace("'", "\\'") + "'"))
    return list(dict.fromkeys(out))


def entity_values(C):
    def col(tables, *cols):
        out = set()
        for t in tables:
            df = C.get(t)
            if df is None: continue
            for c in cols:
                if c in df.columns:
                    for x in df[c].dropna().unique():
                        s = str(x).strip()
                        if s and s.lower() != "nan" and len(s) <= 120: out.add(s)
        return out
    E = {}
    items = col(("lines", "items", "purchases", "lots", "stock", "prices"), "item_name", "item_code")
    lines_extra = set()
    for nm in col(("lines", "items"), "item_name"):
        try:
            ml = industry.model_line(nm)
            if ml and ml != nm: lines_extra.add(str(ml))
        except Exception: pass
    E["item"] = sorted(items | lines_extra)
    E["group"] = sorted(col(("lines", "items"), "item_group"))
    E["customer"] = sorted(col(("orders", "customers"), "customer_name", "customer_id"))
    E["staff"] = sorted(col(("orders", "staff"), "staff", "staff_name"))
    E["supplier"] = sorted(col(("purchases",), "supplier"))
    E["channel"] = sorted(col(("orders",), "channel") | set(CANON_CHANNELS))
    E["payment"] = sorted(col(("orders", "payments"), "payment_method", "method") | set(CANON_PAYMENTS))
    E["status"] = sorted(set(STATUSES))
    E["customer_group"] = sorted(col(("orders", "customers"), "customer_group"))
    E["unit"] = sorted(col(("lines", "items", "conversions", "purchases"), "unit", "base_unit", "big_unit", "unit_name"))
    E["column"] = sorted({str(c) for t, df in C.items() if hasattr(df, "columns") for c in df.columns})
    return E


_CACHE = {}

BASE = r'''
root ::= "<plan>" plan "</plan>\n" body
plan ::= [^<\n]*
body ::= code{ESC}
code ::= "<code>\n" stmt* final "\n</code>"
stmt ::= (ident ws "=" ws expr | "print(" args? ")") "\n"
final ::= "ANSWER" ws "=" ws "{" q "type" q ":" ws atype ws "," ws q "answer" q ":" ws expr ws "}" | "raise Unanswerable(" anystr? ")"
q ::= "\"" | "'"
atype ::= "\"" atypename "\"" | "'" atypename "'"
atypename ::= "number" | "percent" | "string" | "date" | "boolean" | "list" | "table" | "unanswerable"
pyblock ::= "<python>\n" free "</python>"
sqlblock ::= "<sql>\n" free "</sql>"
free ::= [^\x00]*
ws ::= " "?
sep ::= "," " "?
expr ::= lambda | ternary
lambda ::= "lambda " ident ("," ws ident)* ":" ws expr
ternary ::= or_e (" if " or_e " else " expr)?
or_e ::= and_e (" or " and_e)*
and_e ::= not_e (" and " not_e)*
not_e ::= "not " not_e | cmp
cmp ::= arith (ws cmpop ws arith)*
cmpop ::= "==" | "!=" | "<=" | ">=" | "<" | ">" | "in" | "not in" | "is" | "is not"
arith ::= term (ws addop ws term)*
addop ::= "+" | "-" | "&" | "|"
term ::= factor (ws mulop ws factor)*
mulop ::= "*" | "/" | "//" | "%"
factor ::= ("-" | "~" | "+") factor | power
power ::= primary trailer* (ws "**" ws factor)?
primary ::= call | ident | number | anystr | listx | parenx | dictx | "None" | "True" | "False"
trailer ::= "." method "(" args? ")" | "." attr | "[" subs "]"
subs ::= slice (sep slice)*
slice ::= expr? ":" expr? | expr
method ::= {METHODS}
attr ::= {ATTRS}
args ::= arg (sep arg)* ","?
arg ::= ident "=" expr | "*" expr | "**" expr | expr
listx ::= "[" ws (expr (sep expr)* ","? | expr " for " target " in " or_e (" if " or_e)*)? ws "]"
parenx ::= "(" ws (expr (sep expr)* ","? | expr " for " target " in " or_e (" if " or_e)*)? ws ")"
dictx ::= "{" ws (expr ":" ws expr (sep expr ":" ws expr)* ","? | expr ":" ws expr " for " target " in " or_e (" if " or_e)*)? ws "}"
target ::= ident ("," ws ident)* | "(" ident ("," ws ident)* ")"
ident ::= [a-zA-Z_] [a-zA-Z0-9_]*
number ::= [0-9]+ ("_" [0-9]+)* ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
anystr ::= "\"" ([^"\\\n] | "\\" [^\n])* "\"" | "'" ([^'\\\n] | "\\" [^\n])* "'"
int_e ::= expr
'''


def build_grammar(api_spec, entity_values, max_enum=5000, escapes=("python",)):
    key = hashlib.sha256(json.dumps([[e["name"], e["params"], e["filters"]] for e in api_spec], ensure_ascii=False, default=str).encode()
                         + json.dumps(entity_values, ensure_ascii=False, sort_keys=True).encode()
                         + f"{max_enum}|{','.join(escapes)}".encode()).hexdigest()
    if key in _CACHE: return _CACHE[key]
    esc = "".join({"python": " | pyblock", "sql": " | sqlblock"}[e] for e in escapes if e in ("python", "sql"))
    g = BASE.replace("{ESC}", esc).replace("{METHODS}", " | ".join(_lit(m) for m in sorted(METHODS, key=lambda m: (-len(m), m)))) \
            .replace("{ATTRS}", " | ".join(_lit(a) for a in sorted(ATTRS, key=lambda a: (-len(a), a))))
    rules = []
    def ent_rule(kind):
        vals = [v for v in (entity_values.get(kind) or []) if v]
        alts = _pystr_alts(vals) if 0 < len(vals) <= max_enum else []
        if kind == "status": alts = _pystr_alts(STATUSES)
        if not alts: return "anystr"
        rules.append(f"ent_{kind} ::= " + " | ".join(alts))
        return f"ent_{kind}"
    E = {k: ent_rule(k) for k in ENT_KINDS}
    findx = 'find(" "\\"" tkind "\\"" sep anystr (sep expr)? ")" "[" [0-9]+ "]"'
    rules.append("tkind ::= " + " | ".join(_lit(k) for k in TOOL_KINDS))
    def val(kind, fname=""):
        if kind in ("item", "group", "customer", "staff", "supplier", "customer_group"):
            e = E[kind]
            return f'({e} | "[" ws {e} (sep {e})* ws "]" | "find(" "\\"" tkind "\\"" sep anystr (sep expr)? ")" "[" [0-9]+ "]" | ident)'
        if kind in ("channel", "payment", "status", "unit"):
            e = E[kind]
            return f'({e} | "[" ws {e} (sep {e})* ws "]" | ident)'
        if kind == "by":
            from api_spec import BY
            vals = BY.get(fname)
            return "(" + " | ".join(_pystr_alts(vals)) + " | ident)" if vals else "expr"
        if kind == "shift": return '("\\"sáng\\"" | "\\"chiều\\"" | "\'sáng\'" | "\'chiều\'" | ident)'
        if kind == "lambda": return "(lambda | ident)"
        return "expr"
    calls = []
    for e in api_spec:
        n = e["name"]; ps = e["params"]
        rn = "c_" + n
        pos = [val(p["kind"], n) for p in ps]
        kws = [f'{_lit(p["name"] + "=")} {val(p["kind"], n)}' for p in ps]
        if e["filters"]:
            from api_spec import FILTERS
            kws += [f'{_lit(k + "=")} {val(kind, n)}' for k, kind in FILTERS.items() if k not in {p["name"] for p in ps}]
        alts = []
        if pos:
            chain = ""
            for i in range(len(pos), 0, -1):
                chain = pos[i - 1] + (f" (sep {chain})?" if chain else "")
            rules.append(f"{rn}_pos ::= {chain}")
        if kws: rules.append(f"{rn}_kw ::= " + " | ".join(kws))
        inner = []
        if pos and kws: inner.append(f"{rn}_pos (sep {rn}_kw)*")
        elif pos: inner.append(f"{rn}_pos")
        if kws: inner.append(f"{rn}_kw (sep {rn}_kw)*")
        body = " | ".join(inner)
        rules.append(f"{rn} ::= {_lit(n + '(')} ws ({body})? ws \")\"" if body else f"{rn} ::= {_lit(n + '(')} \")\"")
        calls.append(rn)
    from api_spec import BUILTINS
    rules.append("c_builtin ::= (" + " | ".join(_lit(b + "(") for b in sorted(BUILTINS, key=lambda b: (-len(b), b))) + ") ws args? ws \")\"")
    rules.append('c_misc ::= ("Unanswerable(" | "Period(" | "pd.Timestamp(" | "pd.to_datetime(" | "pd.DataFrame(" | "pd.Series(" | "pd.Timedelta(" | "np.round(" | "math.floor(" | "math.ceil(") ws args? ws ")"')
    rules.append("call ::= " + " | ".join(calls + ["c_builtin", "c_misc"]))
    g = g + "\n" + "\n".join(rules) + "\n"
    _CACHE[key] = g
    return g


def wrap(code, plan=""):
    return f"<plan>{' '.join(str(plan).split()).replace('<', ' ')}</plan>\n<code>\n{code.strip()}\n</code>"


def auto_plan(code):
    fns = [f for f in re.findall(r"\b([a-z_][a-z0-9_]*)\(", code) if f not in ("lambda",)]
    from api_spec import GROUP_OF
    per = [f for f in fns if GROUP_OF.get(f) == "period"]
    main = next((f for f in fns if GROUP_OF.get(f) not in ("period", "number")), fns[0] if fns else "")
    filt = re.findall(r"\b(item|group|customer|customer_group|staff|channel|payment|status|shift|where|by|unit)=", code)
    return f"{main} | {', '.join(dict.fromkeys(per)) or '-'} | {', '.join(dict.fromkeys(filt)) or '-'}"


class Checker:
    def __init__(self, grammar):
        import xgrammar as xgr
        comp = xgr.GrammarCompiler(xgr.TokenizerInfo([]), cache_enabled=False)
        self.m = xgr.GrammarMatcher(comp.compile_grammar(xgr.Grammar.from_ebnf(grammar)), terminate_without_stop_token=True)

    def __call__(self, text):
        self.m.reset()
        return bool(self.m.accept_string(text)) and self.m.is_terminated()


def matcher_accepts(grammar, text):
    return Checker(grammar)(text)
