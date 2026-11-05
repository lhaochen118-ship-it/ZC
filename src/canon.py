#!/usr/bin/env python3
"""Canonical stage: raw tables T of one business -> canonical tables C (dm.CANON_DOC) + NOTES.

Two sources of candidate mapping programs, all executed, checked (dm.check) and scored the same way:
  1. transfer: every reference program in refs/compile/ (written for the train businesses) is run on the new business.
     Table names are resolved by columns (an export with the same layout but other file / sheet names still maps),
     so a known export format needs no LLM at all. Programs are generic code over column names; nothing is keyed on
     business ids.
  2. LLM: for formats no reference fits, the LLM writes a mapping program with the same small API, shown the two most
     similar reference programs (by table / column overlap) as examples.

Usage (offline check of the transfer step): canon.py <split_dir> <cache_dir> [bid ...]"""
import os, re, sys, json, glob, keyword, builtins, pickle, difflib, traceback, unicodedata
import numpy as np
import pandas as pd


def _lenient_fillna(orig):
    """fillna(None) / fillna() is a no-op instead of "Must specify a fill 'value' or 'method'" (28 of the 92 failed
    mapping attempts of v4b): a missing value stays missing, which is what such code means."""
    def fillna(self, value=None, *args, **kw):
        if value is None and not args and kw.get("method") is None and kw.get("limit") is None:
            return None if kw.get("inplace") else self.copy()
        return orig(self, value, *args, **kw)
    fillna.__wrapped__ = orig
    return fillna


for _cls in (pd.Series, pd.DataFrame):
    if not hasattr(_cls.fillna, "__wrapped__"): _cls.fillna = _lenient_fillna(_cls.fillna)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dm, ingest, industry

REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refs", "compile")
META = os.path.join(REF_DIR, "_meta.json")


FMT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refs", "formats")


def ref_programs(formats=True):
    """reference mapping programs: one per train business (refs/compile) + generic format adapters (refs/formats, written
    with tbl() so they apply to any export of that layout)."""
    out = {os.path.basename(f)[:-3]: open(f, encoding="utf-8").read() for f in sorted(glob.glob(os.path.join(REF_DIR, "*.py")))}
    if formats: out.update({"fmt:" + os.path.basename(f)[:-3]: open(f, encoding="utf-8").read() for f in sorted(glob.glob(os.path.join(FMT_DIR, "*.py")))})
    hide = {x for x in os.environ.get("ZC_HIDE_REFS", "").split(",") if x}   # tests of the LLM mapping path hide fmt:automap
    return {k: v for k, v in out.items() if k not in hide}


def used_tables(code):
    return list(dict.fromkeys(re.findall(r'(?:T|RAW)\["([^"]+)"\]', code)))


def optional_tables(code):
    """tables read only inside try: blocks (optional parts of a mapping)."""
    import ast
    inside, outside = set(), set()
    def visit(node, in_try):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in ("T", "RAW") \
                and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            (inside if in_try else outside).add(node.slice.value)
        for ch in ast.iter_child_nodes(node): visit(ch, in_try or isinstance(node, ast.Try))
    visit(ast.parse(code), False)
    return sorted(inside - outside)


def build_meta(cache_dir):
    """columns of every table each reference program reads (from the train cache), so that the tables can be matched
    by columns in another business. Written once to refs/compile/_meta.json."""
    meta = {}
    for bid, code in ref_programs(formats=False).items():
        p = os.path.join(cache_dir, bid + ".pkl")
        if not os.path.isfile(p): continue
        biz = pickle.load(open(p, "rb"))
        meta[bid] = {"tables": {t: [c for c in biz.tables[t].columns if c not in ("_source", "_sheet")] for t in used_tables(code) if t in biz.tables},
                     "optional": optional_tables(code),
                     "doc": "\n".join(l[2:] for l in code.splitlines() if l.startswith("# "))}
    json.dump(meta, open(META, "w"), ensure_ascii=False, indent=0)
    return meta


def load_meta(): return json.load(open(META, encoding="utf-8")) if os.path.isfile(META) else {}


class Tables(dict):
    """T / RAW for a transferred program: a reference table name resolves to the business's table whose columns cover
    the reference table's columns best (>= 70 %)."""
    def __init__(self, data, ref_cols):
        super().__init__(data); self.ref_cols = ref_cols; self.used = {}; self.accessed = set()

    def __getitem__(self, name):                                        # every table a program reads (completeness check)
        if dict.__contains__(self, name): self.accessed.add(name); return dict.__getitem__(self, name)
        return self.__missing__(name)

    def get(self, name, default=None):
        try: return self[name]
        except KeyError: return default

    def sheet_alias(self, name):
        """a table named after its sheet as the README writes it ('BanHang_KhachHang', a sheet of BanHang_TongHop_2025.xlsx
        that ingest calls BanHang_TongHop_2: the KeyError of most dev_003 mapping attempts) -> that table, when exactly one
        table comes from a sheet of that name (also 'file_sheet' / 'file[sheet]')."""
        k, hits = _fold(str(name)), set()
        for t in self:
            df = dict.__getitem__(self, t)
            if not isinstance(df, pd.DataFrame) or "_source" not in df.columns: continue
            for src in pd.unique(df["_source"].astype(str))[:20]:
                m = re.match(r"(.+?)(?:\.\w+)?\[(.+)\]$", src)
                if m and k in (_fold(m.group(2)), _fold(f"{m.group(1)}_{m.group(2)}"), _fold(f"{m.group(1)}[{m.group(2)}]")): hits.add(t)
        return hits.pop() if len(hits) == 1 else None

    def __missing__(self, name):
        cols = set(self.ref_cols.get(name, []))
        if not cols:
            alias = self.sheet_alias(name)
            if alias is None: raise KeyError(name)
            self.used[name] = alias; self.accessed.add(alias); return dict.__getitem__(self, alias)
        best, score = None, 0.0
        for t, df in self.items():
            s = len(cols & set(df.columns)) / len(cols)
            if s > score: best, score = t, s
        if best is None or score < 0.7: raise KeyError(f"{name} (no table with its columns)")
        self.used[name] = best; self.accessed.add(best)
        return dict.__getitem__(self, best)


def data_end(biz):
    m = re.search(r"có đến hết (\d{1,2})/(\d{1,2})/(\d{4})", biz.readme or "")
    return pd.Timestamp(int(m.group(3)), int(m.group(2)), int(m.group(1))) if m else None


def namespace(biz, ref_cols=None):
    T = Tables({k: v.copy() for k, v in biz.tables.items()}, ref_cols or {})
    R = Tables(dict(biz.raw), ref_cols or {})
    g = {"T": T, "RAW": R, "README": biz.readme or "", "pd": pd, "np": np, "re": re, "parse_num": ingest.parse_num, "parse_date": ingest.parse_date,
         "BIZ": biz}                                                   # the whole package (refs/formats/automap.py)
    for k in ("finalize", "map_status", "canon_status", "canon_payment", "canon_channel", "key", "is_walkin", "money", "round_half_up",
              "readme_conversions", "conversions_table", "frame", "money_in_text", "debt_from_note"):
        g[k] = getattr(dm, k)
    g["tbl"] = lambda *c, **kw: dm.find_table(T, *c, **kw)
    import industry                                                    # split_vehicle / model_line / brand_of / split_jobs
    g.update(industry.HELPERS)
    for t in T:                                                        # `thong_tin_thuoc` used as a variable (NameError in
        if t.isidentifier() and not keyword.iskeyword(t) and t not in g and t not in TABLE_NAMES and not hasattr(builtins, t):   # v6-v9)
            g[t] = dict.__getitem__(T, t)                              # (not T[t]: that would count every table as used)
    return g


TABLE_NAMES = {"orders", "lines", "items", "payments", "purchases", "stock", "lots", "lot_sales", "ledger", "prices", "conversions",
               "returns", "deliveries", "expenses", "staff", "customers", "C", "NOTES", "tradeins", "instalments", "instalment_payments",
               "warranty"}

CORE = {"orders": ["order_id", "order_date", "order_time", "customer_id", "customer_name", "customer_group", "staff", "channel",
                   "payment_method", "status", "total_payable"],
        "lines": ["item_code", "item_name", "item_group", "unit", "qty", "qty_base", "unit_price", "line_discount", "line_amount", "unit_cost"]}


def coverage(C):
    """how much of the business the canonical tables describe: filled canonical columns + extra tables."""
    s = 0.0
    for t, cols in CORE.items():
        df = C.get(t)
        if df is not None: s += sum(1 for c in cols if c in df.columns and df[c].notna().mean() > 0.05)
    return s + sum(1 for t, v in C.items() if t not in CORE and isinstance(v, pd.DataFrame) and len(v))


def plausibility(C, biz):
    """content checks of a mapping that ran: silent mistakes (wrong date conversion, swapped columns, wrong status
    labels) become errors with a hint, so that the repair round can fix them instead of every answer being wrong.
    Thresholds are conservative: every reference mapping of the training split passes (tests/test_plausibility.py)."""
    errs = []
    o, l = C.get("orders"), C.get("lines")
    if o is None or not len(o): return errs
    end = data_end(biz)
    d = pd.to_datetime(o["order_date"], errors="coerce").dropna() if "order_date" in o else pd.Series(dtype="datetime64[ns]")
    if len(d):
        if end is not None and (d.max() > end + pd.Timedelta(days=2) or d.max() < end - pd.Timedelta(days=60)):
            errs.append(f"orders end on {d.max().date()} but the README says the data runs until {end.date()}: check the order date "
                        "column and its conversion (Excel serial numbers count days from 1899-12-30)")
        if end is not None and d.min() < end - pd.Timedelta(days=5 * 366):
            errs.append(f"orders start on {d.min().date()}, more than 5 years before the end of the data ({end.date()}): check the date column")
        if end is None and (d.min().year < 2000 or d.max().year > 2100):
            errs.append(f"order dates run from {d.min().date()} to {d.max().date()}: check the date column and its conversion")
    if "valid" in o:
        share = float(o["valid"].astype(bool).mean())
        if share < 0.5:
            errs.append(f"only {share:.0%} of the orders are valid (not cancelled / returned): check which status labels map to huỷ / trả hàng")
        if "revenue" in o:
            rv = pd.to_numeric(o.loc[o["valid"].astype(bool), "revenue"], errors="coerce").fillna(0)
            if len(rv) and (rv <= 0).mean() > 0.2:
                errs.append(f"{(rv <= 0).mean():.0%} of the valid orders have revenue <= 0: check line_amount (the line total after the line discount)")
    if l is not None and len(l) and len(o) and len(l) / len(o) > 30:
        errs.append(f"{len(l) / len(o):.0f} lines per order on average: order_id is probably not an order code (a date, status or "
                    "name column?); when the file has no order code, each row (or each group of rows of one order) is one order")
    if l is not None and {"order_id", "line_amount"} <= set(l.columns) and "revenue" in o and len(o) and len(l) / len(o) > 1.2:
        sale = pd.to_numeric(l["qty"], errors="coerce").fillna(1) > 0 if "qty" in l.columns else pd.Series(True, index=l.index)
        ls = pd.to_numeric(l.loc[sale, "line_amount"], errors="coerce").groupby(l.loc[sale, "order_id"]).sum()
        v = o[o["valid"].astype(bool)] if "valid" in o else o
        rv = pd.to_numeric(v["revenue"], errors="coerce"); lv = v["order_id"].map(ls)
        ok = rv.notna() & lv.notna()
        if ok.sum() > 50:
            off = ((rv[ok] - lv[ok]).abs() > (1 + 0.005 * lv[ok].abs())).mean()
            if off > 0.2:
                errs.append(f"order revenue differs from the sum of its line amounts on {off:.0%} of the orders: rulebook §1 revenue is the sum "
                            "of the line amounts (after line discounts); map an order total (after order discount / with shipping) to "
                            "total_payable, not revenue")
    if "customer_name" in o:
        cn = o["customer_name"].dropna().astype(str).str.strip()
        if len(cn) > 50 and cn.str.fullmatch(r"-?[\d.,]+").mean() > 0.5:
            errs.append("customer_name holds numbers: map the column with the customer's NAME (or the customer list) to customer_name")
    if l is not None and {"qty", "unit_price", "line_amount"} <= set(l.columns):
        x = l[["qty", "unit_price", "line_amount"]].apply(pd.to_numeric, errors="coerce").dropna()
        x = x[(x["qty"] > 0) & (x["unit_price"] > 0) & (x["line_amount"] > 0)]
        if len(x) > 50:
            r = float((x["line_amount"] / (x["qty"] * x["unit_price"])).median())
            if not 0.3 <= r <= 1.5:
                errs.append(f"median line_amount / (qty x unit_price) is {r:.2f}, expected about 1: check the qty, unit_price and line_amount columns")
    if l is not None and {"qty", "qty_base"} <= set(l.columns):
        x = l[["qty", "qty_base"]].apply(pd.to_numeric, errors="coerce").dropna(); x = x[x["qty"] > 0]
        if len(x) > 50 and (x["qty_base"] < x["qty"] - 1e-9).mean() > 0.05:
            errs.append("qty_base is smaller than qty on many lines: qty_base must be the quantity in BASE units (qty x unit factor)")
    return errs


def _fold(x):
    x = unicodedata.normalize("NFD", str(x)).replace("đ", "d").replace("Đ", "D")
    return re.sub(r"[^a-z0-9]", "", "".join(c for c in x if unicodedata.category(c) != "Mn").lower())


def key_hint(e, biz):
    """for a KeyError on a table / column name: the closest real names (accents, case and spacing aside), so that the
    repair round can fix it (43 of the 92 failed mapping attempts of v4b were wrong column names)."""
    if not isinstance(e, KeyError) or not e.args: return ""
    k = str(e.args[0]).split(" (")[0]
    names = {}
    for t, df in biz.tables.items():
        names.setdefault(_fold(t), (t, None))
        for c in df.columns:
            if not str(c).startswith("_"): names.setdefault(_fold(c), (str(c), t))
    close = difflib.get_close_matches(_fold(k), list(names), n=5, cutoff=0.6)
    if not close: return ""
    return " | closest names in the data: " + ", ".join(f"{names[c][0]!r} ({'table' if names[c][1] is None else 'column of ' + names[c][1]})" for c in close)


def skip_hint(e, biz):
    """a KeyError on a canonical column that frame() skipped because its raw column does not exist (payments['pay_time']
    after frame(q, pay_time='NgayThu')): name the skipped raw column and the closest real ones."""
    if not isinstance(e, KeyError) or not e.args: return ""
    k = str(e.args[0]); sk = [raw for c, raw in dm.frame.skipped if c == k]
    if not sk: return ""
    cols = sorted({str(c) for df in biz.tables.values() for c in df.columns})
    close = difflib.get_close_matches(_fold(sk[-1]), {_fold(c): c for c in cols}, n=3, cutoff=0.5)
    real = {_fold(c): c for c in cols}
    return (f" | frame() skipped {k}={sk[-1]!r}: the table has no column {sk[-1]!r}, so {k} was never created"
            + (f" (closest columns: {', '.join(repr(real[c]) for c in close)})" if close else ""))


REV_NAMES = {"tienhang", "tienban", "thanhtienhang", "doanhthu"}       # rulebook §1: "tiền hàng, tiền bán ... cùng nghĩa với doanh thu"


def _num(s): return pd.to_numeric(s.map(lambda v: ingest.parse_num(v) if isinstance(v, str) else v), errors="coerce")


OWING_MAX = 0.25        # share of orders owing money at the end of the data above which a mapping is implausible (train refs <= 0.4%)


def warning_penalty(w):
    """coverage points a soft warning costs when candidate mappings are ranked: a wrong revenue column breaks most
    answers (always ranked below), a missing secondary table only its own questions."""
    w = str(w)
    return 100.0 if w.startswith("orders.revenue differs") else 20.0 if w.startswith(("lines.item_name holds the short names", "lines.item_name is empty on", "the device kinds of C disagree", "the data has an instalment book",
                                                                                      "C['instalments'] lacks", "C['instalment_payments'] lacks", "the data records number plates", "orders.plate matches the plates",
                                                                                      "the data has an instalment book")) \
        or re.match(r"orders\.\w+ (?:is -?\d+ on every order|adds up to|has the single value)|C counts \d+ of \d+ orders .* as still owing|.+? marks \d+ orders as (?:returned|cancelled) \(", w) \
        else 10.0 if w.startswith(("orders.order_time has no time", "orders.staff holds given names", "orders.staff does not follow")) \
        or re.match(r"the notes \S+ say \d+ orders (?:on credit are settled|still owe money)|the data writes the vehicle model with its plate|the data has a debt book|items\.\w+ is 0 on every item", w) else 2.0


def rank_score(r):
    """ranking of a candidate mapping that runs: coverage minus its warnings (pipeline.select, replay)."""
    return float(r.get("coverage") or 0) - sum(warning_penalty(w) for w in r.get("warnings") or [])


def _idn(s, norm=True):
    """order ids as text ('1', 1, 1.0 -> '1' when norm)."""
    s = s.astype(str).str.strip()
    return s.str.replace(r"\.0$", "", regex=True) if norm else s


def _order_key_col(df, ids, exclude=(), norm=True):
    """the column of raw table df that holds the order ids of C (ids: _idn of orders.order_id): >= 90% of its values are
    order ids AND it has many distinct values (a quantity column 1, 2, 3 also 'matches' ids 1, 2, 3: public dev_014)."""
    for kc in df.columns:
        if kc in exclude or str(kc).startswith("_"): continue
        h = _idn(df[kc].dropna().head(500), norm)
        if len(h) < 20 or h.nunique() < max(10, 0.2 * len(h)): continue
        if h.isin(ids).mean() > 0.9: return kc
    return None


def secondary_checks(C, biz):
    """soft content checks (warnings, not errors: the mapping is kept): they are sent back once in the completeness round
    and rank candidate mappings (fewer warnings first). Every reference mapping of the training split gives none
    (tests/test_plausibility.py)."""
    w = []
    o, l = C.get("orders"), C.get("lines")
    if o is None or not len(o) or "order_id" not in o: return w
    ids = set(o["order_id"].astype(str).str.strip())
    v = o[o["valid"].astype(bool)] if "valid" in o else o
    # 1. revenue against a column the data itself calls tiền hàng / tiền bán / doanh thu (one value per order), by its name or
    #    by the README's description of it ('**Giá trị hàng**: tiền hàng của đơn ...': public dev_009, 2 of 3 runs took TongTien)
    rdesc = {_fold(m.group(1)) for m in re.finditer(r"(?m)^\s*[-*]\s*\*\*(.+?)\*\*\s*:\s*(.+)$", str(getattr(biz, "readme", "") or ""))
             if re.match(r"(?:tienhang|tienban|doanhthu)", _fold(m.group(2)))}
    if "revenue" in o and len(v) > 50:
        rv = pd.to_numeric(v["revenue"], errors="coerce"); vid = v["order_id"].astype(str).str.strip()
        for t, df in biz.tables.items():
            rcs = [c for c in df.columns if _fold(c) in REV_NAMES or _fold(c) in rdesc]
            if not rcs or len(df) < 50: continue
            for kc in df.columns:
                if kc in rcs or df[kc].head(200).astype(str).str.strip().isin(ids).mean() < 0.9: continue   # cheap pre-check
                kv = df[kc].astype(str).str.strip()
                if kv.isin(ids).mean() < 0.9 or kv.duplicated().mean() > 0.05: continue   # one row per order
                for rc in rcs:                                         # (exported duplicates of a few rows aside)
                    m = pd.Series(_num(df[rc]).values, index=kv.values).groupby(level=0).first()
                    x = vid.map(m); ok = rv.notna() & x.notna()
                    if ok.sum() < 50: continue
                    off = ((rv[ok] - x[ok]).abs() > (1 + 0.005 * x[ok].abs())).mean()
                    tot = float(rv[ok].sum() / x[ok].sum() - 1) if x[ok].sum() else 0.0
                    if off > 0.02 or abs(tot) > 0.005:                 # the reference mappings: 0 % / 0 %
                        w.append(f"orders.revenue differs from the column '{rc}' of {t} on {off:.0%} of the orders (total {tot:+.1%}). Rulebook §1: the "
                                 f"tiền hàng / tiền bán of an order IS its revenue (line amounts after line discounts, before order "
                                 f"discounts, without VAT or shipping): map revenue (and the line amounts of order-only data) to '{rc}'")
                break
    # 1a. revenue against the order's own lines (rulebook §1: revenue of an order = sum of its line amounts after line
    #     discounts; order-level discounts are NOT subtracted, shipping / VAT NOT added). Exact: one order off by 10 000 VND
    #     changes every revenue answer of its month (a public shop's LLM mapping subtracted the order voucher on 34 orders).
    itm = next((c for c in ("item_code", "item_name") if l is not None and c in l.columns), None)   # item lines only (order-only
    if "revenue" in o and itm and len(v) > 50 and {"order_id", "line_amount"} <= set(l.columns):        # data has no lines to add)
        la = pd.to_numeric(l["line_amount"], errors="coerce")
        if la.notna().mean() > 0.9 and l[itm].notna().mean() > 0.9:
            s = la.groupby(l["order_id"].astype(str).str.strip()).sum(min_count=1)
            x = v["order_id"].astype(str).str.strip().map(s); rv = pd.to_numeric(v["revenue"], errors="coerce"); ok = rv.notna() & x.notna()
            if ok.sum() > 50 and ok.mean() > 0.5:
                bad = (rv[ok] - x[ok]).abs() > 1.0
                if bad.any() and not any(str(z).startswith("orders.revenue differs") for z in w):
                    d = (rv[ok] - x[ok])[bad]
                    w.append(f"orders.revenue differs from the sum of the order's line_amount on {int(bad.sum())} valid orders (e.g. order "
                             f"{v.loc[d.index[0], 'order_id']}: revenue {rv[d.index[0]]:.0f}, lines {x[d.index[0]]:.0f}). Rulebook §1: the revenue "
                             "of an order is the sum of its line amounts after line discounts; order-level discounts (voucher, promotion "
                             "on the order total) are NOT subtracted, shipping and VAT are NOT added. Set revenue from the lines")
    # 1a'. order_time without the time of day although the data records one (shift / peak-hour questions answer 0)
    if "order_time" in o and o["order_time"].notna().mean() > 0.5:
        ot = pd.to_datetime(o["order_time"], errors="coerce").dropna()
        if len(ot) > 50 and ((ot.dt.hour == 0) & (ot.dt.minute == 0)).mean() > 0.98:
            src = next((f"{t}.{c}" for t, df in biz.tables.items() for c in df.columns
                        if pd.api.types.is_datetime64_any_dtype(df[c]) and len(df) > 50
                        and (lambda h: len(h) > 50 and ((h.dt.hour != 0) | (h.dt.minute != 0)).mean() > 0.5)(df[c].dropna().head(5000))), None)
            if src:
                w.append(f"orders.order_time has no time of day (every value is 00:00) but {src} records times: order_time is the moment the "
                         "order was created (giờ tạo đơn), used by shift (ca sáng / ca chiều) and peak-hour questions")
    # 1a''. item names: the lines carry the SHORT names of a lookup table (ghi tắt -> tên hàng) whose other column holds the
    #      full names the questions use (a public shop: every item filter missed, accessories ranked as phone lines)
    if l is not None and "item_name" in l.columns and len(l) > 50:
        ln = l["item_name"].dropna().astype(str).map(dm.key); lset = set(ln.unique())
        for t, df in biz.tables.items():
            if not 3 <= len(df) <= 3000: continue
            txt = [c for c in df.columns if not str(c).startswith("_") and (df[c].dtype == object or pd.api.types.is_string_dtype(df[c]))
                   and df[c].notna().mean() > 0.8]
            hit = None
            for a in txt:
                ka = df[a].dropna().astype(str).map(dm.key)
                if ka.nunique() < 3 or not lset & set(ka): continue
                share = float(ln.isin(set(ka)).mean())
                if share < 0.5: continue
                for b in txt:
                    if b == a: continue
                    kb = df[b].dropna().astype(str).map(dm.key)
                    longer = float((df[b].astype(str).str.len() > df[a].astype(str).str.len() + 2).mean())
                    if kb.nunique() >= 0.8 * ka.nunique() and longer > 0.6 and float(ln.isin(set(kb)).mean()) < share / 2:
                        hit = (a, b, share); break
                if hit: break
            if hit:
                w.append(f"lines.item_name holds the short names of {t}.{hit[0]} ({hit[2]:.0%} of the lines) while {t}.{hit[1]} has the full item "
                         f"names that questions use: map item_name to {t}.{hit[1]} through {hit[0]} (with the item group / type of that table)")
                break
            if hit: break
    # 1a-kinds. phone shops: a lookup table gives each item its kind (máy mới / máy cũ / phụ kiện) and the lines must carry it
    #           (public dev_014, v26: the pooled best mapping r1:llm:9 left the kinds out -> accessories ranked as phone models,
    #           0 used phones, 733 'phones' in a month)
    try:
        if l is not None and "item_name" in l.columns and len(l) > 50:
            for t, df in biz.tables.items():
                if not 3 <= len(df) <= 3000: continue
                kc = next((c for c in df.columns if not str(c).startswith("_") and df[c].notna().mean() > 0.8 and 2 <= df[c].nunique() <= 6
                           and df[c].dropna().astype(str).map(industry.device_kind).notna().mean() > 0.9), None)
                if kc is None: continue
                kind = {}
                for c in df.columns:
                    if c == kc or str(c).startswith("_") or not (df[c].dtype == object or pd.api.types.is_string_dtype(df[c])): continue
                    for nm, kd in zip(df[c], df[kc]):
                        if isinstance(nm, str) and nm.strip(): kind.setdefault(dm.key(nm), industry.device_kind(kd))
                exp = l["item_name"].map(lambda n: kind.get(dm.key(n)) if isinstance(n, str) else None)
                m = exp.notna()
                if m.sum() < 50: break
                try: got = dm.Shop(C)._kinds(l); agree = float((got[m] == exp[m]).mean())
                except dm.Unanswerable: agree = 0.0
                if agree < 0.9:
                    w.append(f"the device kinds of C disagree with {t}.{kc} on {1 - agree:.0%} of the lines (máy mới / máy cũ / phụ kiện): set "
                             f"lines.item_group to the kind {t}.{kc} gives each item (looked up by dm.key of the name)")
                break
    except Exception as e: print(f"[canon] kinds check failed: {type(e).__name__}: {e}", flush=True)
    # 1a-cost. catalogue costs all 0 while the raw catalogue records them (public dev_008, v26: stock value 0)
    it_ = C.get("items")
    if isinstance(it_, pd.DataFrame) and len(it_) >= 10:
        for cc in ("unit_cost", "cost"):
            if cc in it_.columns and pd.to_numeric(it_[cc], errors="coerce").fillna(0).eq(0).all():
                src_ = next((f"{t}.{c}" for t, df in biz.tables.items() for c in df.columns
                             if re.fullmatch(r"giavon|gianhap|giavonhang|cost|unitcost", _fold(c)) and _num(df[c]).fillna(0).gt(0).mean() > 0.5), None)
                if src_: w.append(f"items.{cc} is 0 on every item but {src_} records the costs: map items.{cc} from it")
                break
    # 1a-noname. lines without an item name (a name lookup that returned None for the spellings it did not know: public dev_014,
    #           v25: 3 of 12 candidates left 2751 of 10107 lines nameless -> Jan 2026 silicon cases 47 instead of 86)
    if l is not None and "item_name" in l.columns and len(l) > 50:
        nn_ = int(l["item_name"].isna().sum() + l["item_name"].dropna().astype(str).str.strip().isin(["", "None", "nan"]).sum())
        if nn_ >= 20 and nn_ > 0.02 * len(l):
            w.append(f"lines.item_name is empty on {nn_} of {len(l)} lines: every sold line names its item (a lookup of short / "
                     "abbreviated names must fall back to the name as written, matched by dm.key, never to None)")
    # 1a'''. staff written as given names / nicknames ('Đạt', 'a Phúc', 'A.Phúc') while a table lists the full names: answers
    #       must give the name as the data's list writes it, and filters by full name miss the other spellings
    if "staff" in o and o["staff"].notna().mean() > 0.3:
        sv = [x for x in o["staff"].dropna().astype(str).unique() if x.strip()]
        if 2 <= len(sv) <= 300:
            done = False
            for t, df in biz.tables.items():
                if done or not 2 <= len(df) <= 500: continue
                for c in df.columns:
                    if str(c).startswith("_") or not (df[c].dtype == object or pd.api.types.is_string_dtype(df[c])): continue
                    names = [x for x in df[c].dropna().astype(str).unique() if x.strip()]
                    if not 2 <= len(names) <= 200 or sum(len(dm.key(n).split()) >= 2 for n in names) < 0.8 * len(names): continue
                    fk = [dm.key(n).split() for n in names]; fset = {dm.key(n) for n in names}
                    def full_of(x):
                        g = dm.person_key(x).split()
                        hit = [n for n, k in zip(names, fk) if g and k[-len(g):] == g]
                        return hit[0] if len(hit) == 1 else None
                    share = sum(1 for x in sv if full_of(x)) / len(sv); exact = sum(1 for x in sv if dm.key(x) in fset) / len(sv)
                    if share >= 0.8 and exact < 0.5:
                        ex = ", ".join(f"'{x}' -> '{full_of(x)}'" for x in sv[:3] if full_of(x))
                        w.append(f"orders.staff holds given names / nicknames ({ex}) while {t}.{c} lists the full names: map staff to the full "
                                 "name of that list (one person = one name; answers name people as the data's list writes them)")
                        done = True; break
    # 1a-size. the catalogue records sizes / number classes (public rulebook: the size of a diaper, the number of a milk come
    #          from that column, not from the name) but no canonical table carries them
    has_size = any(isinstance(C.get(t), pd.DataFrame) and "size" in C[t].columns for t in ("items", "lines"))
    if not has_size and l is not None and len(l):
        src = next((f"{t}.{c}" for t, df in biz.tables.items() if len(df) <= 5000 for c in df.columns
                    if re.fullmatch(r"(?:so)?size|kichco|kichthuoc|cosize", _fold(c)) and df[c].notna().mean() > 0.2), None)
        if src: w.append(f"the catalogue records sizes in {src} but neither C['items'] nor C['lines'] has a column size: map it (size_qty "
                         "and the rulebook read the size / number from that column, not from the item name)")
    # 1a-credit. orders on credit (ghi nợ) that owe nothing at purchase: paid_at_purchase was set to the order total (public
    #           dev_008: debts only in a free-text note -> every debt answer 0)
    if {"payment_method", "total_payable", "paid_at_purchase"} <= set(o.columns):
        cr = o[o["payment_method"].astype(str).map(lambda x: dm.key(x)) == dm.key("ghi nợ")]
        tp, pp = pd.to_numeric(cr["total_payable"], errors="coerce"), pd.to_numeric(cr["paid_at_purchase"], errors="coerce")
        ok = tp.notna() & pp.notna()
        if ok.sum() >= 20 and float((pp[ok] >= tp[ok] - 1).mean()) > 0.9:
            w.append(f"{int(ok.sum())} orders are on credit (payment_method 'ghi nợ') but {float((pp[ok] >= tp[ok] - 1).mean()):.0%} of them have "
                     "paid_at_purchase = total_payable (nothing owed): paid_at_purchase is what the customer paid WHEN BUYING; read it from "
                     "the data (a paid / owed column, or the order note: money_in_text(note) -> the amounts written in it, e.g. "
                     "'đưa trước 50k, còn thiếu 30 nghìn'); later repayments go to C['payments'] (order_id, pay_date, amount). When the note "
                     "is the debt's state at the END of the data, set orders.debt_end = debt_from_note(note, total_payable) instead")
    # 1a-notes. debts written in free-text notes of the raw data (README: the note is the order's state at the end of the data)
    #           while C records those orders as paid or records no debt at all (public dev_008: v22 runs had no debt information,
    #           a v23 mapping made payments = the order total -> every debt answer 0 / unanswerable)
    try:
        owed_note = None
        for t, df in biz.tables.items():
            ncols = [c for c in df.columns if re.fullmatch(r"(?:note|notes|ghichu|ghichudon|luuy|chuthich|ghichuno)", _fold(c))]
            if not ncols or len(df) < 50: continue
            kcol = _order_key_col(df, ids, exclude=ncols, norm=False)
            if kcol is None: continue
            nn = df[[kcol, ncols[0]]].dropna(); nn = nn[nn[ncols[0]].astype(str).map(dm.key).str.contains(r"\b(?:no|ghi so|thieu|chua tra)\b", regex=True)]
            if len(nn) < 20: continue
            tot = pd.to_numeric(o.set_index(o["order_id"].astype(str).str.strip())["total_payable"], errors="coerce") if "total_payable" in o else None
            tot = tot[~tot.index.duplicated()] if tot is not None else None
            first = nn.groupby(nn[kcol].astype(str).str.strip())[ncols[0]].first()
            owed_note = pd.Series({k: dm.debt_from_note(v, tot.get(k) if tot is not None else None) for k, v in first.items()}).dropna()
            src = f"{t}.{ncols[0]}"; break
        if owed_note is not None and ((owed_note > 0.5).sum() >= 10 or (owed_note <= 0.5).sum() >= 20):
            pos = owed_note[owed_note > 0.5]; settled = owed_note[owed_note <= 0.5]
            if "debt_end" in o and o["debt_end"].notna().any():
                c_owed = pd.to_numeric(o.set_index(o["order_id"].astype(str).str.strip())["debt_end"], errors="coerce")
            elif "total_payable" in o and ("paid_at_purchase" in o or isinstance(C.get("payments"), pd.DataFrame)):
                ot = o.set_index(o["order_id"].astype(str).str.strip())
                tp = pd.to_numeric(ot["total_payable"], errors="coerce")
                if isinstance(C.get("payments"), pd.DataFrame) and len(C["payments"]) and "amount" in C["payments"]:
                    P = C["payments"]; paid = pd.to_numeric(P["amount"], errors="coerce").groupby(P["order_id"].astype(str).str.strip()).sum()
                else: paid = pd.to_numeric(ot["paid_at_purchase"], errors="coerce")
                c_owed = tp - paid.reindex(tp.index).fillna(0)
            else: c_owed = None
            c_owed = c_owed[~c_owed.index.duplicated()] if c_owed is not None else None
            zero = 1.0 if c_owed is None else float((c_owed.reindex(pos.index).fillna(0) <= 0.5).mean()) if len(pos) else 0.0
            if len(settled) >= 20 and c_owed is not None:     # the reverse: notes 'đã trả đủ' while C still counts the order as
                still = float((c_owed.reindex(settled.index).fillna(0) > 0.5).mean())          # owing (public dev_008, a v23 run:
                if still > 0.5:                                                            # paid in full -> owed in full, 19.5 bn)
                    w.append(f"the notes {src} say {len(settled)} orders on credit are settled (e.g. '{first.get(settled.index[0])}') but C "
                             f"records {still:.0%} of them as still owing: the note is the debt at the end of the data, set orders.debt_end "
                             "= debt_from_note(note, total_payable)")
            if len(pos) >= 10 and zero > 0.8:
                w.append(f"the notes {src} say {len(pos)} orders still owe money (e.g. owed {int(pos.iloc[0])} by '{first.get(pos.index[0])}') but "
                         f"C records {zero:.0%} of them as owing nothing: set orders.debt_end = debt_from_note(note, total_payable) (the note "
                         "is the debt at the end of the data) and do not make payments equal the order total")
    except Exception as e: print(f"[canon] note-debt check failed: {type(e).__name__}: {e}", flush=True)
    # 1a-owing. money owed on most orders at the end of the data: orders paid in full recorded as unpaid (public dev_014, v23
    #           runs: C['payments'] held only the debt book's repayments -> 6286 orders owed 22.9 bn, more than a year of sales;
    #           dev_008: a payments table 10k short of every total -> 23349 orders owing)
    try:
        sh = dm.Shop(C); end = sh.last
        if end is not None and "total_payable" in o:
            el = sh._debt_orders(end); dd = sh.debts(end)
            if len(el) >= 200 and len(dd) > OWING_MAX * len(el):
                w.append(f"C counts {len(dd)} of {len(el)} orders ({len(dd) / len(el):.0%}) as still owing money at the end of the data "
                         f"({float(dd.sum()):,.0f} in all): only orders on credit / instalment owe money. Set orders.paid_at_purchase = what was paid "
                         "when buying (= total_payable for an order paid in full; total - the debt left for a debt-book order) and the later "
                         "repayments in C['payments'] (order_id must match orders.order_id)")
    except dm.Unanswerable: pass
    except Exception as e: print(f"[canon] owing check failed: {type(e).__name__}: {e}", flush=True)
    # 1a-status. returns / cancellations written in a text column of the raw data (a note 'khách trả lại', a status 'khách huỷ')
    #           while C keeps those orders with another status (public dev_014, v23: a mapping without the returns -> every
    #           revenue answer counted the returned orders; §1 leaves them out)
    try:
        ids_n = set(_idn(o["order_id"]))
        ost = o.set_index(_idn(o["order_id"]))["status"].astype(str) if "status" in o else None
        ost = ost[~ost.index.duplicated()] if ost is not None else None
        done_kind = set()
        for t, df in (biz.tables.items() if ost is not None else []):
            if len(df) < 50: continue
            kcol = _order_key_col(df, ids_n)
            if kcol is None: continue
            for c in df.columns:
                if c == kcol or str(c).startswith("_") or not (df[c].dtype == object or pd.api.types.is_string_dtype(df[c])): continue
                if not re.search(r"ghichu|note|trangthai|status|tinhtrang|^tt|lydo|xuly", _fold(c)) or re.search(r"khach|ten|name|customer", _fold(c)):
                    continue                                       # note / status columns only (a customer named 'Trần Huy Linh')
                vals = df[c].dropna().astype(str)
                u = pd.unique(vals)
                if not len(u) or len(u) > 20000: continue
                uk = {x: dm.key(x) for x in u}
                for kind, pat, st in (("returned", r"\b(?:tra lai|tra hang|mang tra|hoan tra|hoan hang)\b", "trả hàng"),
                                      ("cancelled", r"\b(?:huy|da huy|khach huy|cancel|cancelled|canceled)\b", "huỷ")):
                    if kind in done_kind: continue
                    hit_vals = {x for x, kx in uk.items() if re.search(pat, kx)}
                    if not hit_vals: continue
                    oids = set(_idn(df.loc[vals.index[vals.isin(hit_vals)], kcol]))
                    m = ost[ost.index.isin(oids)]
                    if len(m) >= 3 and float((m != st).mean()) > 0.8:
                        done_kind.add(kind)
                        w.append(f"{t}.{c} marks {len(m)} orders as {kind} (e.g. '{next(iter(hit_vals))}') but C gives {float((m != st).mean()):.0%} "
                                 f"of them another status: set orders.status = '{st}' for those orders (§1: {kind} orders are not revenue)")
    except Exception as e: print(f"[canon] status-note check failed: {type(e).__name__}: {e}", flush=True)
    # 1a-book. a debt book in the raw data (an owed column + repayment columns) that the toolkit cannot read from C (public
    #          dev_014, v23: a mapping without orders.total_payable -> only the instalments counted as debts, the debt book lost)
    try:
        book = next((t for t, df in biz.tables.items() if len(df) >= 10
                     and any(re.fullmatch(r"no|conno|sono|tienno|notien|sotienno|nocon", _fold(c)) for c in df.columns)
                     and any(re.fullmatch(r"datra|ngaytra|tientra|sotientra|ngaytrano|tientrano", _fold(c)) for c in df.columns)), None)
        if book is not None:
            sh = dm.Shop(C)
            try: bd = sh._book_debts(sh.last if sh.last is not None else pd.Timestamp("2100-01-01"))
            except dm.Unanswerable as e:
                bd = None
                w.append(f"the data has a debt book ({book}) but the toolkit cannot read debts from C ({e}): set orders.total_payable, "
                         "orders.paid_at_purchase (what was paid when buying) and the later repayments in C['payments']")
            if bd is not None:                                  # the book still owes money at the end but C owes nothing
                bk = biz.tables[book]                           # (public dev_014, v28: paid_at_purchase = the whole order + the
                oc = next(c for c in bk.columns if re.fullmatch(r"no|conno|sono|tienno|notien|sotienno|nocon", _fold(c)))   # repayments)
                pc = next(c for c in bk.columns if re.fullmatch(r"datra|tientra|sotientra|tientrano", _fold(c)) or _fold(c) == "datra")
                owed0, paid_ = float(_num(bk[oc]).fillna(0).sum()), float(_num(bk[pc]).fillna(0).sum())
                if owed0 > 0 and owed0 - paid_ > 0.01 * owed0 and float(bd.sum()) <= 0.5:
                    w.append(f"the data has a debt book ({book}) that still owes {owed0 - paid_:,.0f} at the end of the data but C records no debt: "
                             "orders.paid_at_purchase = total - the debt left after paying at purchase (the book's owed column), "
                             "C['payments'] = the later repayments")
    except Exception as e: print(f"[canon] debt-book check failed: {type(e).__name__}: {e}", flush=True)
    # 1c. staff written by given names in the book + a staff list of full names: orders.staff must give each ticket to the person
    #     whose name it writes (public dev_012, v23: a mapping hand-wrote {'An': 'Hà Loan Đạt', 'Đạt': 'Lương Chi Đức', ...})
    try:
        St = C.get("staff")
        if isinstance(St, pd.DataFrame) and "staff" in St.columns and "staff" in o.columns and St["staff"].notna().sum() >= 2:
            full = [x for x in St["staff"].dropna().astype(str)]
            mine = o["staff"].dropna().astype(str).map(industry._person_groups(o["staff"].dropna().astype(str), full)).value_counts()
            for t, df in biz.tables.items():
                if len(df) < 50 or abs(len(df) - len(o)) > 0.05 * len(o): continue
                for c in df.columns:
                    if str(c).startswith("_"): continue
                    v = df[c].dropna().astype(str)
                    if len(v) < 0.5 * len(df) or v.nunique() > 60: continue
                    g = industry._person_groups(v, full); lab = v.map(g)
                    if lab.isin(full).mean() < 0.8: continue
                    # compare (day, person) counts: a swapped name mapping keeps every person's total when the staff work alike
                    od = pd.to_datetime(o["order_date"], errors="coerce").dt.normalize(); days = set(od.dropna())
                    def _isday(x):
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore"); return pd.to_datetime(x, errors="coerce").dt.normalize().isin(days).mean() > 0.9
                    dcol = next((dc for dc in df.columns if dc != c and not str(dc).startswith("_") and not pd.api.types.is_numeric_dtype(df[dc])
                                 and df[dc].nunique() >= 20 and _isday(df[dc])), None)   # ('_sheet' 'T7-2025' parses as a day)
                    if dcol is None: raise StopIteration
                    import collections, warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore"); rd = pd.to_datetime(df[dcol], errors="coerce").dt.normalize()
                    rawc = collections.Counter(zip(rd[v.index], lab))
                    sm = o["staff"].notna(); ol = o.loc[sm, "staff"].astype(str).map(industry._person_groups(o.loc[sm, "staff"].astype(str), full))
                    myc = collections.Counter(zip(od[sm], ol))
                    n = max(1, len(v)); diff = sum(abs(rawc[k] - myc[k]) for k in set(rawc) | set(myc)) / (2 * n)
                    if diff > 0.03:
                        f0 = next((f for f in full if abs(int(lab.eq(f).sum()) - int(mine.get(f, 0))) > 0), full[0])
                        w.append(f"orders.staff does not follow the names written in {t}.{c}: {diff:.0%} of the tickets go to another person than the "
                                 f"name written ({f0}: {int(mine.get(f0, 0))} orders in C, {int(lab.eq(f0).sum())} rows of the book name that person); keep "
                                 "orders.staff as written (the toolkit groups 'Đạt' / 'a Đạt' / 'A.Đạt' with the full name of the staff list) or map each "
                                 "spelling to the full name ENDING with it")
                    raise StopIteration
    except StopIteration: pass
    except Exception as e: print(f"[canon] staff-name check failed: {type(e).__name__}: {e}", flush=True)
    # 1e. order-level columns filled with ONE value for every order (v24 public dev_013 seed 1: orders["order_discount"] = the
    #     voucher total of the whole table, orders["channel"] = the first order's channel): every answer on them is wrong
    if len(o) > 50:
        rev_sum = float(pd.to_numeric(o["revenue"], errors="coerce").fillna(0).sum()) if "revenue" in o else 0.0
        for col in ("order_discount", "shipping_fee", "points_discount", "tradein"):
            if col not in o: continue
            x = pd.to_numeric(o[col], errors="coerce").fillna(0)
            if x.nunique() == 1 and float(x.iloc[0]) != 0:
                w.append(f"orders.{col} is {float(x.iloc[0]):.0f} on every order: a total over the whole table was assigned to each order; "
                         f"{col} is the amount OF THAT ORDER (group the lines by order_id, or take the order's own cell)")
            elif rev_sum > 0 and float(x.abs().sum()) > 0.5 * rev_sum:
                w.append(f"orders.{col} adds up to {float(x.abs().sum()) / rev_sum:.0%} of the revenue: it must hold only that order's {col} amount")
        for col, pat in (("channel", r"kenh\w*|nguondon|nguonban|channel"), ("payment_method", r"(?:phuongthuc|hinhthuc)(?:tt|thanhtoan)?|thanhtoanbang|loaitt|payment\w*")):
            if col not in o or o[col].dropna().astype(str).nunique() != 1: continue
            src_ = next((f"{t}.{c}" for t, df in biz.tables.items() if len(df) >= 50 for c in df.columns
                         if re.fullmatch(pat, _fold(c)) and df[c].dropna().astype(str).str.strip().nunique() >= 2), None)
            if src_: w.append(f"orders.{col} has the single value {o[col].dropna().iloc[0]!r} on every order but {src_} records several: map "
                              "each order's own value")
    # 1d. books of the public rulebook's industries present in the raw data but missing from C (v24 public dev_014 run 2: the chosen
    #     mapping had no instalment book, every instalment answer 0): the toolkit can only use what C holds
    have = {t for t, v in C.items() if isinstance(v, pd.DataFrame) and len(v)}
    for t, df in biz.tables.items():
        if len(df) < 10: continue
        cf = {_fold(c) for c in df.columns}
        def anyc(*pats): return sum(1 for c in cf for p_ in pats if re.fullmatch(p_, c))
        if anyc(r"soky|sokygop|kygop|sothang", r"gopthang|gop1thang|tienmoiky|tiengop|gopmoithang", r"tratruoc", r"hangop|hangopngay|ngaygop") >= 2 \
                and "instalments" not in have:
            w.append(f"the data has an instalment book ({t}: {', '.join(c for c in df.columns if not str(c).startswith('_'))[:120]}) but C has no "
                     "C['instalments'] / C['instalment_payments']: map it (the toolkit counts unpaid instalments as debts)")
        elif anyc(r"soky|sokygop|kygop|sothang", r"gopthang|gop1thang|tienmoiky|tiengop|gopmoithang", r"tratruoc", r"hangop|hangopngay|ngaygop") >= 2 \
                and anyc(r"dagop|tiendagop|ngaygop|sotiengop|datra") >= 1 and "instalment_payments" not in have:
            w.append(f"the data has an instalment book ({t}) that records the instalments paid, but C has no C['instalment_payments'] "
                     "(order_id, pay_date, amount): without it every instalment counts as unpaid")
        if anyc(r"ngaynhan|ngaybaohanh|ngaymangtoi", r"imei|loi|tinhtrangloi|xuly") >= 2 and anyc(r"imei|loi") >= 1 and "warranty" not in have \
                and not ("warranty" in o.columns and o["warranty"].notna().any()):
            w.append(f"the data has a warranty book ({t}) but C has no C['warranty'] (date received, item_name, imei): map it")
        if anyc(r"giathu|giathumay|tienthumay|thumaycu|trudonstt|truvao.*") >= 1 and "tradeins" not in have \
                and not ("tradein" in o.columns and pd.to_numeric(o["tradein"], errors="coerce").fillna(0).gt(0).any()):
            w.append(f"the data records trade-ins ({t}) but C has neither C['tradeins'] nor orders.tradein: map them")
        if anyc(r"hoahong.*|tilehoahong") >= 1 and not (isinstance(C.get("staff"), pd.DataFrame) and "commission_pct" in C["staff"].columns):
            w.append(f"the data has commission rates ({t}) but C['staff'] has no commission_pct: map the staff list (staff, commission_pct)")
        if anyc(r"ngaydangky|ngaydk|ngaythamgia|ngaylamthe") >= 1 and not (isinstance(C.get("customers"), pd.DataFrame) and "registered" in C["customers"].columns):
            w.append(f"the data has registration dates ({t}) but C['customers'] has no registered column: map the member list")
        # number plates (public rulebook 'Sửa xe': vehicles are counted by plate; public dev_012, v23: the pooled best mapping had
        # no orders.plate / orders.vehicle -> every 'bao nhiêu chiếc xe' answer unanswerable)
        if len(df) >= 50:
            for c in df.columns:
                if str(c).startswith("_"): continue
                s = df[c].dropna().astype(str).str.strip().head(400)
                if len(s) < 30: continue
                sp = [industry.split_vehicle(x) for x in s]
                if sum(1 for _, pl in sp if pl) < 0.6 * len(s): continue
                if "plate" in o.columns and o["plate"].notna().mean() > 0.3:   # plates mapped: they must be the book's plates
                    import collections as _co                                    # (public dev_012, v27: candidates with plates
                    raw_ = _co.Counter(industry.plate_key(pl) for _, pl in            # gave 2 / 8 / 25 / 33 / 70 repeat vehicles)
                                       (industry.split_vehicle(x) for x in df[c].dropna().astype(str)) if pl)
                    got_ = _co.Counter(industry.plate_key(x) for x in o["plate"].dropna().astype(str))
                    ov = sum((raw_ & got_).values()) / max(1, sum(raw_.values()))
                    if ov < 0.9:
                        w.append(f"orders.plate matches the plates of {t}.{c} on only {ov:.0%} of the tickets: set orders.vehicle, orders.plate = "
                                 "split_vehicle(cell) for every row (one ticket per row)")
                if not ("plate" in o.columns and o["plate"].notna().mean() > 0.3):
                    w.append(f"the data records number plates in {t}.{c} (e.g. '{s.iloc[0]}') but orders.plate is empty: set orders.vehicle, "
                             "orders.plate = split_vehicle(cell) (vehicles are counted by plate)")
                elif sum(1 for ln, _ in sp if ln) >= 0.6 * len(s) and not ("vehicle" in o.columns and o["vehicle"].notna().mean() > 0.3):
                    w.append(f"the data writes the vehicle model with its plate in {t}.{c} (e.g. '{s.iloc[0]}') but orders.vehicle is empty: "
                             "set orders.vehicle, orders.plate = split_vehicle(cell)")
                break
    # 1b. the toolkit's columns of each secondary table; debts need the order total and an amount paid at purchase
    for t, cols in REQUIRED.items():
        df = C.get(t)
        if isinstance(df, pd.DataFrame) and len(df):
            miss = [c for c in cols if c not in df.columns]
            if miss: w.append(f"C['{t}'] lacks {miss}: the toolkit reads C['{t}'] with the columns {cols} (see the canonical tables)")
    if "paid_at_purchase" in o and (pd.api.types.is_datetime64_any_dtype(o["paid_at_purchase"]) or
                                    pd.to_numeric(o["paid_at_purchase"], errors="coerce").notna().mean() < 0.5 * o["paid_at_purchase"].notna().mean()):
        w.append("orders.paid_at_purchase must be the AMOUNT the customer paid when buying (a number), not a time or a label")
    P_ = C.get("payments")
    if isinstance(P_, pd.DataFrame) and len(P_) and "total_payable" not in o:
        w.append("C['payments'] is mapped but orders has no total_payable (what the customer must pay for the whole order): debts are "
                 "total_payable minus the payments")
    # 1c. item groups reach the lines; returned orders are 'trả hàng'; debts are the exception, not the rule
    it = C.get("items")
    if isinstance(it, pd.DataFrame) and "item_group" in it and it["item_group"].notna().any() and l is not None and len(l):
        fill = float(l["item_group"].notna().mean()) if "item_group" in l else 0.0
        if fill < 0.5:
            w.append(f"items has item_group but only {fill:.0%} of the lines have one: merge item_group into the lines (finalize(..., "
                     "items=...)), otherwise every question about a group of items answers 0")
    rt = C.get("returns")
    if isinstance(rt, pd.DataFrame) and len(rt) > 10 and "order_id" in rt and "status" in o:
        share = float(rt["order_id"].astype(str).map(o.set_index(o["order_id"].astype(str))["status"]).eq("trả hàng").mean())
        if share < 0.9:
            w.append(f"only {share:.0%} of the orders in C['returns'] have status 'trả hàng': an order with any returned item is a "
                     "returned order (rulebook §1) and does not count for revenue")
    if isinstance(C.get("payments"), pd.DataFrame) and len(C["payments"]) and "total_payable" in o and "order_date" in o:
        try:
            sh = dm.Shop(C, None, {"data_end": data_end(biz)}); end = sh.data_end(); base = sh._debt_orders(end)
            owe = len(sh.debts(end)) / max(len(base), 1)
            if len(base) > 100 and owe > 0.05:                         # the reference mappings: at most 0.4 %
                w.append(f"{owe:.0%} of the orders still owe money at the end of the data: check that payments.order_id matches "
                         "orders.order_id (same codes, as text) and that total_payable is what the customer must pay for the order")
        except Exception: pass
    # 2. lots without the allocation of sales to lots: every lot then counts as still full
    lots, ls = C.get("lots"), C.get("lot_sales")
    if isinstance(lots, pd.DataFrame) and len(lots) and l is not None and len(l) and not (isinstance(ls, pd.DataFrame) and len(ls)):
        w.append("C['lots'] is mapped without C['lot_sales'] (order_id, lot_id, qty_base, returned_base): without it every lot "
                 "counts as full in stock (lot_stock, lots_expiring, nearest_expiry). Map the table that records which lot each "
                 "sold quantity came from, if the data has one")
    # 3. purchases are item lines; a stock snapshot has one row per item; conversions are not empty
    p = C.get("purchases")
    if isinstance(p, pd.DataFrame) and len(p) and not any(c in p.columns and p[c].notna().any() for c in ("item_code", "item_name")):
        w.append("C['purchases'] has no item_code / item_name: purchases must be one row per received item line (receipt_id, "
                 "receipt_date, supplier, item_code, unit, qty, unit_cost, amount); a table of receipt headers has no items: map "
                 "the receipt lines table (with the header's date and supplier) instead")
    st = C.get("stock")
    if isinstance(st, pd.DataFrame) and len(st) > 50:
        k = next((c for c in ("item_code", "item_name") if c in st.columns and st[c].notna().any()), None)
        if k is not None and len(st) > 1.5 * max(st[k].nunique(), 1):
            w.append(f"C['stock'] has {len(st) / max(st[k].nunique(), 1):.1f} rows per item, but a stock snapshot has one row per item "
                     "(qty_base at the end of the data). A table of stock movements with the stock after each movement is "
                     "C['ledger'] (item_code, time, qty_base, stock_after)")
    cv = C.get("conversions")
    if isinstance(cv, pd.DataFrame) and not len(cv):
        w.append("C['conversions'] is empty: map the table of unit conversions (item, unit, factor), or leave conversions out")
    # 3b. the data records customer groups but C does not (public dev_008: one of three runs' mappings left them out, another
    #     took each customer's last group instead of the order's): questions by customer group then answer 'unanswerable'
    if "customer_group" not in o or not o["customer_group"].notna().any():
        for t, df in biz.tables.items():
            if len(df) < 50: continue
            for c in df.columns:
                if not re.fullmatch(r"(?:nhomkhach(?:hang)?|loaikhach(?:hang)?|phanloaikhach(?:hang)?|hangkhach(?:hang)?|hangthanhvien|nhomkh|customergroup)", _fold(c)):
                    continue
                vals = df[c].dropna().astype(str).str.strip(); vals = vals[vals.ne("")]
                if 2 <= vals.nunique() <= 20 and len(vals) >= 20:
                    w.append(f"orders.customer_group is empty although {t}.{c} records customer groups ({', '.join(vals.value_counts().index[:4])}): "
                             "map it (the order's own group; rulebook: walk-in orders belong to no group)")
                    break
    # 4. purchases and lots dated inside the data
    end = data_end(biz)
    if end is not None and isinstance(p, pd.DataFrame) and len(p) > 20 and "receipt_date" in p:
        d = pd.to_datetime(p["receipt_date"], errors="coerce").dropna()
        if len(d) and ((d > end + pd.Timedelta(days=2)).mean() > 0.2 or (d < end - pd.Timedelta(days=5 * 366)).mean() > 0.2):
            w.append(f"purchase receipt dates run from {d.min().date()} to {d.max().date()}, the data ends {end.date()}: check the receipt "
                     "date column and its conversion")
    if isinstance(lots, pd.DataFrame) and len(lots) > 20 and "expiry" in lots:
        ex = pd.to_datetime(lots["expiry"], errors="coerce")
        if ex.notna().mean() < 0.8:
            w.append(f"only {ex.notna().mean():.0%} of the lots have an expiry date that parses: check the expiry column and its format")
        elif "receipt_date" in lots:
            rd = pd.to_datetime(lots["receipt_date"], errors="coerce"); ok = ex.notna() & rd.notna()
            if ok.sum() > 20 and (ex[ok] < rd[ok]).mean() > 0.2:
                w.append("more than 20 % of the lots expire before they were received: expiry and receipt_date look swapped or wrong")
    return w


ALIASES = {   # canonical column <- names an LLM mapping uses for it (renamed only when the canonical one is absent)
    "deliveries": {"trip_no": ["trip", "trip_id", "trip_number", "batch", "delivery_batch", "delivery_no", "delivery_trip", "chuyen"],
                   "delivery_date": ["date", "ship_date", "delivered_date"], "qty_base": ["qty_delivered", "delivered_qty"]},
    "payments": {"pay_date": ["payment_date", "paid_date", "date"], "amount": ["paid_amount", "pay_amount", "paid"]},
    "lots": {"expiry": ["expiry_date", "exp_date", "expiration_date", "expire_date"], "qty_in_base": ["qty_base", "qty_in", "received_base"]},
    "lot_sales": {"qty_base": ["qty_sold_base", "sold_base"], "returned_base": ["qty_returned_base", "returned_qty_base"]},
    "ledger": {"stock_after": ["balance", "stock_balance", "balance_after", "qty_after"], "time": ["datetime", "timestamp"]},
    "prices": {"price": ["new_price", "unit_price"], "date": ["effective_date", "valid_from", "apply_date"]},
    "expenses": {"date": ["expense_date"], "amount": ["expense_amount", "cost"]},
    # industry tables (src/industry.py); a staff list copied raw keeps its Vietnamese headers (public dev_012, v23)
    "staff": {"staff": ["full_name", "name", "staff_name", "Họ tên", "Họ và tên", "Tên thợ", "Tên nhân viên", "Họ tên thợ", "Tên"],
              "commission_pct": ["commission", "commission_rate", "commission_percent", "Hoa hồng (%)", "Hoa hồng", "% hoa hồng", "Tỉ lệ hoa hồng"],
              "start_date": ["Ngày vào làm", "join_date", "hire_date"]},
    "customers": {"registered": ["registration_date", "register_date", "reg_date", "joined", "Ngày đăng ký"],
                  "customer_name": ["name", "Tên KH", "Tên khách hàng"]},
    "warranty": {"date": ["received", "received_date", "date_received", "Ngày nhận"], "imei": ["IMEI"]},
    "instalments": {"buy_date": ["date", "purchase_date", "order_date", "Ngày", "Ngày mua"], "n_terms": ["terms", "n_periods", "num_terms", "Số kỳ", "So ky"],
                    "per_term": ["monthly", "per_month", "installment", "monthly_payment", "Góp/tháng", "Góp tháng", "Tiền góp"],
                    "due_day": ["Hạn góp (ngày)", "Hạn góp", "due", "pay_day"], "price": ["Giá", "total_price", "Tổng tiền"],
                    "prepaid": ["Trả trước", "down_payment", "deposit"], "customer_name": ["Khách", "customer", "Tên khách"],
                    "item_name": ["Máy", "device", "product"], "order_id": ["STT", "order", "sales_order_id"]},
    "instalment_payments": {"pay_date": ["date", "payment_date", "paid_date", "Ngày góp"], "amount": ["paid", "paid_amount", "Đã góp"],
                            "order_id": ["STT", "order", "contract_id"]},
    "tradeins": {"amount": ["price", "value", "tradein_value", "Giá thu"], "date": ["tradein_date", "Ngày"], "order_id": ["Trừ vào (STT sổ bán)", "STT"]},
}
REQUIRED = {"deliveries": ["order_id", "delivery_date", "trip_no"], "payments": ["order_id", "pay_date", "amount"],
            "lots": ["lot_id", "item_code", "expiry", "qty_in_base"], "lot_sales": ["order_id", "lot_id", "qty_base"],
            "ledger": ["item_code", "time", "stock_after"], "conversions": ["item", "unit", "factor"], "prices": ["date", "price"],
            "staff": ["staff"], "instalments": ["order_id", "buy_date", "n_terms"], "instalment_payments": ["order_id", "pay_date", "amount"],
            "warranty": ["date", "item_name"], "tradeins": ["order_id", "amount"], "customers": ["registered"]}


def normalize_aliases(C):
    """rename alias columns of the toolkit's tables to their canonical names (an LLM wrote 'delivery_batch' for trip_no:
    every delivery question of that business failed); payments get pay_date from pay_time. In place; returns renames."""
    done = []
    for t, al in ALIASES.items():
        df = C.get(t)
        if not isinstance(df, pd.DataFrame): continue
        for canon_col, names in al.items():
            if canon_col in df.columns: continue
            src = next((n for n in names if n in df.columns), None)
            if src is None:                                            # a raw table kept with its Vietnamese headers ('Số kỳ')
                fn = {_fold(n) for n in names}; src = next((c for c in df.columns if _fold(c) in fn), None)
            if src is not None: df.rename(columns={src: canon_col}, inplace=True); done.append(f"{t}.{src}->{canon_col}")
    P = C.get("payments")
    if isinstance(P, pd.DataFrame) and "pay_date" not in P.columns and "pay_time" in P.columns:
        P["pay_date"] = pd.to_datetime(P["pay_time"], errors="coerce").dt.normalize(); done.append("payments.pay_time->pay_date")
    return done


def skipped_warnings(biz):
    """columns an LLM mapping lost without an error: frame() skips a raw name the table does not have (biz_123 in v6:
    purchases built from chi_tiet_nhap with NgayNhapKho / MaNhaCungCap, columns of phieu_nhap -> no receipt date, no
    supplier: every purchase question abstained). Reference transfers skip absent columns by design: not checked."""
    w, seen = [], set()
    for col, raw in dm.frame.skipped:
        if (col, raw) in seen or len(w) >= 8: continue
        seen.add((col, raw)); where = [t for t, df in biz.tables.items() if raw in df.columns]
        w.append(f"frame() skipped {col}={raw!r}: the table given to frame() has no column {raw!r}, so {col} is missing"
                 + (f"; {raw!r} is a column of {', '.join(where[:3])}: merge that table in first (or map {col} from it)" if where else ""))
    return w


def resolve_short_names(C, biz):
    """cleaning layer (rulebook §8: names that denote one thing are one thing): when the raw data has a lookup table of short
    names -> full item names (ghi tắt -> tên hàng) and the mapping already put the full names on most lines, the lines left
    with a spelling of a short name that differs only in case / accents ('ỐP SILICON IP' for 'Ốp silicon IP') get its full
    name too (public dev_014, v25: 19 such lines in Jan 2026 -> silicon cases 67 instead of 86). Returns the number of lines
    renamed; nothing happens when no lookup table is found or when most lines still hold short names (a mapping that never
    looked the names up: the secondary check asks for the lookup)."""
    l = C.get("lines")
    if not isinstance(l, pd.DataFrame) or "item_name" not in l.columns or len(l) <= 50: return 0
    ln = l["item_name"].map(lambda x: dm.key(x) if isinstance(x, str) else None)
    for t, df in biz.tables.items():
        if not 3 <= len(df) <= 3000: continue
        txt = [c for c in df.columns if not str(c).startswith("_") and (df[c].dtype == object or pd.api.types.is_string_dtype(df[c]))
               and df[c].notna().mean() > 0.8]
        for a in txt:
            for b in txt:
                if b == a: continue
                pair = df[[a, b]].dropna().astype(str)
                if len(pair) < 3: continue
                ka, kb = pair[a].map(dm.key), pair[b].map(dm.key)
                longer = float((pair[b].str.len() > pair[a].str.len() + 2).mean())
                if kb.nunique() < 0.8 * ka.nunique() or longer <= 0.6 or float(ln.isin(set(kb)).mean()) < 0.3: continue
                full = {}
                for x, y in zip(ka, pair[b]):
                    if x not in set(kb): full.setdefault(x, y)
                left = ln.isin(set(full))
                if not left.any(): return 0
                C["lines"] = l = l.copy()
                l.loc[left, "item_name"] = ln[left].map(full)
                return int(left.sum())
    return 0


def run_mapping(code, biz, ref_cols=None, llm=False):
    """execute one mapping program -> dict(ok, C, notes, errors, coverage, tables_used, unused, warnings). llm: the
    columns frame() skipped are warnings (an LLM program names the columns of this business)."""
    g = namespace(biz, ref_cols)
    dm._TOOLKIT_NS.clear(); dm._TOOLKIT_NS.update(g)                  # `from builder import frame` -> this namespace
    dm.frame.skipped = []
    try: exec(compile(code, "<mapping>", "exec"), g)
    except Exception as e:
        tb = [l for l in traceback.format_exc().splitlines() if '"<mapping>"' in l][-1:]
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}" + (f" ({tb[0].strip()})" if tb else "") + key_hint(e, biz) + skip_hint(e, biz)}
    C = g.get("C")
    if not isinstance(C, dict) or "orders" not in C: return {"ok": False, "error": "the program did not build C with C['orders']"}
    normalize_aliases(C)
    try: resolve_short_names(C, biz)
    except Exception as e: print(f"[canon] short-name resolution failed: {type(e).__name__}: {e}", flush=True)
    errs = dm.check(C)
    if errs: return {"ok": False, "error": "canonical tables failed the checks: " + "; ".join(errs), "C": C}
    errs = plausibility(C, biz)                                        # content checks: silent mistakes -> repairable errors
    if errs: return {"ok": False, "error": "the canonical tables look wrong:\n- " + "\n- ".join(errs), "C": C}
    unused = [t for t in biz.tables if t not in g["T"].accessed and t not in g["RAW"].accessed]
    try: warns = secondary_checks(C, biz)
    except Exception as e: warns = []; print(f"[canon] secondary checks failed: {type(e).__name__}: {e}", flush=True)
    if llm: warns += skipped_warnings(biz)
    return {"ok": True, "C": C, "notes": str(g.get("NOTES", "")), "coverage": coverage(C), "tables_used": dict(g["T"].used), "unused": unused,
            "warnings": warns}


def resolvable(biz, ref_meta):
    """cheap pre-check: every table a reference program reads has a counterpart (>= 70 % of its columns) in biz."""
    if ref_meta == "generic": return True                               # format adapters find their tables themselves
    tabs = {t: c for t, c in ((ref_meta or {}).get("tables") or {}).items() if t not in set((ref_meta or {}).get("optional", []))}
    if not tabs: return False
    have = [set(df.columns) for df in biz.tables.values()]
    return all(max((len(set(cols) & h) / max(1, len(cols)) for h in have), default=0) >= 0.7 for cols in tabs.values())


def transfer(biz, meta=None, progs=None):
    """run every reference program on biz; -> list of (coverage, ref_id, result) for the programs that pass, best first."""
    meta = meta if meta is not None else load_meta(); progs = progs or ref_programs(); out = []
    for rid, code in progs.items():
        if not resolvable(biz, "generic" if rid.startswith("fmt:") else meta.get(rid)): continue
        r = run_mapping(code, biz, meta.get(rid, {}).get("tables"))
        if r["ok"]: out.append((r["coverage"], rid, r))
    return sorted(out, key=lambda x: -x[0])


def similarity(biz, ref_meta):
    """column overlap between a business and a reference format (Jaccard of column-name sets)."""
    a = {c for df in biz.tables.values() for c in df.columns if c not in ("_source", "_sheet")}
    b = {c for cols in ref_meta.get("tables", {}).values() for c in cols}
    return len(a & b) / max(1, len(a | b))


def nearest_refs(biz, k=2, meta=None, exclude=()):
    meta = meta if meta is not None else load_meta()
    sc = sorted(((similarity(biz, m), rid) for rid, m in meta.items() if rid not in exclude), reverse=True)
    return [rid for _, rid in sc[:k]]


def card(C, notes=""):
    """compact description of canonical tables for the answer prompt."""
    out = []
    for name, df in C.items():
        if not isinstance(df, pd.DataFrame): continue
        out.append(ingest.table_card(name, df.drop(columns=[c for c in df.columns if c.endswith("_raw")]), max_examples=4))
    return "\n".join(out) + (f"\nNOTES: {notes}" if notes else "")


def main():
    a = sys.argv[1:]; split, cache = a[0], a[1]
    meta = build_meta(cache) if "--build" in a else load_meta()
    bids = [x for x in a[2:] if not x.startswith("--")] or sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(cache, "*.pkl")))
    for bid in bids:
        biz = pickle.load(open(os.path.join(cache, bid + ".pkl"), "rb"))
        progs = {k: v for k, v in ref_programs().items() if k != bid}            # leave-one-out: never its own program
        res = transfer(biz, meta, progs)
        print(f"{bid}: {len(res)} reference programs fit" + (f"; best {res[0][1]} coverage {res[0][0]:.0f} tables {res[0][2]['tables_used']}" if res else
              f"; nearest formats {nearest_refs(biz, 3, meta, exclude=(bid,))}"), flush=True)


if __name__ == "__main__":
    main()
