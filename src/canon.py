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
import os, re, sys, json, glob, pickle, traceback
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dm, ingest

REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refs", "compile")
META = os.path.join(REF_DIR, "_meta.json")


FMT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refs", "formats")


def ref_programs(formats=True):
    """reference mapping programs: one per train business (refs/compile) + generic format adapters (refs/formats, written
    with tbl() so they apply to any export of that layout)."""
    out = {os.path.basename(f)[:-3]: open(f, encoding="utf-8").read() for f in sorted(glob.glob(os.path.join(REF_DIR, "*.py")))}
    if formats: out.update({"fmt:" + os.path.basename(f)[:-3]: open(f, encoding="utf-8").read() for f in sorted(glob.glob(os.path.join(FMT_DIR, "*.py")))})
    return out


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
        super().__init__(data); self.ref_cols = ref_cols; self.used = {}

    def __missing__(self, name):
        cols = set(self.ref_cols.get(name, []))
        if not cols: raise KeyError(name)
        best, score = None, 0.0
        for t, df in self.items():
            s = len(cols & set(df.columns)) / len(cols)
            if s > score: best, score = t, s
        if best is None or score < 0.7: raise KeyError(f"{name} (no table with its columns)")
        self.used[name] = best
        return dict.__getitem__(self, best)


def data_end(biz):
    m = re.search(r"có đến hết (\d{1,2})/(\d{1,2})/(\d{4})", biz.readme or "")
    return pd.Timestamp(int(m.group(3)), int(m.group(2)), int(m.group(1))) if m else None


def namespace(biz, ref_cols=None):
    T = Tables({k: v.copy() for k, v in biz.tables.items()}, ref_cols or {})
    R = Tables(dict(biz.raw), ref_cols or {})
    g = {"T": T, "RAW": R, "README": biz.readme or "", "pd": pd, "np": np, "re": re, "parse_num": ingest.parse_num, "parse_date": ingest.parse_date}
    for k in ("finalize", "map_status", "canon_status", "canon_payment", "canon_channel", "key", "is_walkin", "money", "round_half_up",
              "readme_conversions", "conversions_table", "frame"):
        g[k] = getattr(dm, k)
    g["tbl"] = lambda *c, **kw: dm.find_table(T, *c, **kw)
    return g


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


def run_mapping(code, biz, ref_cols=None):
    """execute one mapping program -> dict(ok, C, notes, errors, coverage, tables_used)."""
    g = namespace(biz, ref_cols)
    try: exec(compile(code, "<mapping>", "exec"), g)
    except Exception as e:
        tb = [l for l in traceback.format_exc().splitlines() if '"<mapping>"' in l][-1:]
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}" + (f" ({tb[0].strip()})" if tb else "")}
    C = g.get("C")
    if not isinstance(C, dict) or "orders" not in C: return {"ok": False, "error": "the program did not build C with C['orders']"}
    errs = dm.check(C)
    if errs: return {"ok": False, "error": "canonical tables failed the checks: " + "; ".join(errs), "C": C}
    return {"ok": True, "C": C, "notes": str(g.get("NOTES", "")), "coverage": coverage(C), "tables_used": dict(g["T"].used)}


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
