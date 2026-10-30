#!/usr/bin/env python3
"""Tested building blocks for the rulebook's sales semantics, exposed to LLM programs (agent.py namespace).

build_sales(...) turns the shop's sales table(s) into canonical `orders` and `lines` given only a column mapping, so the
model does not have to re-derive (and often get wrong) the generic recording rules of the rulebook in every program:
  - blank order-level cells on the following lines of an order inherit the order's first line (rulebook §8);
  - two lines of the SAME order identical in every cell are one export error, kept once; identical lines of different
    orders, or of a table without order ids, are separate sales (§8);
  - each order has exactly one status: any returned line -> "trả hàng" (the whole order, §1), else any cancelled line ->
    "huỷ", else its first status; valid = not cancelled and not returned;
  - revenue of an order = sum of its line amounts after line discounts (order-level discounts, VAT, delivery fees are
    not deducted / added: they are never taken from the line amounts).
Nothing here knows any business: the mapping comes from the model, unmapped status labels raise an error that lists them.
"""
import numpy as np
import pandas as pd

CANON = ("hoàn thành", "huỷ", "trả hàng", "đang giao")
WALKIN = ("khách lẻ", "khach le", "lẻ", "le", "kl", "khách vãng lai", "vãng lai", "khách lạ", "khach vang lai", "walk-in", "walk in",
          "guest", "khách", "-", "none", "nan", "null")


def ingest_norm(x):
    import unicodedata, re
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", unicodedata.normalize("NFC", str(x)).lower())).strip()
_ALIASES = {"hủy": "huỷ", "huy": "huỷ", "tra hang": "trả hàng", "hoan thanh": "hoàn thành", "dang giao": "đang giao"}


def _canon_label(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return None
    s = str(v).strip().lower()
    return s if s in CANON else _ALIASES.get(s)


def build_sales(lines, item, qty=None, amount=None, date=None, order_id=None, status=None, status_map=None,
                qty_base=None, factor=None, unit=None, unit_price=None, line_discount=None, item_code=None,
                customer=None, customer_id=None, time=None, header=None, header_key=None, keep=(),
                conv=None, conv_item=None, conv_unit=None, conv_factor=None, walkin=None):
    """lines: DataFrame of sale lines (one row per item sold). Column arguments are column names of `lines`, or of
    `header` (a DataFrame with one row per order, joined on header_key == order_id) for order-level fields.
    status_map: {raw status label: one of "hoàn thành", "huỷ", "trả hàng", "đang giao"} covering every raw label
    (labels already canonical may be omitted). factor: column name, or dict {unit: factor} / {(item, unit): factor},
    used only when qty_base is not given; or conv = conversion table with columns conv_item, conv_unit, conv_factor
    (unit -> base units of that item; units without a row have factor 1). keep: other columns to carry into the result.
    walkin: extra customer labels that mean a walk-in customer (khách lẻ / vãng lai / blank are always walk-in): their
    customer_name / customer_id become None (rulebook: walk-in orders are not a customer).
    Returns (orders, lines): canonical DataFrames (see agent.ENV_DOC)."""
    L = lines.copy()
    H = header.copy() if header is not None else None
    hkey = header_key or order_id

    def where(c):
        if c is None: return None
        if c in L.columns: return "L"
        if H is not None and c in H.columns: return "H"
        raise KeyError(f"build_sales: column {c!r} is in neither the lines table {list(L.columns)[:30]} nor the header table")

    for c in (item, qty, amount, qty_base, unit, unit_price, line_discount, item_code):
        if c is not None and where(c) != "L": raise KeyError(f"build_sales: line column {c!r} must be in the lines table")
    if order_id is not None and order_id not in L.columns: raise KeyError(f"build_sales: order_id {order_id!r} not in the lines table")
    data_cols = [c for c in L.columns if not str(c).startswith("_")]
    if order_id is not None:
        L[order_id] = L[order_id].ffill()
        # §8: blank order-level cells of following lines inherit the order's first line
        lvl = [c for c in (date, time, status, customer, customer_id) + tuple(keep) if c is not None and where(c) == "L"]
        for c in lvl:
            if L[c].isna().any(): L[c] = L.groupby(order_id, sort=False)[c].transform("first").where(L[c].isna(), L[c])
        L = L[~L.duplicated(subset=data_cols, keep="first")]               # identical lines of the same order: once
    else:
        L["__row"] = np.arange(len(L)); order_id = "__row"
    if H is not None:
        if hkey not in H.columns: raise KeyError(f"build_sales: header_key {hkey!r} not in the header table")
        hcols = [c for c in H.columns if c != hkey and not str(c).startswith("_") and c not in L.columns]
        L = L.merge(H[[hkey] + hcols].drop_duplicates(hkey), left_on=order_id, right_on=hkey, how="left",
                    suffixes=("", "__h"))
    out = pd.DataFrame(index=L.index)
    out["order_id"] = L[order_id].astype(str).str.strip() if order_id != "__row" else L[order_id]
    if date is not None:
        d = pd.to_datetime(L[date], errors="coerce")
        if time is not None:
            tt = L[time]
            if pd.api.types.is_datetime64_any_dtype(tt): d = d.dt.normalize() + (tt - tt.dt.normalize())
            else: d = d.dt.normalize() + pd.to_timedelta(tt.astype(str).str.extract(r"(\d{1,2}:\d{2}(?::\d{2})?)")[0].fillna("00:00:00").map(lambda x: x if x.count(":") == 2 else x + ":00"), errors="coerce").fillna(pd.Timedelta(0))
        out["order_time"] = d; out["order_date"] = d.dt.normalize()
    out["item_name"] = L[item].astype(str).str.strip()
    if item_code is not None: out["item_code"] = L[item_code].astype(str).str.strip()
    if unit is not None: out["unit"] = L[unit]
    q = pd.to_numeric(L[qty], errors="coerce") if qty is not None else pd.Series(1.0, index=L.index)
    out["qty"] = q
    if qty_base is not None: out["qty_base"] = pd.to_numeric(L[qty_base], errors="coerce")
    else:
        f = pd.Series(1.0, index=L.index)
        if conv is not None:
            if not (conv_item and conv_unit and conv_factor): raise ValueError("build_sales: conv needs conv_item, conv_unit, conv_factor column names")
            if unit is None: raise ValueError("build_sales: conv needs unit= (the unit column of the lines)")
            k = lambda x: ingest_norm(x)
            fm = {(k(a), k(b)): float(c) for a, b, c in zip(conv[conv_item], conv[conv_unit], pd.to_numeric(conv[conv_factor], errors="coerce")) if pd.notna(c)}
            f = pd.Series([fm.get((k(a), k(b)), 1.0) for a, b in zip(L[item], L[unit])], index=L.index)
        elif isinstance(factor, (pd.Series, pd.DataFrame)):
            raise ValueError("build_sales: factor must be a column name of lines or a dict; for a conversion table use conv=, conv_item=, conv_unit=, conv_factor=")
        elif isinstance(factor, str): f = pd.to_numeric(L[factor], errors="coerce").fillna(1.0)
        elif isinstance(factor, dict) and factor:
            if isinstance(next(iter(factor)), tuple):
                f = pd.Series([factor.get((a, b), 1.0) for a, b in zip(L[item], L[unit] if unit else [None] * len(L))], index=L.index)
            elif unit is not None: f = L[unit].map(lambda u: factor.get(u, 1.0)).astype(float)
        out["factor"] = f; out["qty_base"] = q * f
    if unit_price is not None: out["unit_price"] = pd.to_numeric(L[unit_price], errors="coerce")
    disc = pd.to_numeric(L[line_discount], errors="coerce").fillna(0) if line_discount is not None else 0
    if line_discount is not None: out["line_discount"] = disc
    if amount is not None: out["line_amount"] = pd.to_numeric(L[amount], errors="coerce")
    elif unit_price is not None: out["line_amount"] = q * out["unit_price"] - disc
    else: raise ValueError("build_sales: give amount (line amount after line discount) or unit_price")
    wk = {ingest_norm(x) for x in WALKIN + tuple(walkin or ())}
    def is_walk(v): return v is None or (isinstance(v, float) and np.isnan(v)) or ingest_norm(v) in wk or ingest_norm(v) == ""
    walk = None
    if customer is not None:
        walk = L[customer].map(is_walk); out["customer_name"] = L[customer].where(~walk, None)
    if customer_id is not None:
        w2 = L[customer_id].map(is_walk); walk = w2 if walk is None else (walk | w2)
        out["customer_id"] = L[customer_id].where(~walk, None)
        if customer is not None: out["customer_name"] = out["customer_name"].where(~walk, None)
    for c in keep:
        if c in L.columns and c not in out.columns: out[c] = L[c]
    # statuses
    if status is not None:
        smap = {str(k).strip(): v for k, v in (status_map or {}).items()}
        raw = L[status]
        def m(v):
            if v is None or (isinstance(v, float) and np.isnan(v)): return None
            s = str(v).strip(); r = smap.get(s, smap.get(s.lower()))
            return _canon_label(r) if r is not None else _canon_label(s)
        st = raw.map(m)
        bad = sorted(set(raw[st.isna() & raw.notna()].astype(str)))
        if bad: raise ValueError(f"build_sales: status labels not mapped to {CANON}: {bad[:12]} (add them to status_map)")
        if st.isna().any() and order_id == "__row":
            raise ValueError(f"build_sales: {int(st.isna().sum())} lines have an empty status")
        out["line_status"] = st
        g = out.groupby("order_id", sort=False)["line_status"]
        has = lambda lab: g.transform(lambda s: (s == lab).any())
        first = g.transform("first")
        ostat = np.where(has("trả hàng"), "trả hàng", np.where(has("huỷ"), "huỷ", first))
        out["status"] = pd.Series(ostat, index=out.index).where(pd.notna(first), None)
        if out["status"].isna().any():
            raise ValueError(f"build_sales: {int(out['status'].isna().groupby(out['order_id']).any().sum())} orders have no status on any line")
    else:
        out["status"] = "hoàn thành"
    out["valid"] = ~out["status"].isin(["huỷ", "trả hàng"])
    out = out.reset_index(drop=True)
    # orders
    agg = {"revenue": ("line_amount", "sum"), "n_lines": ("line_amount", "size"), "status": ("status", "first"), "valid": ("valid", "first")}
    for c in ("order_time", "order_date", "customer_name", "customer_id"):
        if c in out.columns: agg[c] = (c, "first")
    orders = out.groupby("order_id", sort=False).agg(**agg).reset_index()
    if H is not None:                                                  # carry every header column (order-level money, channel, ...)
        hc = [c for c in H.columns if not str(c).startswith("_") and c != hkey and c not in orders.columns]
        orders = orders.merge(H[[hkey] + hc].drop_duplicates(hkey).rename(columns={hkey: "__k"}).assign(__k=lambda x: x["__k"].astype(str).str.strip()),
                              left_on="order_id", right_on="__k", how="left").drop(columns="__k")
    orders["valid"] = orders["valid"].astype(bool); out["valid"] = out["valid"].astype(bool)
    return orders, out
