#!/usr/bin/env python3
"""DataMate agent: messy business files -> clean tables -> (canonical tables) -> LLM-written pandas code -> executed
answers -> vote -> answer / unanswerable / abstain. Nothing is keyed on business or question ids.

Stages
  1. ingest.load_business: header detection, Vietnamese numbers / dates, multi-file merge (cached per business).
  2. compile (--compile, once per business): the LLM writes a program that turns the raw tables into canonical tables
     C = {orders, lines, items, purchases, payments, ...} + NOTES, following the rulebook and README. Kc candidates are
     generated, executed, schema-checked (errors go back to the LLM), and the candidate whose monthly signature
     (orders / valid orders / revenue / quantities) agrees most with the other candidates is kept.
  3. answer: K programs per question, split over two paths by --paths (A = uses C, B = raw tables T only); the first
     sample of each path is greedy, the others sampled with fixed seeds. Failed programs get the error back (repair
     rounds); 0 / empty / NaN results get one self-check round.
  4. vote with the organisers' matching rules (canonical keys); answer if top share >= tau else abstain.
  5. deterministic formatting -> submission.jsonl; per-question logs -> logs.jsonl (code, output, timing).

Usage: agent.py <split_dir> <out_dir> --model <hf id or path> [--k 4] [--paths AABB] [--compile] [--kc 4]
                [--repairs 2] [--subset N] [--tau 0.5] [--cache <dir>] [--tp N] [--dp N] [--think] [--workers N]"""
import os, re, sys, json, time, math, signal, pickle, argparse, traceback
# FlashInfer's sampler is JIT-compiled at start-up; on Kaggle L4 the build failed after 20 min (ld error). The PyTorch
# sampler gives the same results without compiling anything.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ingest
import dm, qparse, fewshot
import canon as canonmod
from score import norm

ENV_DOC = """Environment available to your program (already loaded, do NOT read files):
- `T`: dict table_name -> pandas DataFrame, cleaned: numeric columns are float, date columns are datetime64, text is NFC-normalised. Every table has `_source` (file[sheet]) and maybe `_sheet`; a `<col>__unit` column holds the unit text that followed a number (e.g. "bao 25kg").
- `RAW`: same tables with the original cell strings (use only if a cleaned column looks wrong).
- `CAL`: solar-lunar calendar DataFrame: date (datetime64), weekday, lunar_day, lunar_month, lunar_year, lunar_leap, holiday.
- helpers: `parse_num(x)`, `parse_date(x)`, `norm_text(s)` (lowercase, no punctuation), `strip_accents(s)`, `match_names(query, candidates, k=5)` -> best matching candidate strings (accent/typo tolerant),
  `month_range(y, m)` -> (first_day, last_day), `lunar_to_solar(lunar_year, lunar_month, lunar_day)` -> Timestamp, `round_half_up(x, ndigits=0)` (rulebook rounding: half rounds up; Python's round() does not).
- pandas as `pd`, numpy as `np`, `re`, `math`, `datetime`."""

ANSWER_DOC = """Finish by assigning `ANSWER = {"type": <one of number|percent|string|date|boolean|list|table|unanswerable>, "answer": <value>}`.
   - number/percent: a Python number (money = integer VND; averages / percentages rounded to 2 decimals unless the rulebook says otherwise, with round_half_up). percent 12.4% -> 12.4.
   - string: the name exactly as written in the data. date: "YYYY-MM-DD" (or "YYYY-MM" for month questions). boolean: True/False.
   - list: Python list in the order the question asks (one element is still a list). table: {"columns": [...], "rows": [[...], ...]} with the entity name first, then the measures in the order the question asks.
   - unanswerable: when the released data cannot answer (rulebook section on unanswerable questions), with "answer": None."""

SYSTEM_RAW = f"""You are DataMate, an expert data analyst for Vietnamese small businesses. You answer a question by writing ONE Python program that computes the answer from the business's tables with pandas.

{ENV_DOC}
- `TODAY`: pandas Timestamp of the as-of date ("hôm nay"). Records dated after TODAY have not happened yet unless the rulebook says otherwise.

Rules:
1. Follow the RULEBOOK exactly (definitions of revenue, valid orders, weeks, "tháng này", ranking and tie-breaks, debt, cost, rounding, answer formats). Where the business README explains its own columns, statuses or unit conversions, use it.
2. Inspect values you rely on (statuses, units, names) programmatically rather than guessing spellings; status labels differ between shops (e.g. "Xong", "Đã hoàn thành", "ok" = completed). Use match_names for names written loosely in the question.
3. {ANSWER_DOC}
4. You may print() short diagnostics. Output only one ```python code block, no explanation."""

SYSTEM_CANON = f"""You are DataMate, an expert data analyst for Vietnamese small businesses. You answer a question by writing ONE Python program that computes the answer with pandas.

{ENV_DOC}
- `C`: canonical tables already built for this business from T following the rulebook (described in the CANONICAL TABLES card together with NOTES). Prefer C; fall back to T for anything C does not cover.
- `TODAY`: pandas Timestamp of the as-of date ("hôm nay"). Records dated after TODAY have not happened yet unless the rulebook says otherwise.

Rules:
1. Follow the RULEBOOK exactly (periods, ranking and tie-breaks, debt, cost, rounding, answer formats). In C, `valid` already marks the orders that count for revenue and `revenue` / `line_amount` follow the rulebook definition; re-check the rulebook for anything else.
2. Inspect values you rely on programmatically rather than guessing spellings. Use match_names for names written loosely in the question.
3. {ANSWER_DOC}
4. You may print() short diagnostics. Output only one ```python code block, no explanation."""

SYSTEM_COMPILE = f"""You are DataMate's data engineer. Write ONE Python program that converts the raw tables of a Vietnamese small business into canonical tables, following the RULEBOOK and the business README. The canonical tables are reused to answer about a hundred questions, so they must be exactly right.

{ENV_DOC}
There is no TODAY here: keep the full history, including records after any date.

Build `C`, a dict of pandas DataFrames, using only what the data contains (omit tables or columns that do not exist; never invent values):
- C["orders"]: one row per sales order / invoice / bill: order_id, order_time (datetime, with time of day when recorded), order_date (= order_time normalised to the day), customer_id, customer_name (None for walk-in customers: khách lẻ / vãng lai / blank), customer_group, staff, channel, payment_method, status (exactly one of "hoàn thành", "huỷ", "trả hàng", "đang giao"), valid (bool: the order counts for revenue per the rulebook), revenue (rulebook revenue of the order: sum of line amounts after line discounts, excluding order-level discounts, VAT and delivery fees), order_discount, shipping_fee, vat, total_payable (what the customer must pay for the whole order), paid_at_purchase, plus any other order-level column that may matter.
- C["lines"]: one row per order line: order_id, order_time, order_date, status, valid, item_code, item_name, item_group, unit, qty, factor (units -> base unit), base_unit, qty_base, unit_price (per sold unit, before line discount), line_discount, line_amount (after line discount), unit_cost (per sold unit, when recorded on the line).
- C["items"]: product catalogue: item_code, item_name, item_group, base_unit, larger units with their factors, cost, price, and any attribute columns (prescription, active ingredient, size, ...).
- C["purchases"]: purchase (nhập hàng) lines: receipt_id, receipt_date, supplier, item_code, item_name, qty, unit, qty_base, unit_cost, amount, cancelled (bool).
- C["payments"]: later payments of credit (ghi nợ) orders: order_id, pay_date, amount, method.
- Any other useful table (stock snapshots, lots and expiry dates, expenses, price list, deliveries, warranty slips, shifts, ...) under a clear English name.
Rules:
1. Map every raw status label to the canonical statuses explicitly (inspect the unique values; read the README). An order with any returned item is "trả hàng".
2. Order-level cells left blank on the following lines of the same order inherit the order's first line (rulebook: "Cách ghi trong file"). Drop exact duplicate lines inside the same order; keep identical lines of different orders.
3. Use the README / conversion tables for qty_base. Keep codes as strings.
4. Set `NOTES` (a string): what the canonical tables mean for this shop, every assumption you made, and which kinds of information are NOT recorded (e.g. no cost price, no supplier, no customer address), so that questions needing them are unanswerable.
5. End with print() of a short summary: row count per table, status counts, valid revenue of the first 3 months.
6. The tables are ALREADY loaded in T: never open, read or glob files. Assign the result to the variable `C` (a dict) and `NOTES` (a string).
Keep the program compact (about 150 lines at most); output only one ```python code block, no explanation."""


SYSTEM_LIB = f"""You are DataMate, an expert data analyst for Vietnamese small businesses. You answer a question by writing ONE short Python program that calls the DataMate toolkit on the business's canonical tables.

{dm.API_DOC}
- `TODAY`: pandas Timestamp of the as-of date ("hôm nay").

Rules:
1. Read the question carefully: period, measure (revenue, quantity in base units, number of orders, ...), filters, ranking size k, and the answer type. Follow the RULEBOOK for every definition; the toolkit already implements the defaults, so prefer a toolkit call over re-implementing it.
2. Use the exact names, codes and labels written in the data (see the CANONICAL TABLES card); names in the question may be written loosely (no accents, abbreviations): pick the matching value from the card.
3. If the data does not record what the question needs (the NOTES say what is missing), or the period lies outside the data, `raise Unanswerable("<reason>")`.
4. {ANSWER_DOC}
5. Output only one ```python code block, no explanation. The EXAMPLES come from other shops (their names and data differ); use them for the method only."""

SYSTEM_MAP = f"""You are DataMate's data engineer. Write ONE short Python program that maps the raw tables of a Vietnamese small business onto canonical tables, following the business README and the RULEBOOK. About a hundred questions are answered from these tables, so the mapping must be exactly right.

{dm.MAP_DOC}

Canonical tables (use only what the data contains; never invent values):
{dm.CANON_DOC}

Rules:
1. Map every raw column you need by its exact name. Map status labels with map_status (inspect the README for statuses written in notes or for returns written as negative quantities). Keep codes as text.
2. Customer names: walk-in customers ("khách lẻ", "lẻ", "kl", blank) become None automatically. Use the customer list to add names / groups when orders only carry a code.
3. Build C["orders"] and C["lines"] with finalize, plus the other tables the business has (items with item_group / cost / prescription / active_ingredient, purchases, payments with pay_date, stock, lots, prices, expenses, deliveries, conversions).
4. Set NOTES (a string): what the tables mean for this shop and which kinds of information are NOT recorded (no cost price, no customers, no time of day, ...).
5. The tables are already loaded in T: never read files. Output only one ```python code block (about 15-40 lines), no explanation. The EXAMPLES are mapping programs written for other shops."""


def user_msg_lib(rulebook, biz_readme, canon_card, examples, q):
    return "\n\n".join([f"# RULEBOOK\n{rulebook}", f"# BUSINESS README\n{biz_readme.strip() or '(empty)'}",
                         f"# CANONICAL TABLES (in C; metrics use them)\n{canon_card}",
                         f"# EXAMPLES (other shops)\n{examples or '(none)'}",
                         f"# QUESTION\nas_of_date (TODAY): {q['as_of_date']}\nquestion: {q['question']}\n\nWrite the program."])


def user_msg_map(rulebook, biz_readme, card, examples):
    return "\n\n".join([f"# RULEBOOK (extract)\n{rulebook}", f"# EXAMPLES (mapping programs of other shops)\n{examples}",
                         f"# BUSINESS README\n{biz_readme.strip() or '(empty)'}", f"# RAW TABLES (in T)\n{card}",
                         "# TASK\nWrite the mapping program for this business."])


def user_msg(rulebook, biz_readme, card, q=None, canon_card=None, notes=None):
    parts = [f"# RULEBOOK\n{rulebook}", f"# BUSINESS README\n{biz_readme.strip() or '(empty)'}", f"# RAW TABLES (in T)\n{card}"]
    if canon_card is not None:
        parts.append(f"# CANONICAL TABLES (in C)\n{canon_card}\n\nNOTES:\n{notes or ''}")
    if q is None: parts.append("Write the program that builds C and NOTES.")
    else: parts.append(f"# QUESTION\nas_of_date (TODAY): {q['as_of_date']}\nquestion: {q['question']}\n\nWrite the program.")
    return "\n\n".join(parts)


CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = CODE_RE.findall(text)
    if m: return max(m, key=len)
    m = re.search(r"```(?:python|py)?\s*\n(.*)$", text, re.S)                 # unterminated block (length cut)
    if m and ("ANSWER" in m.group(1) or re.search(r"\bC\s*(=|\[)", m.group(1))): return m.group(1)
    return None


# ------------------------------------------------------------------ helpers exposed to programs
def month_range(y, m):
    s = pd.Timestamp(int(y), int(m), 1); return s, s + pd.offsets.MonthEnd(0)


def round_half_up(x, ndigits=0):
    q = Decimal(1).scaleb(-ndigits)
    v = float(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))
    return int(v) if ndigits == 0 else v


def norm_text(s): return ingest.norm_text(s)


def match_names(query, candidates, k=5):
    import difflib
    cands = list(dict.fromkeys(str(c) for c in candidates if c is not None and str(c) != "nan"))
    qn = ingest.strip_accents(norm_text(query))
    keyed = {}
    for c in cands: keyed.setdefault(ingest.strip_accents(norm_text(c)), c)
    exact = [c for kk, c in keyed.items() if kk == qn]
    sub = [c for kk, c in keyed.items() if qn and (qn in kk or kk in qn) and c not in exact]
    fuzzy = [keyed[x] for x in difflib.get_close_matches(qn, list(keyed), n=k, cutoff=0.5)]
    return list(dict.fromkeys(exact + sub + fuzzy))[:k]


# ------------------------------------------------------------------ sandboxed execution (worker pool)
_BIZ, _CAN, _CTX = {}, {}, {}


def _init_worker(cache_dir, split_dir):
    _CTX["cache"], _CTX["split"] = cache_dir, split_dir
    cal = pd.read_csv(os.path.join(split_dir, "calendar.csv"))
    cal["date"] = pd.to_datetime(cal["date"]); _CTX["cal"] = cal; dm.set_calendar(cal)


def _lru(d, key, loader, cap=6):
    if key not in d:
        if len(d) >= cap: d.pop(next(iter(d)))
        d[key] = loader()
    return d[key]


def _namespace(bid, as_of=None, canon_path=None):
    biz = _lru(_BIZ, bid, lambda: load_biz(_CTX["cache"], _CTX["split"], bid))
    cal = _CTX["cal"]
    def lunar_to_solar(ly, lm, ld, leap=False):
        r = cal[(cal.lunar_year == int(ly)) & (cal.lunar_month == int(lm)) & (cal.lunar_day == int(ld)) &
                (cal.lunar_leap.astype(str).str.lower().eq("true") == bool(leap))]
        return r.date.iloc[0] if len(r) else pd.NaT
    import datetime
    g = {"T": {k: v.copy() for k, v in biz.tables.items()}, "RAW": biz.raw, "CAL": cal.copy(), "pd": pd, "np": np, "re": re,
         "math": math, "datetime": datetime, "parse_num": ingest.parse_num, "parse_date": ingest.parse_date, "norm_text": norm_text,
         "strip_accents": ingest.strip_accents, "match_names": match_names, "month_range": month_range, "round_half_up": round_half_up,
         "lunar_to_solar": lunar_to_solar, "__name__": "__datamate__"}
    if as_of is not None: g["TODAY"] = pd.Timestamp(as_of)
    if canon_path:
        can = _lru(_CAN, canon_path, lambda: pickle.load(open(canon_path, "rb")))
        if can.get("v2"):                                              # toolkit programs: metrics bound to this shop + day
            shop = dm.Shop(can["C"], as_of, {"data_end": can.get("data_end")})
            g.update(dm.api(shop, as_of if as_of is not None else shop.last)); g["T"] = {k: v.copy() for k, v in biz.tables.items()}
        else: g["C"] = {k: v.copy() for k, v in can["C"].items()}
        g["NOTES"] = can.get("notes", "")
    return g


class _Timeout(Exception): pass


def _alarm(*_): raise _Timeout()


def _exec(code, g, timeout):
    import io, contextlib
    buf = io.StringIO()
    signal.signal(signal.SIGALRM, _alarm); signal.alarm(timeout)
    try:
        with contextlib.redirect_stdout(buf): exec(compile(code, "<program>", "exec"), g)
    finally: signal.alarm(0)
    return buf.getvalue()


def _err(e, out=""):
    if isinstance(e, _Timeout): return {"ok": False, "error": "timeout: the program ran too long", "stdout": out[-800:]}
    tb = traceback.format_exc().splitlines()
    where = [l.strip() for l in tb if 'File "<program>"' in l][-2:]        # program lines involved
    msg = f"{type(e).__name__}: {e}"[:600]
    if isinstance(e, FileNotFoundError): msg += "\n(Do not read files: every table is already loaded in T.)"
    return {"ok": False, "error": msg + ("\nat " + " -> ".join(where) if where else ""), "stdout": out[-800:]}


def run_code(task):
    """task = (key, bid, as_of, code, timeout, canon_path) -> (key, result)"""
    key, bid, as_of, code, timeout, canon_path = task
    t0 = time.time(); out = ""
    try:
        g = _namespace(bid, as_of, canon_path)
        out = _exec(code, g, timeout)
        if "ANSWER" not in g: r = {"ok": False, "error": "the program did not assign ANSWER", "stdout": out[-1500:]}
        else: r = {"ok": True, "answer": fmt_answer(g["ANSWER"]), "stdout": out[-1500:]}
    except dm.Unanswerable as e: r = {"ok": True, "answer": {"type": "unanswerable", "answer": None}, "reason": str(e)[:200], "stdout": out[-600:]}
    except BaseException as e: r = _err(e, out)
    r["t"] = time.time() - t0
    return key, r


CANON_STATUS = {"hoàn thành", "huỷ", "trả hàng", "đang giao"}


def check_canon(C):
    """Schema checks of a compiled C. Returns (errors, warnings)."""
    errs, warns = [], []
    if not isinstance(C, dict) or not C: return ["C must be a non-empty dict of DataFrames"], []
    for k, v in C.items():
        if not isinstance(v, pd.DataFrame): errs.append(f"C[{k!r}] is {type(v).__name__}, not a DataFrame")
    if errs: return errs, warns
    o = C.get("orders")
    if o is not None:
        for c in ("order_id", "order_date", "status", "valid"):
            if c not in o.columns: errs.append(f"C['orders'] lacks column {c!r}")
        if "status" in o.columns:
            if o["status"].isna().any(): errs.append(f"C['orders'].status is empty for {int(o['status'].isna().sum())} orders (a raw status label was not mapped)")
            bad = sorted(set(o["status"].dropna().astype(str)) - CANON_STATUS)
            if bad: errs.append(f"C['orders'].status has non-canonical values {bad[:8]} (allowed: {sorted(CANON_STATUS)})")
        if "order_id" in o.columns and o["order_id"].duplicated().any():
            errs.append(f"C['orders'].order_id is not unique ({int(o['order_id'].duplicated().sum())} duplicates); orders must have one row per order")
        if "valid" in o.columns and not pd.api.types.is_bool_dtype(o["valid"]): errs.append("C['orders'].valid must be a bool column")
        if "order_date" in o.columns and not pd.api.types.is_datetime64_any_dtype(o["order_date"]): errs.append("C['orders'].order_date must be datetime64")
        if "valid" in o.columns and "status" in o.columns and pd.api.types.is_bool_dtype(o["valid"]):
            if o.loc[o["status"].isin(["huỷ", "trả hàng"]), "valid"].any(): errs.append("cancelled or returned orders are marked valid")
    l = C.get("lines")
    if l is not None:
        for c in ("order_id", "item_name", "qty"):
            if c not in l.columns: warns.append(f"C['lines'] lacks column {c!r}")
        if o is not None and "order_id" in l.columns and "order_id" in o.columns:
            miss = ~l["order_id"].isin(set(o["order_id"]))
            if miss.mean() > 0.01: errs.append(f"{int(miss.sum())} lines have an order_id that is not in C['orders']")
    return errs, warns


def signature(C):
    """Monthly aggregates used to compare compile candidates."""
    sig = {}
    try:
        o = C.get("orders")
        if o is not None and {"order_date", "valid"} <= set(o.columns):
            m = pd.to_datetime(o["order_date"]).dt.to_period("M").astype(str)
            sig.update({f"orders:{k}": int(v) for k, v in o.groupby(m).size().items()})
            v = o["valid"].astype(bool)
            sig.update({f"valid:{k}": int(x) for k, x in v.groupby(m).sum().items()})
            if "revenue" in o.columns:
                sig.update({f"rev:{k}": int(round(float(x))) for k, x in o.loc[v, "revenue"].groupby(m[v]).sum().items()})
        l = C.get("lines")
        if l is not None and {"order_date", "qty_base", "valid"} <= set(l.columns):
            m = pd.to_datetime(l["order_date"]).dt.to_period("M").astype(str); v = l["valid"].astype(bool)
            sig.update({f"qty:{k}": round(float(x), 2) for k, x in l.loc[v, "qty_base"].groupby(m[v]).sum().items()})
    except Exception as e: sig["error"] = type(e).__name__
    return sig


def run_mapping(task):
    """task = (key, bid, code, ref_id_or_None, out_path) -> (key, result): a v2 mapping program (reference transfer or LLM),
    executed and checked; the canonical tables are pickled to out_path."""
    key, bid, code, rid, out_path = task
    t0 = time.time()
    try:
        biz = _lru(_BIZ, bid, lambda: load_biz(_CTX["cache"], _CTX["split"], bid))
        meta = canonmod.load_meta().get(rid, {}).get("tables") if rid else None
        signal.signal(signal.SIGALRM, _alarm); signal.alarm(300)
        try: r = canonmod.run_mapping(code, biz, meta)
        finally: signal.alarm(0)
        if r["ok"]:
            C = r.pop("C"); pickle.dump({"C": C, "notes": r["notes"], "v2": True, "data_end": canonmod.data_end(biz)}, open(out_path, "wb"), protocol=4)
            r.update(path=out_path, sig=signature(C), card=canonmod.card(C, r["notes"]))
        else: r.pop("C", None)
    except BaseException as e: r = _err(e)
    r["t"] = time.time() - t0
    return key, r


def run_compile(task):
    """task = (key, bid, code, timeout, out_path) -> (key, result); C is pickled to out_path."""
    key, bid, code, timeout, out_path = task
    t0 = time.time(); out = ""
    try:
        g = _namespace(bid)
        out = _exec(code, g, timeout)
        C = g.get("C"); errs, warns = check_canon(C)
        if errs: r = {"ok": False, "error": "canonical tables failed the checks:\n- " + "\n- ".join(errs), "stdout": out[-1200:]}
        else:
            notes = str(g.get("NOTES", ""))
            pickle.dump({"C": C, "notes": notes}, open(out_path, "wb"), protocol=4)
            card = "\n".join(ingest.table_card(k, v) for k, v in C.items())
            r = {"ok": True, "sig": signature(C), "card": card, "notes": notes, "warnings": warns, "stdout": out[-1500:], "path": out_path}
    except BaseException as e: r = _err(e, out)
    r["t"] = time.time() - t0
    return key, r


def agreement(sigs):
    """index of the candidate whose signature agrees with the others on the most keys (ties -> lowest index)."""
    def agree(a, b):
        return sum(1 for k in set(a) & set(b) if (abs(a[k] - b[k]) <= 1 if isinstance(a[k], (int, float)) else a[k] == b[k]))
    scores = [sum(agree(s, t) for j, t in enumerate(sigs) if j != i) for i, s in enumerate(sigs)]
    return max(range(len(sigs)), key=lambda i: (scores[i], -i)), scores


# ------------------------------------------------------------------ formatting / canonical keys
TYPES = {"number", "percent", "string", "date", "boolean", "list", "table", "unanswerable"}


def _py(v):
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): v = float(v)
    if isinstance(v, (np.bool_,)): return bool(v)
    if isinstance(v, (pd.Timestamp,)): return v.strftime("%Y-%m-%d")
    if isinstance(v, float):
        if not math.isfinite(v): raise ValueError("non-finite number in the answer (NaN / inf): check filters and empty selections")
        v = round(v, 2); return int(v) if v == int(v) else v
    return v


def fmt_answer(a):
    if not isinstance(a, dict) or a.get("type") not in TYPES: raise ValueError(f"ANSWER must be a dict with a valid type, got {a!r}"[:300])
    t, v = a["type"], a.get("answer")
    if t == "unanswerable": return {"type": t, "answer": None}
    if v is None: raise ValueError("answer is None (use type unanswerable only if the released data cannot answer)")
    if isinstance(v, (pd.Series, np.ndarray, pd.Index)): v = list(v)
    if t in ("number", "percent"):
        if isinstance(v, (list, tuple)) and len(v) == 1: v = v[0]
        v = _py(v) if not isinstance(v, str) else _py(float(ingest.parse_num(v)))
        if not isinstance(v, (int, float)) or isinstance(v, bool): raise ValueError("number answer is not numeric")
    elif t == "string":
        if isinstance(v, (list, tuple)) and len(v) == 1: v = v[0]
        v = str(v).strip()
        if not v: raise ValueError("empty string answer")
    elif t == "date":
        if not (isinstance(v, str) and re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", v.strip())): v = pd.Timestamp(v).strftime("%Y-%m-%d")
        v = v.strip()
    elif t == "boolean":
        if isinstance(v, str): v = v.strip().lower() in ("true", "có", "co", "yes", "1")
        v = bool(v)
    elif t == "list":
        if not isinstance(v, (list, tuple)): v = [v]
        v = [_py(x) if not isinstance(x, str) else x for x in v]
    elif t == "table":
        if isinstance(v, pd.DataFrame): v = {"columns": [str(c) for c in v.columns], "rows": v.values.tolist()}
        v = {"columns": [str(c) for c in v["columns"]], "rows": [[_py(x) if not isinstance(x, str) else x for x in r] for r in v["rows"]]}
    json.dumps(v)
    return {"type": t, "answer": v}


def canon(a):
    """Hashable key: answers equal under the organisers' matching rules get the same key."""
    t, v = a["type"], a.get("answer")
    def el(x): return ("n", round(float(x), 2)) if isinstance(x, (int, float)) and not isinstance(x, bool) else ("s", norm(x))
    if t in ("number", "percent"): return (t, round(float(v), 2))
    if t == "string": return (t, norm(v))
    if t == "list": return (t, tuple(el(x) for x in v))
    if t == "table": return (t, tuple(norm(c) for c in v["columns"]), tuple(sorted(tuple(el(x) for x in r) for r in v["rows"])))
    return (t, json.dumps(v, ensure_ascii=False, sort_keys=True))


def suspicious(a):
    """Results that usually mean a wrong filter: 0, empty list/table."""
    t, v = a["type"], a.get("answer")
    return (t in ("number", "percent") and v == 0) or (t == "list" and len(v) == 0) or (t == "table" and len(v["rows"]) == 0)


# ------------------------------------------------------------------ data
CARDS = {}


def load_biz(cache_dir, split_dir, bid):
    p = os.path.join(cache_dir, bid + ".pkl") if cache_dir else None
    if p and os.path.isfile(p): return pickle.load(open(p, "rb"))
    biz = ingest.load_business(os.path.join(split_dir, "data", bid))
    if p: os.makedirs(cache_dir, exist_ok=True); pickle.dump(biz, open(p, "wb"), protocol=4)
    return biz


def select_subset(qs, subset):
    byb = defaultdict(list)
    for q in sorted(qs, key=lambda q: q["id"]): byb[q["business_id"]].append(q)
    per = max(1, subset // len(byb)); return [q for b in sorted(byb) for q in byb[b][:per]]


def shard_of(qs, n):
    """business -> shard, greedy balance of question counts (deterministic)."""
    cnt = Counter(q["business_id"] for q in qs); load = [0] * n; out = {}
    for b, c in sorted(cnt.items(), key=lambda x: (-x[1], x[0])):
        j = load.index(min(load)); out[b] = j; load[j] += c
    return out


# ------------------------------------------------------------------ LLM engines
class Engine:
    def __init__(self, a):
        from vllm import LLM
        import torch
        tp = a.tp or max(1, torch.cuda.device_count())
        half = torch.cuda.get_device_capability()[0] < 8
        self.llm = LLM(model=a.model, tensor_parallel_size=tp, dtype="half" if half else "auto", max_model_len=a.max_len,
                       gpu_memory_utilization=0.92, enable_prefix_caching=True, seed=0, enforce_eager=a.eager,
                       **({"enable_lora": True, "max_lora_rank": 64} if a.lora else {}))
        self.lora = None
        if a.lora:                                                     # SFT adapter (src/train_sft.py) on the quantised base
            from vllm.lora.request import LoRARequest
            self.lora = LoRARequest("sft", 1, a.lora)
        self.tok = self.llm.get_tokenizer(); self.a = a

    def render(self, msgs):
        kw = {"enable_thinking": self.a.think} if "qwen3" in self.a.model.lower() else {}
        try: return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        except TypeError: return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def fit(self, conv, max_tokens):
        """Render a conversation within max_len: drop older repair turns first, then shrink the output budget.
        Returns (prompt, max_tokens) or (None, 0) when there is no room (never let one long prompt fail the batch)."""
        for c in (conv, conv[:2] + conv[-2:] if len(conv) > 4 else None, conv[:2] if len(conv) > 2 else None):
            if c is None: continue
            p = self.render(c); n = len(self.tok.encode(p, add_special_tokens=False))
            room = self.a.max_len - n - 16
            if room >= min(max_tokens, 768): return p, min(max_tokens, room)
        return None, 0

    def gen(self, convs, temps, seeds, max_tokens):
        """Requests are generated one prefix group at a time (rulebook + README + card of one business and path, ~10-18k
        tokens): the group's prefix is prefilled once by a 1-token warm-up request and then served from the prefix
        cache to every request of the group. Sending all businesses at once made the prefixes evict each other from the
        KV cache (E2b: ~12x below the expected throughput)."""
        from vllm import SamplingParams
        fitted = [self.fit(c, max_tokens) for c in convs]
        idx = [i for i, (p, _) in enumerate(fitted) if p is not None]
        groups = {}
        for i in idx:
            p = fitted[i][0]; cut = p.rfind("# QUESTION")
            groups.setdefault(p[:cut] if cut > 0 else p, []).append(i)
        texts = [""] * len(convs); ntok = 0
        for prefix, members in groups.items():
            if len(members) > 1:
                self.llm.generate([fitted[members[0]][0]], [SamplingParams(max_tokens=1)], use_tqdm=False, lora_request=self.lora)
            sps = [SamplingParams(temperature=temps[i], top_p=1.0 if temps[i] == 0 else 0.95, max_tokens=fitted[i][1], seed=seeds[i]) for i in members]
            outs = self.llm.generate([fitted[i][0] for i in members], sps, use_tqdm=False, lora_request=self.lora)
            for i, o in zip(members, outs): texts[i] = o.outputs[0].text; ntok += len(o.outputs[0].token_ids)
        if len(idx) < len(convs): print(f"  {len(convs) - len(idx)} prompts too long for max_len={self.a.max_len}: skipped", flush=True)
        return texts, ntok


class FakeEngine:
    """Local testing without a GPU: responder(conv) -> text."""
    def __init__(self, responder): self.responder = responder
    def gen(self, convs, temps, seeds, max_tokens):
        outs = [self.responder(c) for c in convs]; return outs, sum(len(o) // 4 for o in outs)


# ------------------------------------------------------------------ pipeline
def generate_and_run(engine, pool, items, greedy, make_task, runner, repairs, max_tokens, temp, label, reflect=False):
    """items: {key: conversation}, key = (id, sample index); keys in `greedy` decode at temperature 0, the others sample
    with seeds fixed by (round, sample index). Generation + execution + repair rounds.
    Returns {key: [attempt dicts]}; the last ok attempt is the sample's result."""
    attempts = defaultdict(list); convs = dict(items); pending = sorted(convs); reflected = set()
    for rnd in range(repairs + 1):
        if not pending: break
        t0 = time.time()
        temps = [0.0 if k in greedy else temp for k in pending]
        seeds = [1000 * rnd + 17 * k[1] + 1 for k in pending]
        texts, ntok = engine.gen([convs[k] for k in pending], temps, seeds, max_tokens)
        t1 = time.time(); tasks, codes, nxt = [], {}, []
        for k, text in zip(pending, texts):
            convs[k] = convs[k] + [{"role": "assistant", "content": text}]
            code = extract_code(text)
            if code is None:
                attempts[k].append({"round": rnd, "ok": False, "error": "no python code block", "gen": text[-400:]})
                if rnd < repairs:
                    convs[k].append({"role": "user", "content": "No ```python block was found. Output only the complete program in one ```python block."}); nxt.append(k)
                continue
            codes[k] = code; tasks.append(make_task(k, code))
        for k, r in pool.imap_unordered(runner, tasks, chunksize=2):
            att = {"round": rnd, "code": codes[k], **{x: r[x] for x in r if x not in ("t", "card")}, "exec_s": round(r["t"], 2)}
            if "card" in r: att["card"] = r["card"]
            attempts[k].append(att)
            if not r["ok"]:
                if rnd < repairs:
                    convs[k].append({"role": "user", "content": f"The program failed:\n{r['error']}\nstdout:\n{r.get('stdout', '')[-600:]}\n"
                                                                "Fix it and output the full corrected program."}); nxt.append(k)
            elif reflect and rnd < repairs and k not in reflected and suspicious(r["answer"]):
                reflected.add(k); att["reflect"] = True
                convs[k].append({"role": "user", "content": f"The program returned {json.dumps(r['answer'], ensure_ascii=False)}. A zero or empty result often means "
                                                            "a wrong filter (status labels, date range, name spelling: use match_names, unit). Re-check each filter "
                                                            "against the actual values. If the result is really correct, output the same program again."})
                nxt.append(k)
        print(f"[{label}] round {rnd}: {len(pending)} gens, {ntok} tok in {t1 - t0:.0f}s ({ntok / max(t1 - t0, 1e-9):.0f} tok/s), "
              f"exec {time.time() - t1:.0f}s, {len(nxt)} to redo", flush=True)
        pending = sorted(nxt)
    return attempts


def last_ok(atts):
    ok = [x for x in atts if x.get("ok")]
    return ok[-1] if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("out")
    ap.add_argument("--model", default="Qwen/Qwen3-14B-AWQ"); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--paths", default="", help="path of each sample: A = canonical C, B = raw T (default: all B, or AABB.. with --compile)")
    ap.add_argument("--compile", action="store_true"); ap.add_argument("--kc", type=int, default=4)
    ap.add_argument("--repairs", type=int, default=2); ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.5); ap.add_argument("--cache", default="")
    ap.add_argument("--tp", type=int, default=0); ap.add_argument("--max-len", type=int, default=24576)
    ap.add_argument("--think", action="store_true"); ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=2048); ap.add_argument("--compile-tokens", type=int, default=6144)
    ap.add_argument("--timeout", type=int, default=60); ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--eager", action="store_true"); ap.add_argument("--no-reflect", action="store_true")
    ap.add_argument("--lora", default="", help="LoRA adapter dir from train_sft.py")
    ap.add_argument("--dp", type=int, default=1, help="data parallel: one model copy per GPU, businesses split across them")
    ap.add_argument("--shard", default="", help="internal: i/N")
    ap.add_argument("--chunk", type=int, default=0, help="businesses per chunk (answers appended after each chunk)")
    ap.add_argument("--budget", type=float, default=0, help="wall-clock budget in seconds: fast mode / abstain when running out")
    ap.add_argument("--log-dir", default="", help="also write one JSON log per question here (Docker: /result/logs)")
    a = ap.parse_args()
    if a.dp > 1 and not a.shard: return run_dp(a)
    return run(a)


def run(a, engine=None):
    t_start = time.time(); os.makedirs(a.out, exist_ok=True)
    cache = a.cache or os.path.join(a.out, "cache")
    qs = [json.loads(l) for l in open(os.path.join(a.split, "questions.jsonl"), encoding="utf-8") if l.strip()]
    if a.subset: qs = select_subset(qs, a.subset)
    if a.shard:
        i, n = map(int, a.shard.split("/")); sh = shard_of(qs, n); qs = [q for q in qs if sh[q["business_id"]] == i]
    rulebook = open(os.path.join(a.split, "rulebook.md"), encoding="utf-8").read()
    bids = sorted({q["business_id"] for q in qs})

    # 1. ingest (parallel, cached)
    from concurrent.futures import ProcessPoolExecutor
    nw = a.workers or max(1, (os.cpu_count() or 4))
    os.makedirs(cache, exist_ok=True)
    todo = [b for b in bids if not os.path.isfile(os.path.join(cache, b + ".pkl"))]
    if todo:
        with ProcessPoolExecutor(nw) as ex: list(ex.map(ingest._cache_one, [(a.split, cache, b) for b in todo]))
    bizs = {b: load_biz(cache, a.split, b) for b in bids}
    for b, z in bizs.items(): CARDS[b] = z.card()
    print(f"ingest {len(bids)} businesses {time.time() - t_start:.0f}s", flush=True)

    engine = engine or Engine(a)
    print(f"model ready {time.time() - t_start:.0f}s", flush=True)
    import multiprocessing as mp
    pool = mp.get_context("fork").Pool(nw, initializer=_init_worker, initargs=(cache, a.split), maxtasksperchild=100)
    full_paths = (a.paths or ("".join("A" if i % 2 == 0 else "B" for i in range(a.k)) if a.compile else "B" * a.k))
    full_paths = (full_paths * a.k)[:a.k]

    # answers are appended chunk by chunk so that a time-out still leaves every finished answer scored (Rule 8)
    sub = open(os.path.join(a.out, "submission.jsonl"), "w", encoding="utf-8")
    logf = open(os.path.join(a.out, "logs.jsonl"), "w", encoding="utf-8")
    clog = open(os.path.join(a.out, "compile_logs.jsonl"), "w", encoding="utf-8")
    if a.log_dir: os.makedirs(a.log_dir, exist_ok=True)
    chunks = [bids[i:i + a.chunk] for i in range(0, len(bids), a.chunk)] if a.chunk else [bids]
    sec_per_q = None
    for ci, cb in enumerate(chunks):
        cq = [q for q in qs if q["business_id"] in set(cb)]; t_chunk = time.time(); elapsed = t_chunk - t_start
        mode = "full"
        if a.budget:
            if elapsed > 0.97 * a.budget: mode = "abstain"
            elif sec_per_q and elapsed + sec_per_q * len(cq) > 0.92 * a.budget: mode = "fast"
        print(f"=== chunk {ci + 1}/{len(chunks)}: {len(cb)} businesses, {len(cq)} questions, mode={mode}, elapsed {elapsed:.0f}s", flush=True)
        if mode == "abstain":
            for q in cq: write_answer(sub, logf, a, q, {"type": "abstain"}, 0.0, Counter(), {}, {}, 0.0)
            continue
        paths = full_paths if mode == "full" else "B"
        k = len(paths)
        canon = compile_stage(engine, pool, a, rulebook, bizs, cb, clog) if (a.compile and "A" in paths) else {}
        qatt, path_of, by_q = answer_stage(engine, pool, a, rulebook, bizs, cq, canon, paths, a.repairs if mode == "full" else 1)
        per_q = (time.time() - t_chunk) / max(len(cq), 1)
        for q in cq:
            res = [(kk, last_ok(qatt[kk])) for kk in sorted(by_q[q["id"]])]
            res = [(kk, r["answer"]) for kk, r in res if r]
            votes = Counter(canon_of(x) for _, x in res); rep = {canon_of(x): x for _, x in res}
            if votes:
                top, n = votes.most_common(1)[0]; conf = n / k
                final = rep[top] if conf >= a.tau else {"type": "abstain"}
            else: conf, final = 0.0, {"type": "abstain"}
            write_answer(sub, logf, a, q, final, conf, votes, {f"{kk[1]}{path_of[kk]}": qatt[kk] for kk in sorted(by_q[q["id"]])}, path_of, per_q)
        sub.flush(); logf.flush()
        sec_per_q = (time.time() - t_start) / max(1, sum(len([q for q in qs if q["business_id"] in set(c)]) for c in chunks[:ci + 1]))
    pool.close(); sub.close(); logf.close(); clog.close()
    print(f"DONE {len(qs)} questions in {time.time() - t_start:.0f}s", flush=True)
    score_if_possible(a, qs)


def write_answer(sub, logf, a, q, final, conf, votes, samples, path_of, elapsed):
    sub.write(json.dumps({"id": q["id"], **final}, ensure_ascii=False) + "\n")
    rec = {"id": q["id"], "question": q["question"], "business_id": q["business_id"], "as_of_date": q["as_of_date"], "final": final,
           "conf": conf, "elapsed_s": round(elapsed, 2), "votes": [[str(k), v] for k, v in votes.most_common()], "samples": samples}
    logf.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    if a.log_dir:                                                      # one file per question (Docker log requirement)
        json.dump(rec, open(os.path.join(a.log_dir, f"{q['id']}.json"), "w", encoding="utf-8"), ensure_ascii=False, default=str, indent=1)


def compile_stage(engine, pool, a, rulebook, bizs, bids, clog):
    cdir = os.path.join(a.cache or a.out, "canon_" + (a.shard.replace("/", "of") or "main")); os.makedirs(cdir, exist_ok=True); canon = {}
    items = {(b, s): [{"role": "system", "content": SYSTEM_COMPILE}, {"role": "user", "content": user_msg(rulebook, bizs[b].readme, CARDS[b])}]
             for b in bids for s in range(a.kc)}
    catt = generate_and_run(engine, pool, items, {k for k in items if k[1] == 0},
                            lambda k, code: (k, k[0], code, 300, os.path.join(cdir, f"{k[0]}.c{k[1]}.pkl")),
                            run_compile, a.repairs, a.compile_tokens, a.temp, "compile")
    for b in bids:
        cands = [(s, last_ok(catt[(b, s)])) for s in range(a.kc)]
        cands = [(s, r) for s, r in cands if r]
        if cands:
            best, scores = agreement([r["sig"] for _, r in cands])
            s, r = cands[best]; canon[b] = r
            print(f"  {b}: {len(cands)}/{a.kc} compiled, chose c{s} agreement {scores}", flush=True)
        else: print(f"  {b}: compile failed", flush=True)
        clog.write(json.dumps({"business_id": b, "chosen": canon[b]["path"] if b in canon else None,
                               "attempts": {s: catt[(b, s)] for s in range(a.kc)}}, ensure_ascii=False, default=str) + "\n")
    clog.flush()
    return canon


def rulebook_extract(rulebook):
    """the rulebook sections a mapping needs: orders / revenue, industry rules, how files are written."""
    secs = re.split(r"\n(?=## )", rulebook)
    keep = [x for x in secs if re.match(r"## \d+\.\s*(Đơn hàng|Theo ngành|Cách ghi)", x)]
    return "\n".join(keep) if keep else rulebook


def resolvable(biz, ref_meta):
    """cheap pre-check: every table a reference program reads has a counterpart (>= 70 % of its columns) in biz."""
    tabs = ref_meta.get("tables") or {}
    if not tabs: return False
    have = [set(df.columns) for df in biz.tables.values()]
    return all(max((len(set(cols) & h) / max(1, len(cols)) for h in have), default=0) >= 0.7 for cols in tabs.values())


def map_examples(biz, bid, exclude_own, k=2):
    """the k reference mapping programs of the most similar formats, with the cards of the tables they read."""
    meta = canonmod.load_meta(); progs = canonmod.ref_programs()
    cards = json.load(open(os.path.join(canonmod.REF_DIR, "_cards.json"), encoding="utf-8")) if os.path.isfile(os.path.join(canonmod.REF_DIR, "_cards.json")) else {}
    out = []
    for rid in canonmod.nearest_refs(biz, k, meta, exclude=(bid,) if exclude_own else ()):
        out.append(f"## Example shop: raw tables\n{cards.get(rid, '(card not available)')}\n## Its mapping program\n```python\n{progs[rid].strip()}\n```")
    return "\n\n".join(out)


def canon_stage_v2(engine, pool, a, rulebook, bizs, bids, clog):
    """canonical tables per business: reference programs transferred by table columns; the LLM writes a mapping (with the
    two most similar reference programs as examples) for businesses no reference fits (or always with --map-llm)."""
    cdir = os.path.join(a.cache or a.out, "canon2_" + (a.shard.replace("/", "of") or "main")); os.makedirs(cdir, exist_ok=True)
    exclude_own = os.path.isfile(os.path.join(a.split, "answers.jsonl"))      # honest train runs: never a shop's own program
    progs, meta = canonmod.ref_programs(), canonmod.load_meta()
    tasks = [((b, rid), b, code, rid, os.path.join(cdir, f"{b}.ref_{rid}.pkl")) for b in bids for rid, code in progs.items()
             if not (exclude_own and rid == b) and resolvable(bizs[b], meta.get(rid, {}))]
    cands = defaultdict(list); t0 = time.time()
    for k, r in pool.imap_unordered(run_mapping, tasks):
        if r.get("ok"): cands[k[0]].append(("ref:" + k[1], r))
    print(f"[map] transfer: {len(tasks)} reference programs tried, {sum(len(v) for v in cands.values())} fit, "
          f"{sum(1 for b in bids if cands[b])}/{len(bids)} businesses mapped ({time.time() - t0:.0f}s)", flush=True)
    need = [b for b in bids if a.map_llm or not cands[b]]
    if need and engine is not None:
        rb = rulebook_extract(rulebook)
        items = {(b, s): [{"role": "system", "content": SYSTEM_MAP},
                          {"role": "user", "content": user_msg_map(rb, bizs[b].readme, CARDS[b], map_examples(bizs[b], b, exclude_own))}]
                 for b in need for s in range(a.kc)}
        catt = generate_and_run(engine, pool, items, {k for k in items if k[1] == 0},
                                lambda k, code: (k, k[0], code, None, os.path.join(cdir, f"{k[0]}.llm{k[1]}.pkl")),
                                run_mapping, a.repairs, a.compile_tokens, a.temp, "map")
        for (b, sidx), atts in catt.items():
            r = last_ok(atts)
            if r: cands[b].append((f"llm:{sidx}", r))
        for b in need: clog.write(json.dumps({"business_id": b, "llm_attempts": {s: catt[(b, s)] for s in range(a.kc)}}, ensure_ascii=False, default=str) + "\n")
    canon = {}
    for b in bids:
        cs = cands[b]
        if not cs: print(f"  {b}: no canonical tables", flush=True); continue
        best, scores = agreement([r["sig"] for _, r in cs])
        order = sorted(range(len(cs)), key=lambda i: (-cs[i][1]["coverage"], -scores[i], i))
        src, r = cs[order[0]]; canon[b] = r
        print(f"  {b}: {len(cs)} candidates, chose {src} coverage {r['coverage']:.0f}", flush=True)
        clog.write(json.dumps({"business_id": b, "chosen": src, "candidates": [[x, y["coverage"]] for x, y in cs]}, ensure_ascii=False) + "\n")
    clog.flush()
    return canon


def run_qparse(task):
    """task = (key, bid, as_of, question, canon_path, timeout) -> (key, result): the rule-based template program."""
    key, bid, as_of, question, canon_path, timeout = task
    t0 = time.time(); r = {"ok": False, "error": "unparsed"}
    try:
        can = _lru(_CAN, canon_path, lambda: pickle.load(open(canon_path, "rb")))
        shop = dm.Shop(can["C"], as_of, {"data_end": can.get("data_end")})
        pr = qparse.parse(question, as_of, shop)
        if pr:
            tpl, code = pr
            _, r = run_code((key, bid, as_of, code, timeout, canon_path)); r.update(code=code, tpl=tpl)
    except BaseException as e: r = _err(e)
    r["t"] = time.time() - t0
    return key, r


def answer_stage_v2(engine, pool, a, rulebook, bizs, qs, canon, k, repairs):
    """K toolkit programs per question from the LLM (few-shot: similar verified train programs) + the rule-based program."""
    ex = fewshot.AnswerExamples(); exclude_own = os.path.isfile(os.path.join(a.split, "answers.jsonl"))
    qmap = {q["id"]: q for q in qs}; items = {}; greedy = set(); path_of = {}
    for q in qs:
        b = q["business_id"]
        if b in canon:
            exs = ex.pick(q["question"], k=a.shots, exclude_business=b if exclude_own else None)
            msgs = [{"role": "system", "content": SYSTEM_LIB},
                    {"role": "user", "content": user_msg_lib(rulebook, bizs[b].readme, canon[b]["card"], fewshot.render_examples(exs), q)}]
        else:
            msgs = [{"role": "system", "content": SYSTEM_RAW}, {"role": "user", "content": user_msg(rulebook, bizs[b].readme, CARDS[b], q)}]
        for sidx in range(k):
            items[(q["id"], sidx)] = msgs; path_of[(q["id"], sidx)] = "L" if b in canon else "B"
        greedy.add((q["id"], 0))
    def make_task(kk, code):
        q = qmap[kk[0]]; b = q["business_id"]
        return (kk, b, q["as_of_date"], code, a.timeout, canon[b]["path"] if b in canon else None)
    qatt = generate_and_run(engine, pool, items, greedy, make_task, run_code, repairs, a.max_tokens, a.temp, "answer",
                            reflect=not a.no_reflect) if (engine is not None and k) else defaultdict(list)
    rule = {}
    if a.qw > 0:
        tasks = [((q["id"], "Q"), q["business_id"], q["as_of_date"], q["question"], canon[q["business_id"]]["path"], a.timeout)
                 for q in qs if q["business_id"] in canon]
        for kk, r in pool.imap_unordered(run_qparse, tasks, chunksize=4): rule[kk[0]] = r
        print(f"[rule] {sum(1 for r in rule.values() if r.get('ok'))}/{len(tasks)} questions answered by templates", flush=True)
    by_q = defaultdict(list)
    for kk in items: by_q[kk[0]].append(kk)
    return qatt, path_of, by_q, rule


def answer_stage(engine, pool, a, rulebook, bizs, qs, canon, paths, repairs):
    qmap = {q["id"]: q for q in qs}; items = {}; path_of = {}; greedy = set()
    for q in qs:
        b = q["business_id"]
        for s in range(len(paths)):
            p = paths[s] if (paths[s] == "B" or b in canon) else "B"
            if p == "A":
                msgs = [{"role": "system", "content": SYSTEM_CANON},
                        {"role": "user", "content": user_msg(rulebook, bizs[b].readme, CARDS[b], q, canon[b]["card"], canon[b]["notes"])}]
            else:
                msgs = [{"role": "system", "content": SYSTEM_RAW}, {"role": "user", "content": user_msg(rulebook, bizs[b].readme, CARDS[b], q)}]
            key = (q["id"], s); items[key] = msgs; path_of[key] = p
            if paths.index(paths[s]) == s: greedy.add(key)              # first sample of each path decodes greedily
    def make_task(k, code):
        q = qmap[k[0]]
        return (k, q["business_id"], q["as_of_date"], code, a.timeout, canon[q["business_id"]]["path"] if path_of[k] == "A" else None)
    qatt = generate_and_run(engine, pool, items, greedy, make_task, run_code, repairs, a.max_tokens, a.temp, "answer", reflect=not a.no_reflect)
    by_q = defaultdict(list)
    for k in items: by_q[k[0]].append(k)
    return qatt, path_of, by_q


def canon_of(x): return str(canon(x))


def score_if_possible(a, qs):
    ans_path = os.path.join(a.split, "answers.jsonl")
    if not os.path.isfile(ans_path): return
    import score
    ids = {q["id"] for q in qs}
    refs = {k: v for k, v in score.load(ans_path).items() if k in ids}
    res, _ = score.score(refs, score.load(os.path.join(a.out, "submission.jsonl")), by="type")
    print("SCORE", json.dumps({k: res.get(k, 0) for k in ("PA", "n", "correct", "wrong", "abstain")}), flush=True)
    json.dump(res, open(os.path.join(a.out, "score.json"), "w"), ensure_ascii=False, indent=1)


def run_dp(a):
    """One agent process per GPU (CUDA_VISIBLE_DEVICES=i), then merge submissions / logs and score."""
    import subprocess, torch
    n = min(a.dp, max(1, torch.cuda.device_count())); t0 = time.time()
    argv = list(sys.argv[1:]); i = argv.index("--dp"); del argv[i:i + 2]
    cache = a.cache or os.path.join(a.out, "cache"); os.makedirs(a.out, exist_ok=True)
    nw = str(max(1, (a.workers or os.cpu_count() or 4) // n))
    procs = []
    for g in range(n):
        args = [sys.executable, os.path.abspath(__file__), a.split, os.path.join(a.out, f"shard{g}")] + argv[2:] + \
               ["--shard", f"{g}/{n}", "--cache", cache, "--tp", "1", "--workers", nw]
        procs.append(subprocess.Popen(args, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g)),
                                      stdout=open(os.path.join(a.out, f"shard{g}.log"), "w"), stderr=subprocess.STDOUT))
    def merge(files=("submission.jsonl",)):                           # atomic merge of the shards' (partial) outputs
        for f in files:
            tmp = os.path.join(a.out, f + ".tmp")
            with open(tmp, "w", encoding="utf-8") as w:
                for g in range(n):
                    p = os.path.join(a.out, f"shard{g}", f)
                    if os.path.isfile(p):
                        txt = open(p, encoding="utf-8").read(); w.write(txt[:txt.rfind("\n") + 1])   # complete lines only
            os.replace(tmp, os.path.join(a.out, f))
    while any(p.poll() is None for p in procs):                        # keep the merged submission current (Rule 8: time-outs)
        merge(); time.sleep(30)
    rcs = [p.wait() for p in procs]
    for g in range(n): print(f"--- shard{g} rc={rcs[g]}\n" + "".join(open(os.path.join(a.out, f"shard{g}.log")).readlines()[-12:]), flush=True)
    merge(("submission.jsonl", "logs.jsonl", "compile_logs.jsonl"))
    print(f"DP DONE {n} shards in {time.time() - t0:.0f}s", flush=True)
    qs = [json.loads(l) for l in open(os.path.join(a.split, "questions.jsonl"), encoding="utf-8") if l.strip()]
    score_if_possible(a, select_subset(qs, a.subset) if a.subset else qs)


if __name__ == "__main__":
    main()
