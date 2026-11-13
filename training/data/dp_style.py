#!/usr/bin/env python3
"""Style augmentation of training questions (plan 3, docs/16 §3: DP, DL), labels unchanged (the original checked program):
  period  the period the PROGRAM uses (training/data/spec_from_program.spec_of: month(2025, 4), quarter(...), this_month(TODAY) ...)
          is found in the question in one of its usual spellings and rewritten in another Vietnamese notation (T4/2025, Th4/25,
          4-2025, 04/2025, tháng tư năm 2025, tháng 4 năm ngoái, Q2/25, quý II, th này, tuần rồi, hqua ...). The training questions
          never use the short notations (docs/15 §6: 0 % of train vs 25.9 % of public), so the model never learned them.
  noise   chat noise learned from ViLexNorm (EACL 2024; train + dev splits, CC BY-NC-SA 4.0, research use): word -> noisy forms
          with their counts ('không' -> ko / k / kh / khum, 'được' -> đc / dc, 'nhiều' -> nhìu ...), sampled by count.
  both    period + noise.
The renderers only build TRAINING data: nothing here runs at inference (the pure flow has no question-reading rule), the model
has to learn the notations. Output: pure records (src/hybrid.py --pure 1 prompt: no hints, examples retrieved by the question
text, never the same shop / question), label optionally prefixed with its <spec> (--spec).
Usage: dp_style.py <train_split> <ingest_cache> <canon_dir> <in.jsonl>[,...] <out.jsonl> --vilexnorm DIR [--kinds period,noise,both]
       [--per 1] [--seed 20261113] [--mode full] [--k 3] [--spec] [--limit N]"""
import os, sys, re, csv, json, time, random, argparse, difflib, unicodedata, zlib
from collections import defaultdict, Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from spec_from_program import spec_of, spec_block

WORD = {1: "một", 2: "hai", 3: "ba", 4: "tư", 5: "năm", 6: "sáu", 7: "bảy", 8: "tám", 9: "chín", 10: "mười", 11: "mười một", 12: "mười hai"}
ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV"}


def strip_acc(s):
    s = unicodedata.normalize("NFD", s).replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFC", "".join(ch for ch in s if unicodedata.category(ch) != "Mn"))


def has_accents(s): return strip_acc(s) != s


# ---------------------------------------------------------------- periods: spellings to find, notations to write
def _year_re(y): return rf"(?:{y}|{y % 100:02d})"


def find_res(expr, today):
    """regexes of the usual spellings of a period expression in a question (accent-free matching is done by the caller)."""
    m = re.fullmatch(r"(\w+)\((.*)\)", expr.replace(" ", ""))
    if not m: return []
    f, args = m.group(1), m.group(2).split(",")
    num = [int(x) for x in args if x.isdigit()]
    if f == "month" and len(num) == 2:
        y, mo = num
        return [rf"(?:thang|thg|th|t)\s*0?{mo}\s*(?:/|-|\.|\s+nam\s+)\s*{_year_re(y)}(?!\d)", rf"(?<![\d/]){mo:d}\s*[/-]\s*{y}(?!\d)",
                rf"(?<![\d/])0{mo}\s*/\s*{y}(?!\d)" if mo < 10 else r"(?!x)x"]
    if f == "quarter" and len(num) == 2:
        y, q = num
        return [rf"(?:quy|q)\s*(?:0?{q}|{ROMAN[q]})\s*(?:/|-|\s+nam\s+)\s*{_year_re(y)}(?!\d)"]
    if f == "year" and len(num) == 1:
        return [rf"nam\s*{num[0]}(?!\d)"]
    if f == "month_noyear" and num:
        return [rf"(?:thang|thg|th)\s*0?{num[0]}(?![\d/])"]
    if f == "day" and len(num) == 3:
        y, mo, d = num
        return [rf"(?<![\d/])0?{d}\s*/\s*0?{mo}\s*/\s*{_year_re(y)}(?!\d)"]
    fixed = {"this_month": r"thang\s*(?:nay|ni)", "prev_month": r"thang\s*(?:truoc|roi|vua roi|vua qua)", "this_week": r"tuan\s*(?:nay|ni)",
             "last_week": r"tuan\s*(?:truoc|roi|vua roi|vua qua)", "yesterday": r"hom\s*qua|hqua|bua\s*qua", "today_p": r"hom\s*nay|hnay|bua\s*nay"}
    return [fixed[f]] if f in fixed else []


def notations(expr, today):
    """other ways to write the same period (absolute, short, relative to the question's day when exact)."""
    m = re.fullmatch(r"(\w+)\((.*)\)", expr.replace(" ", ""))
    if not m: return []
    f, args = m.group(1), m.group(2).split(",")
    num = [int(x) for x in args if x.isdigit()]
    ty = today.year if today else None
    if f == "month" and len(num) == 2:
        y, mo = num; yy = f"{y % 100:02d}"
        out = [f"T{mo}/{y}", f"Th{mo}/{y}", f"th {mo}/{y}", f"T{mo}/{yy}", f"Th{mo}/{yy}", f"{mo}/{yy}", f"{mo}-{y}", f"tháng {mo}/{y}",
               f"thg {mo}/{y}", f"tháng {mo} năm {y}", f"tháng {WORD[mo]} năm {y}", f"{mo:02d}/{y}", f"T{mo:02d}.{y}", f"th{mo}/{yy}"]
        if ty and y == ty and (not today or mo <= today.month): out += [f"tháng {mo} năm nay", f"T{mo} năm nay", f"th {mo} năm nay"]
        if ty and y == ty - 1: out += [f"tháng {mo} năm ngoái", f"T{mo} năm ngoái", f"tháng {mo} năm rồi", f"th{mo} năm trước"]
        return out
    if f == "quarter" and len(num) == 2:
        y, q = num; yy = f"{y % 100:02d}"
        out = [f"Q{q}/{y}", f"Q{q}/{yy}", f"quý {q}/{y}", f"quý {ROMAN[q]}/{y}", f"quý {ROMAN[q]} năm {y}", f"Q{q}-{y}", f"quý {q} năm {y}",
               f"Q{q} {y}"]
        if ty and y == ty: out += [f"quý {q} năm nay", f"Q{q} năm nay"]
        if ty and y == ty - 1: out += [f"quý {q} năm ngoái", f"Q{q} năm ngoái"]
        return out
    if f == "year" and len(num) == 1:
        y = num[0]; out = [f"năm {y}", f"nam {y}", f"năm {y % 100:02d}"]
        if ty and y == ty: out += ["năm nay", "năm ni"]
        if ty and y == ty - 1: out += ["năm ngoái", "năm rồi", "năm trước", "năm qua"]
        return out
    if f == "month_noyear" and num:
        mo = num[0]; return [f"T{mo}", f"Th{mo}", f"th {mo}", f"tháng {WORD[mo]}", f"thg {mo}", f"tháng {mo:02d}"]
    if f == "day" and len(num) == 3:
        y, mo, d = num; yy = f"{y % 100:02d}"
        return [f"{d}/{mo}/{y}", f"{d:02d}/{mo:02d}/{y}", f"{d}/{mo}/{yy}", f"ngày {d} tháng {mo} năm {y}", f"{d}-{mo}-{y}", f"{d}.{mo}.{y}"]
    fixed = {"this_month": ["th này", "thg này", "tháng ni", "th nay", "tháng hiện tại", "T này"],
             "prev_month": ["th trước", "tháng vừa rồi", "tháng rồi", "thg trước", "tháng trc", "th trc"],
             "this_week": ["tuần ni", "tuần nay", "tuần hiện tại", "tuần này nè"],
             "last_week": ["tuần rồi", "tuần trc", "tuần vừa rồi", "tuần vừa qua"],
             "yesterday": ["hqua", "hôm wa", "bữa qua", "hum qua", "hôm qua"],
             "today_p": ["hnay", "bữa nay", "hôm ni", "hum nay", "hôm nay"]}
    return fixed.get(f, [])


def rewrite_periods(q, spec, today, rng):
    """q with each period of the program written in another notation; None when no period spelling is found."""
    per = spec.get("period"); per = [per] if isinstance(per, str) else (per or [])
    if not per: return None
    plain = strip_acc(q).lower(); spans = []
    for e in per:
        alts = [a for a in notations(e, today)]
        for rx in find_res(e, today):
            m = re.search(r"(?<!\w)(?:" + rx + r")(?!\w)", plain)
            if m and alts:
                old = q[m.start():m.end()]
                cand = [a for a in alts if strip_acc(a).lower() != strip_acc(old).lower()]
                if cand: spans.append((m.start(), m.end(), rng.choice(cand))); break
    if not spans: return None
    spans.sort(); out, last = [], 0
    for s, e, a in spans:
        if s < last: continue
        out.append(q[last:s]); out.append(a if has_accents(q) else strip_acc(a)); last = e
    out.append(q[last:])
    return "".join(out)


# ---------------------------------------------------------------- chat noise learned from ViLexNorm
def noise_table(d):
    """canonical word / phrase -> Counter of noisy forms, from ViLexNorm train + dev (aligned word by word)."""
    pairs = defaultdict(Counter)
    for s in ("train", "dev"):
        p = os.path.join(d, s + ".csv")
        if not os.path.isfile(p): continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            a, b = r["original"].lower().split(), r["normalized"].lower().split()
            for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=b, b=a, autojunk=False).get_opcodes():
                if op != "replace": continue
                if i2 - i1 == j2 - j1:
                    for x, y in zip(b[i1:i2], a[j1:j2]):
                        if x != y and x.isalpha() and y.replace(".", "").isalpha() and len(y) <= 12: pairs[x][y] += 1
                elif j2 - j1 == 1 and 2 <= i2 - i1 <= 3:
                    x, y = " ".join(b[i1:i2]), a[j1]
                    if y.isalpha(): pairs[x][y] += 1
    return {k: Counter({y: n for y, n in c.items() if n >= 2}) for k, c in pairs.items() if sum(c.values()) >= 3}


def literals(program):
    """lowercase string values of a program (item / group / customer names ...): words inside them are never rewritten."""
    return [x.lower() for t in re.findall(r'"([^"]{2,})"|\'([^\']{2,})\'', program or "") for x in t if x]


def protected(low, lits):
    return any(low in l or any(w in l for w in low.split() if len(w) > 2) for l in lits)


def add_noise(q, table, rng, p=0.4, cap=4, lits=()):
    """q with 1-cap words / phrases replaced by noisy forms (never names: capitalised words, words of the program's values,
    never numbers)."""
    toks = q.split(" "); idx = []
    i = 0
    while i < len(toks):
        for n in (3, 2, 1):
            if i + n > len(toks): continue
            ph = " ".join(toks[i:i + n]); low = ph.lower().strip(",.?!")
            if low in table and table[low] and not protected(low, lits) and not any(t[:1].isupper() or re.search(r"\d", t) for t in toks[i:i + n]):
                idx.append((i, n, low)); i += n; break
        else: i += 1
    if not idx: return None
    rng.shuffle(idx); pick = [x for x in idx if rng.random() < p][:cap] or idx[:1]
    for i, n, low in sorted(pick, reverse=True):
        forms, w = zip(*table[low].items())
        new = rng.choices(forms, weights=w)[0]
        tail = toks[i + n - 1][len(toks[i + n - 1].rstrip(",.?!")):]
        toks[i:i + n] = [new + tail]
    return " ".join(toks)


def translit_table(path):
    """Vietnamese word -> spoken / written loanword forms, from the model's own list (src/lrl_gen.py --mode translit): 'đơn hàng' ->
    ['order', <how it is typed as it sounds>]. Forms of 1-2 characters and identical pairs are dropped."""
    tab = defaultdict(list)
    if not path or not os.path.isfile(path): return {}
    for l in open(path, encoding="utf-8"):
        c = json.loads(l); t = c.get("translit") or {}
        vi = str(t.get("vietnamese") or "").strip().lower()
        if not vi or len(vi) < 2: continue
        for f in [t.get("english")] + list(t.get("phonetic") or []):
            f = str(f or "").strip()
            if len(f) >= 3 and f.lower() != vi and f not in tab[vi]: tab[vi].append(f)
    return dict(tab)


def add_translit(q, tab, rng, lits=()):
    """q with one Vietnamese word (whole word, not in a name / a value of the program) replaced by a loanword form of it;
    None when none occurs (13/11: 120 / 3096 replacements fell inside item names before this check)."""
    low = q.lower(); hits = []
    for vi, forms in tab.items():
        if protected(vi, lits): continue
        for m in re.finditer(r"(?<!\w)" + re.escape(vi) + r"(?!\w)", low):
            if not q[m.start():m.start() + 1].isupper(): hits.append((m.start(), m.end(), forms))
    if not hits: return None
    s, e, forms = rng.choice(hits)
    return q[:s] + rng.choice(forms) + q[e:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("cache"); ap.add_argument("canon_dir"); ap.add_argument("inputs"); ap.add_argument("out")
    ap.add_argument("--vilexnorm", default=""); ap.add_argument("--kinds", default="period,noise,both"); ap.add_argument("--per", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261113); ap.add_argument("--mode", default="full"); ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--spec", action="store_true"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--all-shops", action="store_true")
    ap.add_argument("--translit", default="", help="cards_translit.jsonl of src/lrl_gen.py: kind 'translit' replaces a word by a loanword form")
    a = ap.parse_args()
    import retrieval
    import datetime as dt
    rng = random.Random(a.seed); t0 = time.time()
    table = noise_table(a.vilexnorm) if a.vilexnorm else {}
    ttab = translit_table(a.translit)
    kinds = [k for k in a.kinds.split(",") if k]
    bank = retrieval.Bank.load(include_holdout=a.all_shops)
    src = []
    for p in a.inputs.split(","): src += common.jl(p)
    if a.limit: src = src[:a.limit]
    rows, stat = [], Counter(); seen = set()
    for r in src:
        if not a.all_shops and r["business_id"] in common.TR_HO: continue
        sp = spec_of(r.get("program"))
        try: today = dt.date.fromisoformat(r["as_of_date"])
        except Exception: today = None
        for i in range(a.per):
            kind = kinds[(zlib.crc32(r["id"].encode()) + i) % len(kinds)] if len(kinds) > 1 else kinds[0]   # reproducible (not hash())
            q0 = r["question"]; nq = None
            if kind in ("period", "both"): nq = rewrite_periods(q0, sp, today, rng)
            lits = literals(r.get("program"))
            if kind == "translit": nq = add_translit(q0, ttab, rng, lits) if ttab else None
            if kind == "both" and nq is None: kind = "noise"
            if kind in ("noise", "both") and table: nq = add_noise(nq or q0, table, rng, lits=lits)
            if not nq or nq == q0 or (r["business_id"], nq) in seen: stat["unchanged_" + kind] += 1; continue
            seen.add((r["business_id"], nq)); stat[kind] += 1
            base = r["id"].split("~")[0]
            q = {"id": f"{r['id']}~{kind[0]}{i}", "business_id": r["business_id"], "question": nq, "as_of_date": r["as_of_date"]}
            c = common.ctx(a.split, a.cache, a.canon_dir, r["business_id"], a.mode)
            exs = bank.retrieve(nq, {}, k=a.k, exclude_bids={r["business_id"]}, exclude_ids={base})
            s = common.sample(c, q, r["program"], exs, None, (r.get("source") or "D?").rstrip("ph") + "S" + kind[0].upper(), r.get("answer"))
            if a.spec and sp: s["messages"][-1]["content"] = spec_block(sp) + s["messages"][-1]["content"]
            s["orig_question"] = q0; rows.append(s)
    common.write_jl(a.out, rows)
    print(f"style: {len(rows)} records from {len(src)} ({dict(stat)}) -> {a.out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
