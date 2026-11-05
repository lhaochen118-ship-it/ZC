#!/usr/bin/env python3
"""Offline evaluation of the deterministic stack (no LLM): mapping (automap or the train reference programs) + rule
programs (qparse) + toolkit, scored against a reference file (train answers.jsonl, or a public reference file).

Usage: eval_split.py <split_dir> <cache_dir> <ref.jsonl> [--map auto|ref] [--bids b1,b2] [--out results.jsonl]
                     [--clean question_cleaning/train_questions_clean.jsonl] [--workers 4] [--show N]
Prints PA per business / template / type, mapping errors, and writes one JSON line per question (--out) for analysis.
The judge accepts other spellings of a name that the shop's own data uses (the organisers accept every spelling in the
shop's files and the code), so that a reference file without aliases is scored like the portal."""
import os, re, sys, json, time, pickle, argparse, traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import pandas as pd, numpy as np
import dm, qparse, score, ingest, automap

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "refs", "compile")


def ref_build(biz, bid):
    g = {"T": {k: v.copy() for k, v in biz.tables.items()}, "RAW": biz.raw, "pd": pd, "np": np, "re": re,
         "parse_num": ingest.parse_num, "parse_date": ingest.parse_date, "README": biz.readme}
    g.update({k: getattr(dm, k) for k in ("finalize", "map_status", "canon_status", "key", "is_walkin", "money", "round_half_up",
                                          "readme_conversions", "conversions_table", "frame")})
    g["tbl"] = lambda *c, **kw: dm.find_table(g["T"], *c, **kw)
    exec(compile(open(os.path.join(REF, bid + ".py"), encoding="utf-8").read(), bid, "exec"), g)
    return g["C"], g.get("NOTES", "")


def alias_forms(C):
    """key -> set of keys of the same entity (every spelling of an item / customer in the canonical data + codes)."""
    groups = defaultdict(set)
    L, O, it = C.get("lines"), C.get("orders"), C.get("items")
    def add(frame, keycol, cols):
        if frame is None or keycol not in frame: return
        for c in cols:
            if c in frame:
                for k, v in zip(frame[keycol].astype(str), frame[c]):
                    if isinstance(v, str): groups[("i", k)].add(score.norm(v))
    add(L, "item_code", ("item_name", "item_name_raw")); add(it, "item_code", ("item_name",))
    for frame in (L, it):
        if frame is not None and "item_code" in frame:
            for k in frame["item_code"].dropna().astype(str).unique(): groups[("i", k)].add(score.norm(k))
    if O is not None and "customer_name" in O:
        raw = O["customer_name_raw"] if "customer_name_raw" in O else O["customer_name"]
        for k, v in zip(O["customer_name"].astype(str), raw):
            if isinstance(v, str): groups[("c", k)].add(score.norm(v)); groups[("c", k)].add(score.norm(k))
    out = defaultdict(set)
    for g in groups.values():
        for k in g: out[k] |= g
    return out


def judge(pred, ref, forms):
    v = score.judge(pred, ref)
    if v != "wrong" or pred.get("type") != ref.get("type"): return v
    def same(p, r):
        if not isinstance(p, str) or not isinstance(r, str): return False
        pn, rn = score.norm(p), score.norm(r)
        if pn == rn or pn in forms.get(rn, ()) or rn in forms.get(pn, ()): return True
        return score.norm(automap.strip_phone(p)) == rn
    t = ref["type"]; a, r = pred.get("answer"), ref.get("answer")
    if t == "string": return "correct" if same(a, r) else "wrong"
    if t == "list" and isinstance(a, list) and isinstance(r, list) and len(a) == len(r):
        ok = all((score.elem_eq(x, y) or same(x, y)) for x, y in zip(a, r))
        return "correct" if ok else "wrong"
    if t == "table" and isinstance(a, dict) and isinstance(r, dict):
        try:
            rows = [[automap.strip_phone(c) if isinstance(c, str) else c for c in row] for row in a["rows"]]
            if score.table_eq({"columns": a["columns"], "rows": rows}, r): return "correct"
            # names of the reference rows -> one of the predicted spellings
            fixed = []
            for row in rows:
                nm = row[0]; hit = next((rr[0] for rr in r["rows"] if same(nm, rr[0])), nm)
                fixed.append([hit] + list(row[1:]))
            return "correct" if score.table_eq({"columns": a["columns"], "rows": fixed}, r) else "wrong"
        except Exception: return "wrong"
    return v


def eval_biz(args):
    split, cache, bid, mode, qs, refs, clean = args
    t0 = time.time(); out = []
    biz = pickle.load(open(os.path.join(cache, bid + ".pkl"), "rb"))
    try:
        if mode == "ref": C, notes = ref_build(biz, bid); rep = {}
        else: C, notes, rep = automap.build(biz)
        errs = dm.check(C)
    except Exception as e:
        tb = traceback.format_exc().splitlines()[-3:]
        return bid, [{"id": q["id"], "business_id": bid, "question": q["question"], "as_of": q["as_of_date"], "ref": refs[q["id"]],
                      "verdict": "abstain", "tpl": "NOMAP", "err": f"{type(e).__name__}: {e}"} for q in qs], \
            {"error": f"{type(e).__name__}: {e}", "tb": tb, "t": time.time() - t0}
    dm.set_calendar(pd.read_csv(os.path.join(split, "calendar.csv")))
    meta = {"data_end": automap.parse_readme(biz.readme)["data_end"]}
    if clean == "QNORM":                                               # the runtime cleaning (pipeline: --qclean 1)
        import qnorm
        nz = qnorm.Normalizer.for_split(split, exclude_business=bid, extra_texts=qnorm.Normalizer.entity_texts(biz))
        clean = {}
        for q in qs:
            try: clean[q["id"]] = nz.clean(q["question"], q["as_of_date"])["clean"]
            except Exception: pass
    forms = alias_forms(C)
    for q in qs:
        shop = dm.Shop(C, q["as_of_date"], meta)
        text = clean.get(q["id"], q["question"])
        try: pr = qparse.parse(text, q["as_of_date"], shop)
        except Exception as e: pr = None
        rec = {"id": q["id"], "business_id": bid, "question": q["question"], "as_of": q["as_of_date"], "ref": refs[q["id"]], "clean": text}
        if not pr: rec.update(verdict="abstain", tpl="UNPARSED"); out.append(rec); continue
        tpl, code = pr
        try:
            import agent
            ans = agent.fmt_answer(dm.run_program(code, shop, q["as_of_date"]))
        except Exception as e: ans = {"type": "error", "answer": f"{type(e).__name__}: {e}"[:200]}
        v = judge(ans, refs[q["id"]], forms) if ans.get("type") not in ("error", None) else "wrong_err"
        rec.update(verdict=v, tpl=tpl, code=code, pred=ans); out.append(rec)
    return bid, out, {"orders": len(C["orders"]), "lines": len(C["lines"]) if C.get("lines") is not None else 0, "check": errs,
                      "tables": sorted(C), "t": round(time.time() - t0, 1), "kinds": rep.get("kinds"), "unused": rep.get("unused")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("cache"); ap.add_argument("ref")
    ap.add_argument("--map", default="auto"); ap.add_argument("--bids", default=""); ap.add_argument("--out", default="")
    ap.add_argument("--clean", default=""); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    qs = [json.loads(l) for l in open(os.path.join(a.split, "questions.jsonl"), encoding="utf-8")]
    refs = score.load(a.ref)
    clean = "QNORM" if a.clean == "qnorm" else {r["id"]: r["clean"] for r in map(json.loads, open(a.clean, encoding="utf-8"))} if a.clean else {}
    by = defaultdict(list)
    for q in qs: by[q["business_id"]].append(q)
    bids = [b for b in sorted(by) if not a.bids or b in a.bids.split(",")]
    tasks = [(a.split, a.cache, b, a.map, by[b], {q["id"]: refs[q["id"]] for q in by[b]}, clean if clean == "QNORM" else {q["id"]: clean[q["id"]] for q in by[b] if q["id"] in clean}) for b in bids]
    res = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for bid, out, info in ex.map(eval_biz, tasks): res[bid] = (out, info)
    tot = Counter(); tpl = defaultdict(Counter); typ = defaultdict(Counter)
    f = open(a.out, "w", encoding="utf-8") if a.out else None
    for bid in bids:
        out, info = res[bid]; c = Counter()
        for r in out:
            v = r["verdict"]; v = "wrong" if v == "wrong_err" else v
            c[v] += 1; tot[v] += 1; tpl[r["tpl"]][v] += 1; typ[r["ref"]["type"]][v] += 1
            if f: f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        n = sum(c.values())
        print(f"{bid}: PA {(c['correct'] - c['wrong']) / max(n, 1):+.3f} ok {c['correct']} wrong {c['wrong']} abst {c['abstain']} | "
              f"{ {k: v for k, v in info.items() if k in ('orders', 'check', 'error', 't')} }", flush=True)
        if a.show:
            for r in [r for r in out if r["verdict"].startswith("wrong")][:a.show]:
                print(f"     [{r['tpl']}] {r['question'][:90]} | ref {json.dumps(r['ref'].get('answer'), ensure_ascii=False)[:60]} | got {json.dumps(r.get('pred', {}).get('answer'), ensure_ascii=False)[:60]}")
    n = sum(tot.values())
    print(f"TOTAL PA {(tot['correct'] - tot['wrong']) / max(n, 1):+.4f} correct {tot['correct']} wrong {tot['wrong']} abstain {tot['abstain']} (n={n})")
    print("by type:", {t: f"{c['correct']}/{c['wrong']}/{c['abstain']}" for t, c in sorted(typ.items(), key=lambda x: -sum(x[1].values()))})
    worst = sorted(tpl.items(), key=lambda x: -x[1]["wrong"])[:12]
    print("templates with most wrong:", {t: dict(c) for t, c in worst if c["wrong"]})


if __name__ == "__main__":
    main()
