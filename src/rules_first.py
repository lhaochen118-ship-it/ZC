#!/usr/bin/env python3
"""Rule-first answering on CPU, no model: automap (raw tables -> canonical C, from the README and the values) + qnorm
(question cleaning) + qparse (rule programs over the dm / industry toolkit). Train 3000 / 3000 and public 1000 / 1000
(tests/eval_split.py). Questions the rules do not read are left to the model (docker/predict.sh runs agent.py on them only).

Usage: rules_first.py <split> <out_dir> [--cache DIR] [--workers N] [--log-dir DIR] [--timeout S]
Writes, in <out_dir>:
  submission.jsonl  one line per question the rules answer ({"id", "type", "answer"}), appended business by business
  pending.jsonl     the original lines of questions.jsonl left to the model (not read, program failed, mapping failed)
  rules_log.jsonl   per question: template, program, cleaned text, answer or error, seconds
and, with --log-dir, one JSON log per answered question (Docker log requirement)."""
import os, sys, json, time, signal, pickle, argparse, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ZC_ADJ_ROWS", "move")
import pandas as pd
import dm, ingest, automap, qparse, qnorm


REFS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refs")
_STOP = set("""va hoac cua cho la mot cac nhung trong ky theo khi thi bang ve tu den de duoc co khong""".split())


def _rule_rows(text):
    """{term key: definition key} of a rulebook's table rows ('| Khách sộp nhất | khách có ... |')."""
    out = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")] if line.strip().startswith("|") else []
        if len(cells) < 2 or set(cells[0]) <= set("-: ") or cells[0].lower() in ("luật", "luat", "khái niệm", "khai niem"): continue
        out[" ".join(qparse.dm.key(cells[0]).split())] = " ".join(qparse.dm.key(" ".join(cells[1:])).split())
    return out


def changed_terms(split):
    """terms the split's rulebook defines that the rules were not written for: a row whose term is in neither the train
    nor the public rulebook (public added 'Khách sộp nhất', 'Giờ cao điểm', industry sections ... this way). A question
    naming such a term goes to the model, which reads the rulebook. Rows only reworded (train -> public clarified 'Đơn
    thuộc kỳ nào', 'Đến hết ngày X') keep the rules: their words are in most questions. Each term -> its sets of
    content words (one per comma variant: 'Nhân viên, thợ chăm chỉ nhất' -> {nhan vien cham chi nhat}, {tho cham chi nhat})."""
    try: rows = _rule_rows(open(os.path.join(split, "rulebook.md"), encoding="utf-8").read())
    except OSError: return []
    known = set()
    for f in ("train_rulebook.md", "public_rulebook.md"):
        try: known |= set(_rule_rows(open(os.path.join(REFS, f), encoding="utf-8").read()))
        except OSError: pass
    out = []
    for term in rows:
        if term in known: continue
        parts = [p.strip() for p in term.split(",")]
        tail = parts[-1].split()
        variants = [set(parts[-1].split())] + [set(p.split()) | set(tail[len(p.split()):]) for p in parts[:-1]]
        variants = [v - _STOP for v in variants if len(v - _STOP) >= 2]
        if variants: out.append((term, variants))
    return out


def names_changed_term(question, terms):
    """the first changed term whose content words all occur in the question (accent-free), else None."""
    w = set(qparse.dm.key(question).split())
    for term, variants in terms:
        if any(v <= w for v in variants): return term
    return None


class _Timeout(Exception): pass


def _alarm(_s, _f): raise _Timeout()


def answer_business(args):
    """(bid, [record per question]) for one business; a record carries 'final' only when the rules answered."""
    split, cache, bid, qs, timeout = args[:5]
    terms = args[5] if len(args) > 5 else []
    t0 = time.time(); out = []
    try:
        biz = pickle.load(open(os.path.join(cache, bid + ".pkl"), "rb"))
        C, notes, rep = automap.build(biz)
    except Exception as e:
        err = f"mapping: {type(e).__name__}: {e}"[:300]
        return bid, [{"id": q["id"], "error": err, "tb": traceback.format_exc().splitlines()[-3:]} for q in qs], round(time.time() - t0, 1)
    dm.set_calendar(pd.read_csv(os.path.join(split, "calendar.csv")))
    meta = {"data_end": automap.parse_readme(biz.readme)["data_end"]}
    try: nz = qnorm.Normalizer.for_split(split, exclude_business=bid, extra_texts=qnorm.Normalizer.entity_texts(biz))
    except Exception: nz = None
    import agent                                                         # fmt_answer: the answer format of the task
    signal.signal(signal.SIGALRM, _alarm)
    for q in qs:
        t1 = time.time(); rec = {"id": q["id"]}
        try:
            text = nz.clean(q["question"], q["as_of_date"])["clean"] if nz is not None else q["question"]
        except Exception: text = q["question"]
        rec["clean"] = text
        ct = names_changed_term(text, terms) or names_changed_term(q["question"], terms) if terms else None
        if ct:                                                           # a term this split's rulebook (re)defines: the model
            rec["tpl"] = "CHANGED_TERM"; rec["error"] = f"rulebook term changed: {ct}"; rec["s"] = 0.0; out.append(rec); continue
        try:
            signal.alarm(timeout)
            shop = dm.Shop(C, q["as_of_date"], meta)
            pr = qparse.parse(text, q["as_of_date"], shop)
            if pr:
                rec["tpl"], rec["code"] = pr
                rec["final"] = agent.fmt_answer(dm.run_program(pr[1], shop, q["as_of_date"]))
            else: rec["tpl"] = "UNPARSED"
        except _Timeout: rec["error"] = f"timeout {timeout}s"
        except Exception as e: rec["error"] = f"{type(e).__name__}: {e}"[:300]
        finally: signal.alarm(0)
        rec["s"] = round(time.time() - t1, 3); out.append(rec)
    return bid, out, round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("out")
    ap.add_argument("--cache", default=""); ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--log-dir", default=""); ap.add_argument("--timeout", type=int, default=120, help="seconds per question")
    a = ap.parse_args()
    t0 = time.time(); os.makedirs(a.out, exist_ok=True)
    cache = a.cache or os.path.join(a.out, "cache"); os.makedirs(cache, exist_ok=True)
    lines = [l for l in open(os.path.join(a.split, "questions.jsonl"), encoding="utf-8") if l.strip()]
    qs = [json.loads(l) for l in lines]; raw = {q["id"]: l.rstrip("\n") for q, l in zip(qs, lines)}
    by = {}
    for q in qs: by.setdefault(q["business_id"], []).append(q)
    nw = a.workers or max(1, os.cpu_count() or 4)
    todo = [b for b in by if not os.path.isfile(os.path.join(cache, b + ".pkl"))]
    if todo:
        with ProcessPoolExecutor(min(nw, len(todo))) as ex: list(ex.map(ingest._cache_one, [(a.split, cache, b) for b in todo]))
    print(f"rules: ingest {len(by)} businesses {time.time() - t0:.0f}s", flush=True)
    if a.log_dir: os.makedirs(a.log_dir, exist_ok=True)
    sub = open(os.path.join(a.out, "submission.jsonl"), "w", encoding="utf-8")
    pend = open(os.path.join(a.out, "pending.jsonl"), "w", encoding="utf-8")
    log = open(os.path.join(a.out, "rules_log.jsonl"), "w", encoding="utf-8")
    qmap = {q["id"]: q for q in qs}; n_ok = n_pend = 0
    order = sorted(by, key=lambda b: -len(by[b]))                     # big businesses first (balanced pool)
    terms = changed_terms(a.split)
    if terms: print(f"rules: {len(terms)} rulebook term(s) new or redefined in this split -> their questions go to the model: "
                    + "; ".join(t for t, _ in terms[:12]), flush=True)
    with ProcessPoolExecutor(min(nw, len(order))) as ex:
        futs = [ex.submit(answer_business, (a.split, cache, b, by[b], a.timeout, terms)) for b in order]
        for fu in as_completed(futs):
            bid, recs, secs = fu.result(); ok = 0
            for r in recs:
                q = qmap[r["id"]]
                log.write(json.dumps({"business_id": bid, "question": q["question"], "as_of_date": q["as_of_date"], **r}, ensure_ascii=False, default=str) + "\n")
                if "final" in r:
                    sub.write(json.dumps({"id": r["id"], **r["final"]}, ensure_ascii=False) + "\n"); ok += 1
                    if a.log_dir:
                        json.dump({"id": r["id"], "question": q["question"], "business_id": bid, "as_of_date": q["as_of_date"], "final": r["final"],
                                   "source": "rules", "template": r.get("tpl"), "program": r.get("code"), "clean": r.get("clean"),
                                   "elapsed_s": r.get("s")}, open(os.path.join(a.log_dir, f"{r['id']}.json"), "w", encoding="utf-8"),
                                  ensure_ascii=False, default=str, indent=1)
                else: pend.write(raw[r["id"]] + "\n")
            n_ok += ok; n_pend += len(recs) - ok
            sub.flush(); pend.flush(); log.flush()
            print(f"  {bid}: {ok}/{len(recs)} answered by the rules ({secs:.0f}s)", flush=True)
    sub.close(); pend.close(); log.close()
    print(f"rules: {n_ok} answered, {n_pend} left to the model, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
