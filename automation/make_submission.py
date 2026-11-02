#!/usr/bin/env python3
"""Validate a run's submission.jsonl against the task format and copy it to submissions/<name>.jsonl (the user uploads
it on the portal; nothing is submitted from here).

Checks (docs/00_TASK.md §2): one line per question of the split, ids exactly the split's, type in the allowed set,
number / percent numeric, date yyyy-mm-dd or yyyy-mm, boolean bool, list a list, table {columns, rows},
unanswerable with answer null, abstain allowed (scored 0).
Usage: make_submission.py <run_dir with submission.jsonl> <split_dir> <name>"""
import os, re, sys, json, collections

TYPES = {"number", "percent", "string", "date", "boolean", "list", "table", "unanswerable", "abstain"}


def problems(rec):
    t, a = rec.get("type"), rec.get("answer")
    if t not in TYPES: return f"unknown type {t!r}"
    if t in ("number", "percent") and not (isinstance(a, (int, float)) and not isinstance(a, bool)): return f"{t} answer not numeric: {a!r}"
    if t == "date" and not (isinstance(a, str) and re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", a)): return f"bad date {a!r}"
    if t == "boolean" and not isinstance(a, bool): return f"boolean answer {a!r}"
    if t == "list" and not isinstance(a, list): return f"list answer {type(a).__name__}"
    if t == "table" and not (isinstance(a, dict) and isinstance(a.get("columns"), list) and isinstance(a.get("rows"), list)): return "bad table"
    if t == "string" and not isinstance(a, str): return f"string answer {type(a).__name__}"
    if t == "unanswerable" and a is not None: return "unanswerable with an answer"
    return None


def main():
    run, split, name = sys.argv[1], sys.argv[2], sys.argv[3]
    want = [json.loads(l)["id"] for l in open(os.path.join(split, "questions.jsonl"), encoding="utf-8")]
    rows = [json.loads(l) for l in open(os.path.join(run, "submission.jsonl"), encoding="utf-8") if l.strip()]
    by = {}
    for r in rows: by[r["id"]] = r                                     # the last line of an id wins (merged shards)
    errs = [f"{r['id']}: {p}" for r in by.values() if (p := problems(r))]
    missing, extra = [q for q in want if q not in by], [q for q in by if q not in set(want)]
    print(f"{len(rows)} lines, {len(by)} ids | missing {len(missing)} | extra {len(extra)} | format problems {len(errs)}")
    for e in errs[:10]: print("  ", e)
    if missing or extra or errs:
        print("NOT WRITTEN: fix the problems first"); sys.exit(1)
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "submissions")
    os.makedirs(out_dir, exist_ok=True); out = os.path.join(out_dir, name + ".jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for q in want: f.write(json.dumps(by[q], ensure_ascii=False) + "\n")
    print("types:", dict(collections.Counter(by[q]["type"] for q in want)))
    print("written:", out)


if __name__ == "__main__":
    main()
