#!/usr/bin/env python3
"""Local re-implementation of the DataMate scorer (docs/00_TASK.md §4): per-question correct / wrong / abstain and
Penalized Accuracy PA = mean(s_i), s_i = +1 correct, 0 abstain (or missing), -1 wrong.

The reference file format is a GUESS until the organisers' training data is inspected (docs/00_TASK.md §3): one JSON
object per line with "id", "type", "answer"; a string answer may be one accepted form or a list of accepted forms
(also read from "accepted_forms" / "accepted"); list elements may themselves be lists of accepted forms.
Adjust `ref_forms` once the real format is known.

Usage: score.py <reference.jsonl> <submission.jsonl> [--by type|category] [--errors N]"""
import sys, json, re, math, unicodedata
from collections import Counter, defaultdict


def norm(s):
    """String rule: NFC, lowercase, non letter/digit/space -> space, collapse spaces. Diacritics kept."""
    s = unicodedata.normalize("NFC", str(s)).lower()
    s = "".join(c if (c.isalnum() or c.isspace()) else " " for c in s)
    return re.sub(r"\s+", " ", s).strip()


def to_num(x):
    if isinstance(x, bool): return None
    if isinstance(x, (int, float)): return float(x)
    try: return float(str(x).replace(",", "").strip())
    except ValueError: return None


def num_eq(a, b):
    a, b = to_num(a), to_num(b)
    return a is not None and b is not None and math.isfinite(a) and round(a, 2) == round(b, 2)


def forms(x):
    return [norm(f) for f in (x if isinstance(x, list) else [x])]


def elem_eq(pred, ref):
    """A list element / table cell: number rule if the reference is numeric, else string rule."""
    if not isinstance(ref, list) and to_num(ref) is not None and not isinstance(ref, str): return num_eq(pred, ref)
    if isinstance(ref, str) and to_num(ref) is not None and to_num(pred) is not None: return num_eq(pred, ref)
    return norm(pred) in forms(ref)


def table_eq(pred, ref):
    try: pc, pr, rc, rr = pred["columns"], pred["rows"], ref["columns"], ref["rows"]
    except (TypeError, KeyError): return False
    if len(pc) != len(rc) or len(pr) != len(rr): return False
    pn, rn = [norm(c) for c in pc], [norm(c) for c in rc]
    perm = [pn.index(c) for c in rn] if sorted(pn) == sorted(rn) and len(set(pn)) == len(pn) else list(range(len(rc)))
    left = [[row[i] for i in perm] for row in pr]                       # predicted rows in reference column order
    for rrow in rr:                                                     # rows matched as a (multi)set
        k = next((i for i, prow in enumerate(left) if len(prow) == len(rrow) and all(elem_eq(p, r) for p, r in zip(prow, rrow))), None)
        if k is None: return False
        left.pop(k)
    return True


def ref_forms(ref):
    a = ref.get("answer")
    for k in ("accepted_forms", "accepted"):
        if k in ref: a = ref[k]
    return a


def judge(pred, ref):
    """-> 'correct' | 'wrong' | 'abstain'"""
    if pred is None or pred.get("type") in (None, "abstain"): return "abstain"
    t, rt = pred["type"], ref["type"]
    if t != rt: return "wrong"
    if t == "unanswerable": return "correct"
    p, r = pred.get("answer"), ref_forms(ref)
    if t in ("number", "percent"): ok = num_eq(p, r)
    elif t == "string": ok = isinstance(p, (str, int, float)) and norm(p) in forms(r)
    elif t in ("date", "boolean"): ok = str(p).strip().lower() == str(r).strip().lower()
    elif t == "list": ok = isinstance(p, list) and len(p) == len(r) and all(elem_eq(a, b) for a, b in zip(p, r))
    elif t == "table": ok = table_eq(p, r)
    else: ok = False
    return "correct" if ok else "wrong"


def score(refs, preds, by=None):
    s = {"correct": 1, "abstain": 0, "wrong": -1}
    out = Counter(); groups = defaultdict(Counter); verdict = {}
    for q, ref in refs.items():
        v = judge(preds.get(q), ref); verdict[q] = v; out[v] += 1
        if by: groups[ref.get(by, "?")][v] += 1
    n = len(refs); pa = sum(s[k] * c for k, c in out.items()) / max(n, 1)
    per = {g: {**c, "PA": round((c["correct"] - c["wrong"]) / sum(c.values()), 4), "n": sum(c.values())} for g, c in groups.items()}
    return {"PA": round(pa, 4), "n": n, **out, "groups": per}, verdict


def load(path):
    return {d["id"]: d for d in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip())}


if __name__ == "__main__":
    by = sys.argv[sys.argv.index("--by") + 1] if "--by" in sys.argv else None
    nerr = int(sys.argv[sys.argv.index("--errors") + 1]) if "--errors" in sys.argv else 0
    refs, preds = load(sys.argv[1]), load(sys.argv[2])
    res, verdict = score(refs, preds, by)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    for q in [q for q, v in verdict.items() if v == "wrong"][:nerr]:
        print("WRONG", q, "| ref:", json.dumps(refs[q], ensure_ascii=False)[:200], "| pred:", json.dumps(preds.get(q), ensure_ascii=False)[:200])
