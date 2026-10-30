#!/usr/bin/env python3
"""Local re-implementation of the DataMate scorer (docs/00_TASK.md §4): per-question correct / wrong / abstain and
Penalized Accuracy PA = mean(s_i), s_i = +1 correct, 0 abstain (or missing), -1 wrong.

Reference format (answers.jsonl): {"id", "type", "answer", "alias"}; alias = other accepted spellings + the code:
a list of strings for `string`, one list per element for `list`, one list per row (entity-name column) for `table`.
Format rules of the rulebook (§9): numbers must be JSON numbers (not strings), booleans JSON booleans.

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


def elem_eq(pred, ref, extra=()):
    """A list element / table cell: number rule if the reference is numeric, else string rule (with accepted aliases)."""
    if isinstance(ref, (int, float)) and not isinstance(ref, bool): return num_eq(pred, ref)
    if isinstance(ref, str) and to_num(ref) is not None and to_num(pred) is not None and not extra: return num_eq(pred, ref)
    return norm(pred) in forms([ref] + list(extra))


def table_eq(pred, ref, row_alias=None):
    try: pc, pr, rc, rr = pred["columns"], pred["rows"], ref["columns"], ref["rows"]
    except (TypeError, KeyError): return False
    if len(pc) != len(rc) or len(pr) != len(rr): return False
    pn, rn = [norm(c) for c in pc], [norm(c) for c in rc]
    perm = [pn.index(c) for c in rn] if sorted(pn) == sorted(rn) and len(set(pn)) == len(pn) else list(range(len(rc)))
    left = [[row[i] for i in perm] if len(row) == len(rc) else row for row in pr]   # predicted rows in reference column order
    for j, rrow in enumerate(rr):                                       # rows matched as a (multi)set
        al = row_alias[j] if row_alias and j < len(row_alias) else []
        k = next((i for i, prow in enumerate(left) if len(prow) == len(rrow) and
                  all(elem_eq(p, r, al if c == 0 else ()) for c, (p, r) in enumerate(zip(prow, rrow)))), None)
        if k is None: return False
        left.pop(k)
    return True


def ref_forms(ref):
    """Accepted forms of a string reference: the answer plus every alias (other spellings in the data, the code)."""
    a = ref.get("answer")
    extra = ref.get("alias") or ref.get("accepted_forms") or ref.get("accepted") or []
    return ([a] if not isinstance(a, list) else a) + [x for x in extra if isinstance(x, str)]


def judge(pred, ref):
    """-> 'correct' | 'wrong' | 'abstain'   (ref: {"type", "answer", "alias"} as in answers.jsonl)"""
    if pred is None or pred.get("type") in (None, "abstain"): return "abstain"
    t, rt = pred["type"], ref["type"]
    if t != rt: return "wrong"
    if t == "unanswerable": return "correct"
    p, r, al = pred.get("answer"), ref.get("answer"), ref.get("alias") or []
    if t in ("number", "percent"): ok = isinstance(p, (int, float)) and not isinstance(p, bool) and num_eq(p, r)
    elif t == "string": ok = isinstance(p, str) and norm(p) in forms(ref_forms(ref))
    elif t == "date": ok = isinstance(p, str) and p.strip() == str(r).strip()
    elif t == "boolean": ok = isinstance(p, bool) and p == r
    elif t == "list":
        ok = isinstance(p, list) and len(p) == len(r) and all(elem_eq(a, b, al[i] if i < len(al) and isinstance(al[i], list) else ())
                                                             for i, (a, b) in enumerate(zip(p, r)))
    elif t == "table": ok = table_eq(p, r, al)
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
