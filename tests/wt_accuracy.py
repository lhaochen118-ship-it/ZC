#!/usr/bin/env python3
"""v2 G gate (handoff/05 step 4.1): accuracy of a tool writer on the rows of the 6 held-out training shops (never trained on).

Reference = the teacher's function kept by self-consistency (training/data/wt_build.py -> wt_eval_rows.jsonl). Candidate = the
units a writer arm kept for the same rows (src/lrl_gen.py --mode unit --rows wt_eval_rows.jsonl -> cards_unit*.jsonl). Both run on
the shop's data over the same 13 periods (units.verify signature); a row counts as right when the two signatures are equal.
Reports per arm: rows with a kept unit (coverage), rows right (accuracy over all rows), precision (right / kept), by source.
Usage: wt_accuracy.py <wt_eval_rows.jsonl> <arm>=<dir with cards_unit*.jsonl>[,<arm>=<dir>...] [--split data/raw/train/train] [--out F]"""
import os, sys, json, glob, argparse
from collections import Counter, defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def signature(u, shop):
    import units
    C, de, T = shop
    try:
        ok, why, sample = units.verify(u, C, de, T)
        return u.get("_sig") if ok else None
    except Exception: return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows"); ap.add_argument("arms"); ap.add_argument("--split", default=os.path.join(ROOT, "data", "raw", "train", "train"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import lrl_gen
    rows = [json.loads(l) for l in open(a.rows, encoding="utf-8") if l.strip()]
    ref = {}
    for r in rows:
        sh = lrl_gen._shop_load(a.split, "", r["bid"])
        if not sh: continue
        u = {"kind": "unit", "bid": r["bid"], "term": r["term"], "name": r["teacher"]["name"], "code": r["teacher"]["code"]}
        ref[(r["bid"], r["term"])] = (signature(u, sh), r.get("src"))
    res = {}
    for spec in [x for x in a.arms.split(",") if x]:
        arm, d = spec.split("=", 1)
        got = {}
        for f in glob.glob(os.path.join(d, "**", "cards_unit*.jsonl"), recursive=True):
            for l in open(f, encoding="utf-8"):
                if l.strip(): u = json.loads(l); got.setdefault((u["bid"], u["term"]), u)
        c = Counter(); by = defaultdict(Counter)
        for k, (rs, src) in ref.items():
            c["rows"] += 1; by[src]["rows"] += 1
            u = got.get(k)
            if not u: continue
            c["kept"] += 1; by[src]["kept"] += 1
            s = signature(u, lrl_gen._shop_load(a.split, "", k[0]))
            if rs is not None and s == rs: c["right"] += 1; by[src]["right"] += 1
        res[arm] = {"rows": c["rows"], "kept": c["kept"], "right": c["right"], "accuracy": round(c["right"] / max(1, c["rows"]), 4),
                    "precision": round(c["right"] / max(1, c["kept"]), 4), "by_src": {k: dict(v) for k, v in by.items()}}
        print(arm, json.dumps(res[arm], ensure_ascii=False), flush=True)
    if a.out: json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
