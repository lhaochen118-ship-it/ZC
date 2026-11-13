#!/usr/bin/env python3
"""Shared helpers of the training-data factory (docs/08_MASTER_PLAN.md §7): shop contexts built exactly as at inference
(src/shopctx.py), the checked example bank, the held-out shops. Nothing here reads the public test (forbidden for training)."""
import os, sys, json, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src") if os.path.isdir(os.path.join(ROOT, "src")) else ROOT   # repo layout, or the kernel's (src files at the root)
if SRC not in sys.path: sys.path.insert(0, SRC)
os.environ.setdefault("ZC_ADJ_ROWS", "move")
TR_HO = ("biz_105", "biz_109", "biz_114", "biz_123", "biz_127", "biz_130")


def jl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def write_jl(p, rows):
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    os.replace(p + ".tmp", p)


def canon(split, cache, canon_dir, bid):
    """(biz, C, notes, data_end) of a shop: ingest cache + automap canonical tables (pickled once in canon_dir)."""
    import ingest, automap
    pc = os.path.join(canon_dir, bid + ".pkl")
    bp = os.path.join(cache, bid + ".pkl")
    biz = pickle.load(open(bp, "rb")) if os.path.isfile(bp) else ingest.load_business(os.path.join(split, "data", bid))
    if os.path.isfile(pc):
        d = pickle.load(open(pc, "rb")); return biz, d["C"], d.get("notes", ""), d.get("data_end")
    C, notes, _ = automap.build(biz); de = automap.parse_readme(biz.readme)["data_end"]
    os.makedirs(canon_dir, exist_ok=True); pickle.dump({"C": C, "notes": notes, "data_end": de}, open(pc, "wb"), protocol=4)
    return biz, C, notes, de


_CTX = {}


def ctx(split, cache, canon_dir, bid, mode="compact", hide=()):
    import shopctx
    k = (split, bid, mode, tuple(hide))
    if k not in _CTX:
        biz, C, notes, de = canon(split, cache, canon_dir, bid)
        rb = open(os.path.join(split, "rulebook.md"), encoding="utf-8").read()
        _CTX[k] = shopctx.ShopCtx(split, bid, biz, C, notes, de, rb, hide=hide, mode=mode)
    return _CTX[k]


def sample(c, q, code, examples, hints, source, answer=None, variant="direct", plan=None, rules=""):
    """one SFT record: the inference prompt of shop context c + the labelled output (docs/08 §5.2 / §5.3); rules: the # RULES
    block of the learned rule cards (src/lrl.render, plan 3)."""
    import grammar
    msgs = c.messages(q["question"], q["as_of_date"], examples, hints, variant, rules)
    label = grammar.wrap(code, plan or grammar.auto_plan(code))
    return {"id": q["id"], "business_id": q["business_id"], "source": source, "messages": msgs + [{"role": "assistant", "content": label}],
            "program": code, "answer": answer, "as_of_date": q["as_of_date"], "question": q["question"]}
