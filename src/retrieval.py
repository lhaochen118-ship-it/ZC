#!/usr/bin/env python3
"""Example retrieval of the hybrid flow (docs/08_MASTER_PLAN.md §5.1, module M6): checked (question -> program) pairs of the
training shops, chosen by the question's SKELETON (entities, numbers and periods masked, as in DAIL-SQL) so that the examples
show the method, not the names; at most one example per function family (the program's main toolkit function), most similar
first. mode "ngram": character n-gram cosine (fewshot.Index, no model); mode "embed": an embedding model (WP8, S1).

    Bank.load(path=None, exclude_bids=TR_HO) ; bank.retrieve(question, hints=None, k=4, exclude_bids=(), exclude_ids=()) -> [example]
Held-out shops (tests/make_splits.py TR_HO) are left out of the bank unless include_holdout=True (final model only)."""
import os, re, json
from collections import Counter
import dm, fewshot

HERE = os.path.dirname(os.path.abspath(__file__))
EX_PATH = os.path.join(HERE, "refs", "bank", "examples.jsonl")      # training/data/d1_train_pairs.py (skeletons computed)
if not os.path.isfile(EX_PATH): EX_PATH = os.path.join(HERE, "refs", "answer_examples.jsonl")
TR_HO = ("biz_105", "biz_109", "biz_114", "biz_123", "biz_127", "biz_130")
PERIOD_RX = re.compile(r"\b(?:thang|thg|th|quy|q|nam|tuan|ngay|hom|bua|tet)\s*(?:nay|qua|truoc|roi|sau|toi|0+(?:\s*0+)*)?\b")


def family(code):
    """the main toolkit function of a program (its family), e.g. 'revenue', 'top_items'."""
    import api_spec
    g = api_spec.GROUP_OF
    fns = re.findall(r"\b([a-z_][a-z0-9_]*)\(", code or "")
    return next((f for f in fns if g.get(f) not in ("period", "number") and f not in ("len", "int", "round", "sum", "list", "key")), fns[0] if fns else "")


def skeleton(question, hints=None):
    """accent-free question with entity mentions -> <kind>, digits -> 0 (and the period words kept: they carry the method)."""
    t = " ".join(dm.key((hints or {}).get("clean") or question).split())
    for e in sorted((hints or {}).get("entities") or [], key=lambda e: -len(e.get("text") or "")):
        x = e.get("text") or ""
        if len(x) >= 3: t = re.sub(r"(?<!\S)" + re.escape(x) + r"(?!\S)", f"<{e.get('kind', 'x')}>", t)
    return re.sub(r"\d+", "0", t)


class Bank:
    def __init__(self, ex, mode="ngram"):
        self.ex = ex; self.mode = mode
        for e in self.ex: e.setdefault("family", family(e.get("code")))
        self.index = fewshot.Index([e.get("skeleton") or skeleton(e["question"], e.get("hints")) for e in ex]) if ex else None

    @classmethod
    def load(cls, path=None, exclude_bids=TR_HO, include_holdout=False, mode="ngram", extra=()):
        rows = []
        for p in [path or EX_PATH] + list(extra):
            if p and os.path.isfile(p): rows += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        ex = [r for r in rows if include_holdout or r.get("business_id") not in set(exclude_bids or ())]
        return cls(ex, mode)

    def retrieve(self, question, hints=None, k=4, exclude_bids=(), exclude_ids=(), per_family=1):
        """k examples, most similar first, at most per_family per function family; never one of exclude_bids / exclude_ids
        (an id excludes its paraphrases '<id>~p<k>' too)."""
        if not self.index or k <= 0: return []
        base_ids = {str(i).split("~")[0] for i in exclude_ids}
        out, used = [], Counter()
        for s, i in self.index.search(skeleton(question, hints), 80):
            e = self.ex[i]
            if e.get("business_id") in exclude_bids or str(e.get("id", "")).split("~")[0] in base_ids: continue
            if used[e["family"]] >= per_family: continue
            out.append(e); used[e["family"]] += 1
            if len(out) >= k: break
        return out
