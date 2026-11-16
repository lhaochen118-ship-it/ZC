#!/usr/bin/env python3
import os, re, json
from collections import Counter
import dm, fewshot

HERE = os.path.dirname(os.path.abspath(__file__))
EX_PATH = os.path.join(HERE, "refs", "bank", "examples.jsonl")
if not os.path.isfile(EX_PATH): EX_PATH = os.path.join(HERE, "refs", "answer_examples.jsonl")
TR_HO = ("biz_105", "biz_109", "biz_114", "biz_123", "biz_127", "biz_130")


def family(code):
    import api_spec
    g = api_spec.GROUP_OF
    fns = re.findall(r"\b([a-z_][a-z0-9_]*)\(", code or "")
    return next((f for f in fns if g.get(f) not in ("period", "number") and f not in ("len", "int", "round", "sum", "list", "key")), fns[0] if fns else "")


SIG_PATH = os.path.join(HERE, "refs", "bank", "schema_sig.json")


def schema_sig(C):
    out = set()
    for t, df in (C or {}).items():
        try:
            if df is None or not len(df): continue
            out.add(str(t))
            for c in df.columns:
                if df[c].notna().any(): out.add(f"{t}.{c}")
        except Exception: continue
    return out


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def skeleton(question, hints=None):
    t = " ".join(dm.key((hints or {}).get("clean") or question).split())
    for e in sorted((hints or {}).get("entities") or [], key=lambda e: -len(e.get("text") or "")):
        if e.get("kind") in ("unit", "base_unit"): continue
        x = e.get("text") or ""
        if len(x) >= 3: t = re.sub(r"(?<!\S)" + re.escape(x) + r"(?!\S)", f"<{e.get('kind', 'x')}>", t)
    return re.sub(r"\d+", "0", t)


class Bank:
    def __init__(self, ex, mode="ngram", embedder=None, vectors=None):
        self.ex = ex; self.mode = mode
        for e in self.ex: e.setdefault("family", family(e.get("code")))
        self.index = fewshot.Index([e.get("skeleton") or skeleton(e["question"], e.get("hints")) for e in ex]) if ex else None
        self.emb, self.vec = embedder, vectors
        self.sig = {}
        if os.path.isfile(SIG_PATH):
            try: self.sig = {b: set(v) for b, v in json.load(open(SIG_PATH, encoding="utf-8")).items()}
            except Exception: self.sig = {}

    def ranked(self, question, hints, n=80):
        sk = skeleton(question, hints)
        if self.mode == "embed" and self.emb is not None and self.vec is not None:
            import numpy as np
            q = self.emb.encode([sk], query=True)[0]; sc = self.vec @ q
            top = np.argsort(-sc)[:n]
            return [(float(sc[i]), int(i)) for i in top]
        return self.index.search(sk, n)

    @classmethod
    def load(cls, path=None, exclude_bids=TR_HO, include_holdout=False, mode="ngram", extra=(), embed_model="", vectors=""):
        rows = []
        for p in [path or EX_PATH] + list(extra):
            if p and os.path.isfile(p): rows += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        keep = [i for i, r in enumerate(rows) if include_holdout or r.get("business_id") not in set(exclude_bids or ())]
        ex = [rows[i] for i in keep]
        emb = vec = None
        if mode == "embed":
            import numpy as np, embedder
            vp = vectors or os.path.join(os.path.dirname(path or EX_PATH), "embeddings.npy")
            V = np.load(vp)
            if len(V) != len(rows): raise ValueError(f"{vp}: {len(V)} vectors for {len(rows)} bank rows")
            vec = V[keep]; emb = embedder.Embedder(embed_model)
        return cls(ex, mode, emb, vec)

    def retrieve(self, question, hints=None, k=4, exclude_bids=(), exclude_ids=(), per_family=1, schema=None, w_schema=0.0):
        if not self.index or k <= 0: return []
        base_ids = {str(i).split("~")[0] for i in exclude_ids}
        out, used = [], Counter()
        cand = self.ranked(question, hints, 200 if (w_schema and schema) else 80)
        if w_schema and schema and self.sig:
            cand = sorted(((s + w_schema * jaccard(self.sig.get(self.ex[i].get("business_id"), set()), schema), i) for s, i in cand),
                          key=lambda x: -x[0])
        for s, i in cand:
            e = self.ex[i]
            if e.get("business_id") in exclude_bids or str(e.get("id", "")).split("~")[0] in base_ids: continue
            if used[e["family"]] >= per_family: continue
            out.append(e); used[e["family"]] += 1
            if len(out) >= k: break
        return out
