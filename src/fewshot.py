#!/usr/bin/env python3
"""Few-shot examples for the LLM: verified (question -> toolkit program) pairs from the train split, and the reference
mapping programs, retrieved by similarity.

answer examples: refs/answer_examples.jsonl, one JSON per line {question, as_of_date, tpl, code, business_id}; built by
                 tests/eval_lib.py --dump (only programs whose answer matched the train answer) + the train questions
                 whose answer is unanswerable.
Similarity: cosine over character 3-5-grams of the accent-free question with digits masked (pure Python, no extra
model, so it costs nothing in the 15B parameter budget)."""
import os, re, json, math
from collections import Counter, defaultdict
import dm

HERE = os.path.dirname(os.path.abspath(__file__))
EX_PATH = os.path.join(HERE, "refs", "answer_examples.jsonl")


def qkey(q):
    k = dm.key(q); k = re.sub(r"\d", "0", k)
    return re.sub(r"\s+", " ", k)


def grams(s, lo=3, hi=5):
    s = f" {s} "; return Counter(s[i:i + n] for n in range(lo, hi + 1) for i in range(len(s) - n + 1))


class Index:
    """inverted-index cosine retrieval over character n-grams."""
    def __init__(self, texts):
        self.vecs = [grams(t) for t in texts]; self.inv = defaultdict(list); df = Counter()
        for v in self.vecs: df.update(v.keys())
        n = len(texts); self.idf = {g: math.log((1 + n) / (1 + c)) + 1 for g, c in df.items()}
        self.norm = []
        for i, v in enumerate(self.vecs):
            w = {g: c * self.idf[g] for g, c in v.items()}; self.vecs[i] = w; self.norm.append(math.sqrt(sum(x * x for x in w.values())) or 1.0)
            for g in w: self.inv[g].append(i)

    def search(self, text, k=10):
        v = grams(text); w = {g: c * self.idf.get(g, 1.0) for g, c in v.items()}; nq = math.sqrt(sum(x * x for x in w.values())) or 1.0
        sc = defaultdict(float)
        for g, x in w.items():
            for i in self.inv.get(g, ()): sc[i] += x * self.vecs[i][g]
        return sorted(((s / (nq * self.norm[i]), i) for i, s in sc.items()), reverse=True)[:k]


class AnswerExamples:
    def __init__(self, path=EX_PATH):
        self.ex = [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.isfile(path) else []
        self.index = Index([qkey(e["question"]) for e in self.ex]) if self.ex else None

    def pick(self, question, k=4, exclude_business=None, per_tpl=2):
        """k similar examples (at most per_tpl of one template), excluding one business (honest evaluation on train)."""
        if not self.index: return []
        out, used = [], Counter()
        for s, i in self.index.search(qkey(question), 60):
            e = self.ex[i]
            if exclude_business and e.get("business_id") == exclude_business: continue
            if used[e["tpl"]] >= per_tpl: continue
            out.append(e); used[e["tpl"]] += 1
            if len(out) >= k: break
        return out


def render_examples(exs):
    """examples block for the answer prompt (most similar last, next to the question)."""
    parts = []
    for e in reversed(exs):
        parts.append(f"Q (as_of_date {e['as_of_date']}): {e['question']}\n```python\n{e['code']}\n```")
    return "\n\n".join(parts)
