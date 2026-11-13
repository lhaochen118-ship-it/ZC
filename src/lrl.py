#!/usr/bin/env python3
"""Learned rule layer (plan 3, docs/16_PLAN3_LEARNED_RULES.md §2): rule CARDS written by the model itself (src/lrl_gen.py), never
by hand — rulebook rows compiled to toolkit terms (P2, at inference, from the split's own rulebook), value aliases read from a
shop's data (P3), rules induced from the model's mistakes on TRAINING questions and accepted only when they help on other shops
(P1, tests/lrl_accept.py), chat-noise words of ViLexNorm (P4). This module only stores, matches, renders and checks cards;
nothing here decides an answer.

card: {"id", "kind": period|lexicon|definition|unit|value|check, "trigger": {"words": [...], "industry": None|key, "bid": None|shop},
       "content": one sentence, "code_hint": toolkit expression / parameters, "check": None | python expression over
       (q: accent-free lowercase question, code: program text) that is True when the program BREAKS the rule,
       "source", "stats"}

    lib = Library.load(paths...) ; lib.match(question, bid=None, industry=None, k=6) -> [card] ; render(cards) -> prompt block
    lib.violations(cards, question, code) -> ["R12: <content>"]"""
import os, re, json
import dm

MAX_CARDS = 6


def key(s):
    """accent-free lowercase words (the question's and the triggers' matching form)."""
    return " ".join(re.findall(r"[^\W_]+", dm.strip_accents(str(s or "")).lower()))


def _trig_keys(c):
    out = []
    for w in ((c.get("trigger") or {}).get("words") or []):
        k = key(w)
        if len(k) >= 2 and k not in out: out.append(k)
    return out


class Library:
    def __init__(self, cards):
        self.cards = [c for c in cards if c and not c.get("disabled")]
        for c in self.cards: c["_keys"] = _trig_keys(c)

    @classmethod
    def load(cls, *paths):
        cards = []
        for p in paths:
            for f in (p.split(",") if isinstance(p, str) else p):
                if f and os.path.isfile(f):
                    cards += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        return cls(cards)

    def __len__(self): return len(self.cards)

    def match(self, question, bid=None, industry=None, k=MAX_CARDS):
        """cards whose trigger words occur in the question (whole words, accent-free), longest trigger first; a shop card (P3)
        only for its shop, an industry card only for that industry."""
        q = " " + key(question) + " "
        hits = []
        for c in self.cards:
            t = c.get("trigger") or {}
            if t.get("bid") and bid and t["bid"] != bid: continue
            if t.get("industry") and industry and t["industry"] != industry: continue
            best = max((len(w) for w in c["_keys"] if f" {w} " in q), default=0)
            if best: hits.append((best, c.get("kind") == "check", c))
        hits.sort(key=lambda x: (-x[0], x[1]))
        out, seen = [], set()
        for _, _, c in hits:
            sig = (c.get("content"), c.get("code_hint"))
            if sig in seen: continue
            seen.add(sig); out.append(c)
            if len(out) >= k: break
        return out

    @staticmethod
    def violations(cards, question, code):
        """messages of the matched cards whose check flags the program (a repair turn shows them)."""
        out = []
        q = key(question)
        for c in cards:
            ex = c.get("check")
            if not ex or not code: continue
            try:
                if eval(ex, {"__builtins__": {}}, {"q": q, "code": code, "re": re, "len": len, "any": any, "all": all}):
                    out.append(f"{c.get('id', 'R?')}: {c.get('content', '')}" + (f" (use {c['code_hint']})" if c.get("code_hint") else ""))
            except Exception:
                continue
        return out


def render(cards):
    """the # RULES block of a question (after the examples, before the question: per-question part of the prompt)."""
    if not cards: return ""
    lines = []
    for c in cards:
        s = f"- {c.get('content', '').strip()}"
        if c.get("code_hint"): s += f" -> {c['code_hint']}"
        lines.append(s)
    return "# RULES (learned from data and the rulebook; use the ones that apply to this question)\n" + "\n".join(lines)
