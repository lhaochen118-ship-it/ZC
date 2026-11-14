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

MAX_CARDS = 4


def key(s):
    """accent-free lowercase words (the question's and the triggers' matching form)."""
    return " ".join(re.findall(r"[^\W_]+", dm.strip_accents(str(s or "")).lower()))


def akey(s):
    """lowercase words WITH accents (short triggers must match their accents: 'hãng' is not 'hàng')."""
    import unicodedata
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFC", str(s or "")).lower()))


def _short(k): return " " not in k and len(k) < 6


def _trig_keys(c):
    """(matching form, accented?) of each trigger: long triggers accent-free, short single words with their accents."""
    out = []
    for w in ((c.get("trigger") or {}).get("words") or []):
        k = key(w)
        if len(k) < 2: continue
        e = (akey(w), True) if _short(k) and akey(w) != k else (k, False)
        if e not in out: out.append(e)
    return out


def hidden_names():
    """toolkit names hidden from this run (config hide_api, exported as $ZC_HIDE_API by predict_flow / eval_flow / hybrid):
    'industry' = every function of the hand-written industry module (mode H, user 14/11: no hand-written industry rule)."""
    h = {x for x in os.environ.get("ZC_HIDE_API", "").split(",") if x}
    if "industry" in h:
        try:
            import industry
            h |= set(industry.INDUSTRY_NAMES) | set(industry.HELPERS)
        except Exception: pass
    return h - {"industry"}


def mentions(text, names):
    """does a code hint / check expression name one of these functions?"""
    return bool(text) and bool(names) and bool(re.search(r"\b(" + "|".join(sorted(map(re.escape, names))) + r")\b", str(text)))


def strip_hidden(card, names=None):
    """the card without the parts that call a hidden function (its definition text stays: it was written by the model from the
    rulebook); unchanged when nothing is hidden."""
    names = hidden_names() if names is None else names
    if not names or not isinstance(card, dict): return card
    out = dict(card)
    if mentions(out.get("code_hint"), names): out["code_hint"] = None; out["hint_hidden"] = True
    if mentions(out.get("check"), names): out["check"] = None
    return out


class Library:
    def __init__(self, cards):
        hid = hidden_names()
        self.cards = [strip_hidden(c, hid) for c in cards if c and not c.get("disabled")]
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

    def prune(self, questions, max_df=0.05):
        """drop the triggers that occur in more than max_df of the given questions (the run's own questions: inputs only, no
        answer): a trigger like 'doanh thu' or 'tiền' says nothing about which rule applies. Cards left without a trigger go."""
        qs = [(" " + key(q) + " ", " " + akey(q) + " ") for q in questions if q]
        if not qs: return self
        lim = max(3, max_df * len(qs)); keep = []
        for c in self.cards:
            ks = [(w, acc) for w, acc in c["_keys"] if sum(1 for q, qa in qs if f" {w} " in (qa if acc else q)) <= lim]
            if ks: c["_keys"] = ks; keep.append(c)
        n0 = len(self.cards); self.cards = keep; self.pruned = n0 - len(keep)
        return self

    def match(self, question, bid=None, industry=None, k=MAX_CARDS, rulebook=None):
        """cards whose trigger words occur in the question (whole words, accent-free), longest trigger first; a shop card (P3)
        only for its shop, an industry card only for that industry, a compiled rulebook card (P2) only for shops read under that
        rulebook (rulebook = rb_key(text) of the shop's rulebook)."""
        q = " " + key(question) + " "; qa = " " + akey(question) + " "
        hits = []
        for c in self.cards:
            t = c.get("trigger") or {}
            if t.get("bid") and bid and t["bid"] != bid: continue
            if t.get("rulebook") and rulebook and t["rulebook"] != rulebook: continue
            if t.get("industry") and industry and t["industry"] != industry: continue
            best = max((len(w) for w, acc in c["_keys"] if f" {w} " in (qa if acc else q)), default=0)
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


def rb_key(text):
    """the key of a rulebook (compiled cards belong to it)."""
    import hashlib
    return hashlib.sha1(" ".join(str(text or "").split()).encode("utf-8")).hexdigest()[:10]


def render(cards):
    """the # RULES block of a question (after the examples, before the question: per-question part of the prompt)."""
    if not cards: return ""
    lines = []
    for c in cards:
        s = f"- {c.get('content', '').strip()}"
        if c.get("code_hint"): s += f" -> {c['code_hint']}"
        lines.append(s)
    return "# RULES (learned from data and the rulebook; use the ones that apply to this question)\n" + "\n".join(lines)
