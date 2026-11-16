#!/usr/bin/env python3
import os, re, json
import dm

MAX_CARDS = 4


def key(s):
    return " ".join(re.findall(r"[^\W_]+", dm.strip_accents(str(s or "")).lower()))


def akey(s):
    import unicodedata
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFC", str(s or "")).lower()))


def _short(k): return " " not in k and len(k) < 6


def _trig_keys(c):
    out = []
    for w in ((c.get("trigger") or {}).get("words") or []):
        k = key(w)
        if len(k) < 2: continue
        e = (akey(w), True) if _short(k) and akey(w) != k else (k, False)
        if e not in out: out.append(e)
    return out


def hidden_names():
    h = {x for x in os.environ.get("ZC_HIDE_API", "").split(",") if x}
    if "industry" in h:
        try:
            import industry
            h |= set(industry.INDUSTRY_NAMES) | set(industry.HELPERS)
        except Exception: pass
    if "industry_metrics" in h:
        try:
            import industry
            h |= set(industry.INDUSTRY_NAMES)
        except Exception: pass
    return h - {"industry", "industry_metrics"}


def mentions(text, names):
    return bool(text) and bool(names) and bool(re.search(r"\b(" + "|".join(sorted(map(re.escape, names))) + r")\b", str(text)))


def strip_hidden(card, names=None):
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
        qs = [(" " + key(q) + " ", " " + akey(q) + " ") for q in questions if q]
        if not qs: return self
        lim = max(3, max_df * len(qs)); keep = []
        for c in self.cards:
            ks = [(w, acc) for w, acc in c["_keys"] if sum(1 for q, qa in qs if f" {w} " in (qa if acc else q)) <= lim]
            if ks: c["_keys"] = ks; keep.append(c)
        n0 = len(self.cards); self.cards = keep; self.pruned = n0 - len(keep)
        return self

    def match(self, question, bid=None, industry=None, k=MAX_CARDS, rulebook=None):
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
    import hashlib
    return hashlib.sha1(" ".join(str(text or "").split()).encode("utf-8")).hexdigest()[:10]


def render(cards):
    if not cards: return ""
    lines = []
    for c in cards:
        s = f"- {c.get('content', '').strip()}"
        if c.get("code_hint"): s += f" -> {c['code_hint']}"
        lines.append(s)
    return "# RULES (learned from data and the rulebook; use the ones that apply to this question)\n" + "\n".join(lines)
