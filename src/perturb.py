"""Question perturbations of the stress tests (tests/eval_split.py --perturb-q, src/stress_flow.py): not used to fit
anything, only to measure how the reader copes with question styles it was not written for."""
import re
import dm

ABBR_NOISE = [("bao nhiêu", "bnhiu"), ("được", "dc"), ("không", "ko"), ("doanh thu", "dthu"), ("tháng", "th"), ("hôm qua", "hqua"),
              ("với", "vs"), ("trước", "trc"), ("vậy", "z"), ("cho", "cko"), ("mình", "mik"), ("nhiều", "nhìu"), ("nhất", "nhứt")]
SWAPS = [("s", "x"), ("x", "s"), ("tr", "ch"), ("gi", "d"), ("l", "n")]


def perturb_question(q, mode, rng):
    """stress test of the question reader (not used to fit anything): 'accentfree' strips every accent; 'noise' applies
    1-3 chat-noise edits of the kinds the questions show (abbreviations, Telex, a word without accents, s/x tr/ch swaps,
    two words glued), never inside capitalised words (names) or numbers."""
    import unicodedata as ud
    if mode == "accentfree": return dm.strip_accents(q)
    words = q.split(" ")
    def plain(w): return w[:1].islower() and not re.search(r"\d", w)
    for _ in range(rng.randint(1, 3)):
        op = rng.choice(["abbr", "telex", "noacc", "swap", "glue"])
        if op == "abbr":
            cand = [(a, b) for a, b in ABBR_NOISE if a in q.lower()]
            if cand: a, b = rng.choice(cand); q = re.sub(re.escape(a), b, q, count=1, flags=re.I); words = q.split(" ")
            continue
        idx = [i for i, w in enumerate(words) if plain(w) and len(w) > 1]
        if not idx: continue
        i = rng.choice(idx); w = words[i]
        if op == "noacc": words[i] = dm.strip_accents(w)
        elif op == "telex":
            tone = {"\u0301": "s", "\u0300": "f", "\u0309": "r", "\u0303": "x", "\u0323": "j"}
            d = ud.normalize("NFD", w); t = "".join(tone.get(c, "") for c in d)
            base = "".join(c for c in d if c not in tone)
            base = ud.normalize("NFC", base).replace("đ", "dd").replace("ơ", "ow").replace("ư", "uw").replace("ô", "oo").replace("â", "aa").replace("ê", "ee").replace("ă", "aw")
            words[i] = base + t
        elif op == "swap":
            for a, b in SWAPS:
                if w.startswith(a): words[i] = b + w[len(a):]; break
        elif op == "glue" and i + 1 < len(words) and plain(words[i + 1]):
            words[i] = w + words[i + 1]; del words[i + 1]
        q = " ".join(words)
    return q
