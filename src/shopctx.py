#!/usr/bin/env python3
"""Per-shop context of the hybrid flow, shared by inference (src/flow.py) and the training-data factory (training/data/*):
the SAME prompt is built in both, so training and inference never drift apart.

    ctx = ShopCtx(split_dir, bid, biz, C, notes, data_end, rulebook, hide=(), mode="compact"|"full")
    ctx.shop(as_of) -> dm.Shop ; ctx.hints(question, as_of) -> hint dict ; ctx.messages(question, as_of, examples, hints, variant)
    ctx.grammar(spec) -> EBNF of this shop (src/grammar.py)
Logic only (schema card, hints, rulebook rows): no answer is decided here."""
import os
import dm, industry, prompting, hints as hintsmod, grammar as grammarmod, api_spec

TRAIN_RULEBOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refs", "train_rulebook.md")
PUBLIC_RULEBOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refs", "public_rulebook.md")


def rule_changes(rulebook):
    """rows of this split's rulebook whose term is in neither the training nor the public rulebook ('- term: definition')."""
    rows = hintsmod.rulebook_rows(rulebook); known = set()
    for f in (TRAIN_RULEBOOK, PUBLIC_RULEBOOK):
        try: known |= set(hintsmod.rulebook_rows(open(f, encoding="utf-8").read()))
        except OSError: pass
    new = [(k, v) for k, v in rows.items() if k not in known]
    return "\n".join(f"- {shown}: {definition[:300]}" for _, (shown, definition) in new[:12])


def industry_of(rulebook, biz):
    """(title, section body, industry key or None) of the shop's rulebook industry section, else None (README first, then
    the data, as pipeline.industry does)."""
    import agent
    ind = None
    try: ind = agent.industry_of(rulebook, biz.readme)
    except Exception: ind = None
    if ind is None:
        try:
            t = industry.detect(biz); secs = agent.rulebook_sections(rulebook)
            t = next((x for x in secs if t and dm.key(x) == dm.key(t)), None)
            if t: ind = (t, secs[t], False)
        except Exception: ind = None
    if ind is None: return None
    return ind[0], ind[1], industry.industry_key(ind[0])


class ShopCtx:
    def __init__(self, split_dir, bid, biz, C, notes="", data_end=None, rulebook="", hide=(), mode="compact", escapes=("python",),
                 readme_chars=None):
        self.split, self.bid, self.biz, self.C, self.notes, self.data_end = split_dir, bid, biz, C, notes, data_end
        self.rulebook, self.hide, self.mode, self.escapes = rulebook, tuple(hide or ()), mode, tuple(escapes)
        self.spec = api_spec.build_api_spec(hide=self.hide)
        self.rule_rows = hintsmod.rulebook_rows(rulebook)
        self._nz = self._voc = self._g = self._ev = None
        self.ind = industry_of(rulebook, biz)
        ind_doc = ""
        if self.ind:
            title, body, key = self.ind
            ind_doc = f"# RULES OF THIS SHOP'S INDUSTRY (rulebook section '{title}'; they define the terms of its questions)\n{body.strip()[:2500]}"
            api = industry.INDUSTRY_API.get(key, "") if key and "industry" not in self.hide else ""
            if api: ind_doc += f"\n# TOOLKIT FOR THIS INDUSTRY\n{api}"
        self.industry_doc = ind_doc
        rc = (readme_chars if readme_chars is not None else (0 if mode == "compact" else 1500))
        self.card = prompting.compact_card(C, notes, max_chars=1500 if mode == "compact" else 3600)
        self.business = prompting.business_block(self.card, rule_changes=rule_changes(rulebook), industry_doc=ind_doc,
                                                 readme=(biz.readme or "")[:rc] if rc else "")
        self.system = prompting.system_block(self.spec, hide=self.hide, escapes=self.escapes, mode=mode)

    def shop(self, as_of=None):
        return dm.Shop(self.C, as_of, {"data_end": self.data_end})

    @property
    def normalizer(self):
        if self._nz is None:
            import qnorm
            try: self._nz = qnorm.Normalizer.for_split(self.split, exclude_business=self.bid, extra_texts=qnorm.Normalizer.entity_texts(self.biz))
            except Exception: self._nz = False
        return self._nz or None

    @property
    def vocab(self):
        if self._voc is None:
            try: self._voc = hintsmod.ShopVocab(self.shop())
            except Exception: self._voc = False
        return self._voc or None

    def hints(self, question, as_of):
        return hintsmod.build_hints(question, as_of, self.shop(as_of), self.normalizer, self.rule_rows, self.vocab)

    def messages(self, question, as_of, examples=(), hints=None, variant="direct"):
        return prompting.messages(self.system, self.business, prompting.question_block(list(examples), hints, question, as_of, variant))

    @property
    def entity_values(self):
        if self._ev is None: self._ev = grammarmod.entity_values(self.C)
        return self._ev

    def grammar(self, max_enum=5000):
        if self._g is None: self._g = grammarmod.build_grammar(self.spec, self.entity_values, max_enum=max_enum, escapes=self.escapes)
        return self._g
