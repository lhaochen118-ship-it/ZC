#!/usr/bin/env python3
"""Question <-> program consistency facts (a general check layer for the model's programs).

Reads only what any Vietnamese data question states explicitly, with general language parsers (no shop, no rulebook, no
industry knowledge): the time period(s) it names (qparse.parse_period / month_list: Vietnamese date and period expressions),
the numeric thresholds it sets ('ít nhất N', 'từ N ... trở lên', 'trên N', 'dưới N'; hints.thresholds) and the answer form its
interrogative asks for (hints.answer_type: 'bao nhiêu' -> number, '... nào' -> a name / list, 'có ... không' -> boolean).
precheck.precheck compares a program with these facts: a program that ignores a period or a threshold the question states gets
a warning, and a warned sample counts half in the vote (hybrid.py). The facts never answer and never reach the prompt.

    facts(question, as_of) -> {"period": {...} | None, "periods": [...], "thresholds": [...], "answer_type": str | None}"""
import qparse


def facts(question, as_of):
    import hints
    h = {"period": None, "periods": [], "thresholds": [], "answer_type": None, "entities": []}
    q = str(question or "")
    try:
        p, code = qparse.parse_period(q, as_of)
        if p is not None: h["period"] = {"code": code, "from": str(p.start.date()), "to": str(p.end.date())}
    except Exception:
        pass
    try:
        ms = qparse.month_list(q, as_of)
        if len(ms) >= 2: h["periods"] = ms
    except Exception:
        pass
    t = qparse.untelex(qparse.K(q))
    try: h["thresholds"] = hints.thresholds(t)
    except Exception: pass
    try: h["answer_type"] = hints.answer_type(t)
    except Exception: pass
    return h
