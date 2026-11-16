#!/usr/bin/env python3
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
