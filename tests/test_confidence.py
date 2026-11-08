#!/usr/bin/env python3
"""src/confidence.py: the calibrator is monotone in the vote share on synthetic logs, recovers a known threshold, and the
threshold search maximises PA exactly (brute force check); features() reads the hybrid flow's log format."""
import os, sys, random
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import confidence as cf


def rec(k_agree, k_all, typ="number", early=False):
    S = [{"ok": True, "answer": {"type": typ, "answer": 1}, "turns": [{"ntok": 20}], "lp": -2.0, "flags": []} for _ in range(k_agree)]
    S += [{"ok": True, "answer": {"type": typ, "answer": 100 + i}, "turns": [{"ntok": 20}], "lp": -9.0, "flags": []} for i in range(k_all - k_agree)]
    return {"samples": {"samples": S, "hints": {"answer_type": "number"}, "stop": "early" if early else "round2"}}


def main():
    rng = random.Random(0); recs, y = [], []
    for _ in range(600):
        k = rng.randint(1, 8); a = rng.randint(1, k)
        recs.append(rec(a, k)); y.append(1 if rng.random() < a / k else 0)
    X = cf.matrix(recs); cal = cf.Calibrator().fit(X, y)
    p_lo, p_hi = cal.predict(cf.matrix([rec(2, 8)]))[0], cal.predict(cf.matrix([rec(8, 8)]))[0]
    assert p_hi > p_lo + 0.2, (p_lo, p_hi)
    p = np.array([rng.random() for _ in range(300)]); r = (p > 0.6).astype(float)
    th, pa = cf.choose_threshold(p, r)
    brute = max(((r[p >= t].sum() - (1 - r[p >= t]).sum()) / len(r), t) for t in sorted(set(p)) + [1.01])
    assert abs(pa - brute[0]) < 1e-9 and 0.55 < th <= 0.65, (th, pa, brute)
    err, ok = cf.conformal_check(p, r, th, 0.0); assert ok and err == 0.0
    fin, pp = cf.decide(rec(8, 8, early=True), cal, 0.5); assert fin["answer"] == 1, fin
    print(f"monotone {p_lo:.2f} < {p_hi:.2f}; theta {th:.3f} PA {pa:.3f}")
    print("test_confidence PASS")


if __name__ == "__main__":
    main()
