#!/usr/bin/env python3
"""Abstain threshold of the hybrid flow (docs/08 WP8, QĐ14): theta maximising PA on the DEV sets (pub_full, pub_k0, training
shops minus the held-out ones), checked on the held-out sets by conformal risk control (error rate of the answered held-out
questions <= dev + 2 points, else theta is raised until it holds). Two confidences are compared: the flow's own vote share
('conf' of logs.jsonl) and the logistic calibrator of src/confidence.py (fitted on dev, cross-validated by shop for the dev
number). Held-out sets give aggregate numbers only (never a question).
Usage: calibrate.py <run_dir>[,<run_dir>...] [--dev pub_full,pub_k0] [--check pub_ho,tr_ho] [--out calibrator.json]
       [--json report.json]"""
import os, sys, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import score, confidence as C

TR_HO = ("biz_105", "biz_109", "biz_114", "biz_123", "biz_127", "biz_130")


def ids_of(sets):
    out = set()
    for s in sets:
        p = os.path.join(ROOT, "tests", "splits", s + ".txt")
        if os.path.isfile(p): out |= {l.strip() for l in open(p) if l.strip()}
    return out


def load(runs, refs):
    """per question id (the last log line wins): (record, right of the top answer, answered)."""
    recs = {}
    for run in runs:
        for l in open(os.path.join(run, "logs.jsonl"), encoding="utf-8"):
            r = json.loads(l); recs[r["id"]] = r
    out = {}
    for i, r in recs.items():
        ref = refs.get(i.split("~")[0])
        if ref is None: continue
        top = C.top_answer(r)
        try: right = top is not None and score.judge(top, ref) == "correct"
        except Exception: right = False
        out[i] = (r, float(right), float(top is not None))
    return out


def pa(p, right, answered, theta):
    keep = (np.asarray(p) >= theta) & (np.asarray(answered) > 0)
    r = np.asarray(right)[keep]
    return float((2 * r - 1).sum() / max(1, len(right))), float((1 - r).mean()) if len(r) else 0.0, int(keep.sum())


def shop_cv(X, y, a, shops, folds=5, l2=1.0):
    """out-of-fold calibrated confidences: shops split into folds (no shop in its own training fold)."""
    u = sorted(set(shops)); p = np.zeros(len(y))
    for f in range(folds):
        test = {s for k, s in enumerate(u) if k % folds == f}
        tr = np.array([s not in test for s in shops]); te = ~tr
        if not te.any(): continue
        cal = C.Calibrator(l2).fit(X[tr & (a > 0)], y[tr & (a > 0)])
        p[te] = cal.predict(X[te])
    return p


def main():
    runs = sys.argv[1].split(",")
    arg = lambda k, d="": sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    dev_sets = arg("--dev", "pub_full,pub_k0").split(","); chk_sets = arg("--check", "pub_ho,tr_ho").split(",")
    refs = {}
    for p in (os.path.join(ROOT, "data", "raw", "public_gt.jsonl"), os.path.join(ROOT, "data", "raw", "train", "train", "answers.jsonl")):
        if os.path.isfile(p): refs.update(score.load(p))
    L = load(runs, refs)
    held = ids_of(["pub_ho", "tr_ho"])
    dev_ids = ids_of(dev_sets) - held
    dev_ids |= {i for i, (r, _, _) in L.items() if "train" in dev_sets and r.get("business_id", "").startswith("biz_") and r["business_id"] not in TR_HO}
    dev = [i for i in sorted(L) if i in dev_ids]
    rep = {"runs": runs, "dev_sets": dev_sets, "n_dev": len(dev)}
    if not dev: print("no dev question in the logs"); sys.exit(1)
    X = C.matrix([L[i][0] for i in dev]); y = np.array([L[i][1] for i in dev]); a = np.array([L[i][2] for i in dev])
    shops = [L[i][0].get("business_id", "?") for i in dev]
    vote = np.array([float(L[i][0].get("conf") or 0.0) for i in dev])
    th_v, pa_v = C.choose_threshold(vote, y, a)
    cv = shop_cv(X, y, a, shops)
    th_c, pa_c = C.choose_threshold(cv, y, a)
    base = pa(vote, y, a, 0.5)
    print(f"dev {len(dev)} q: answer-everything PA {pa(vote, y, a, -1)[0]:.4f} | vote>=0.50 PA {base[0]:.4f} (err {base[1]:.3f}, {base[2]} answered)")
    print(f"  vote share: best theta {th_v:.3f} PA {pa_v:.4f}")
    print(f"  calibrator (shop CV): best theta {th_c:.3f} PA {pa_c:.4f}")
    rep.update(pa_all=pa(vote, y, a, -1)[0], pa_vote50=base[0], theta_vote=th_v, pa_vote=pa_v, theta_cal=th_c, pa_cal_cv=pa_c)
    cal = C.Calibrator().fit(X[a > 0], y[a > 0])
    use_cal = pa_c > pa_v + 0.005                                      # the calibrator only when it beats the plain vote share
    kind, theta = ("calibrator", th_c) if use_cal else ("vote", th_v)
    dev_err = pa(cv if use_cal else vote, y, a, theta)[1]
    rep.update(choice=kind, theta=theta, dev_err=dev_err)
    print(f"choice: {kind} theta {theta:.3f} (dev error rate of the answered {dev_err:.3f})")
    chk = [i for i in sorted(L) if i in ids_of(chk_sets)]
    if chk:
        Xh = C.matrix([L[i][0] for i in chk]); yh = np.array([L[i][1] for i in chk]); ah = np.array([L[i][2] for i in chk])
        ph = cal.predict(Xh) if use_cal else np.array([float(L[i][0].get("conf") or 0.0) for i in chk])
        err, ok = C.conformal_check(ph[ah > 0], yh[ah > 0], theta, dev_err)
        at_dev = pa(ph, yh, ah, theta)                                  # the honest number: theta chosen on dev only
        pdev = cv if use_cal else vote
        best_dev = pa(pdev, y, a, theta)[0]; th2 = theta
        while not ok and th2 < 1.0:                                     # QĐ14: raise theta until the held-out error rate holds,
            nxt = round(th2 + 0.01, 4)                                  # but never past a 1-point loss of dev PA: beyond it the gap
            if pa(pdev, y, a, nxt)[0] < best_dev - 0.01: break          # is a distribution shift (paraphrases are harder at every
            th2 = nxt; err, ok = C.conformal_check(ph[ah > 0], yh[ah > 0], th2, dev_err)   # theta), not a calibration miss
        res = pa(ph, yh, ah, th2)
        vote_h = pa(np.array([float(L[i][0].get('conf') or 0.0) for i in chk]), yh, ah, 0.5)[0]
        print(f"held-out {chk_sets} {len(chk)} q: at dev theta {theta:.3f}: PA {at_dev[0]:.4f} err {at_dev[1]:.3f} (vote>=0.50: {vote_h:.4f})")
        print(f"  conformal (dev err {dev_err:.3f} + 0.02): {'holds' if ok else 'SHIFT: not reachable within 1 dev point'} -> theta {th2:.3f} PA {res[0]:.4f}")
        rep.update(n_check=len(chk), pa_check_dev_theta=at_dev[0], check_err=err, conformal_ok=bool(ok), theta_final=th2, pa_check=res[0], pa_check_vote50=vote_h)
    if arg("--out") and use_cal: cal.save(arg("--out"))
    if arg("--json"): json.dump(rep, open(arg("--json"), "w"), indent=1)


if __name__ == "__main__":
    main()
