#!/usr/bin/env python3
import json, math
import numpy as np

FEATS = ["vote_share", "top_votes", "n_valid", "n_samples", "n_groups", "flags_top", "early", "lp_mean", "lp_min", "turns_mean",
         "precheck_err", "exec_err", "escape", "unans", "t_number", "t_percent", "t_list", "t_string", "t_date", "t_table", "t_boolean",
         "hint_type_agree", "hint_period", "hint_entities"]


def canon_key(ans):
    try:
        import agent
        return agent.canon_of(ans)
    except Exception: return json.dumps(ans, sort_keys=True, ensure_ascii=False)


def features(rec):
    lg = rec.get("samples") if isinstance(rec.get("samples"), dict) else {}
    S = lg.get("samples") or []; h = lg.get("hints") or {}
    valid = [s for s in S if s.get("ok")]
    groups = {}
    for s in valid: groups.setdefault(canon_key(s["answer"]), []).append(s)
    top = max(groups.values(), key=len) if groups else []
    ans = top[0]["answer"] if top else {}
    lps = []
    for s in valid:
        lp = s.get("lp"); nt = (s.get("turns") or [{}])[-1].get("ntok") or 0
        if lp is not None and nt: lps.append(lp / nt)
    f = {"vote_share": len(top) / max(1, len(S)), "top_votes": float(len(top)), "n_valid": float(len(valid)), "n_samples": float(len(S)),
         "n_groups": float(len(groups)), "flags_top": float(sum(1 for s in top if s.get("flags"))), "early": float(lg.get("stop") == "early"),
         "lp_mean": float(np.mean(lps)) if lps else 0.0, "lp_min": float(np.min(lps)) if lps else 0.0,
         "turns_mean": float(np.mean([len(s.get("turns") or []) for s in S])) if S else 0.0,
         "precheck_err": float(sum(1 for s in S for t in (s.get("turns") or []) if any(str(x).startswith("error") for x in (t.get("precheck") or [])))),
         "exec_err": float(sum(1 for s in S for t in (s.get("turns") or []) if (t.get("exec") or {}).get("ok") is False)),
         "escape": float(any(s.get("escape") for s in top)), "unans": float(ans.get("type") == "unanswerable")}
    for t in ("number", "percent", "list", "string", "date", "table", "boolean"): f["t_" + t] = float(ans.get("type") == t)
    f["hint_type_agree"] = float(bool(h.get("answer_type")) and h.get("answer_type") == ans.get("type"))
    f["hint_period"] = float(bool(h.get("period"))); f["hint_entities"] = float(len(h.get("entities") or []))
    return f


def matrix(recs): return np.array([[features(r)[k] for k in FEATS] for r in recs], dtype=float)


class Calibrator:
    def __init__(self, l2=1.0): self.l2 = l2; self.w = None; self.mu = None; self.sd = None

    def fit(self, X, y, iters=500, lr=0.5):
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-6
        Z = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1))]); w = np.zeros(Z.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-Z @ w)); g = Z.T @ (p - y) / len(y) + self.l2 * np.r_[w[:-1], 0] / len(y)
            w -= lr * g
        self.w = w; return self

    def predict(self, X):
        X = np.asarray(X, float); Z = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1))])
        return 1 / (1 + np.exp(-Z @ self.w))

    def save(self, path): json.dump({"feats": FEATS, "w": self.w.tolist(), "mu": self.mu.tolist(), "sd": self.sd.tolist(), "l2": self.l2}, open(path, "w"))

    @classmethod
    def load(cls, path):
        d = json.load(open(path)); c = cls(d.get("l2", 1.0)); c.w, c.mu, c.sd = np.array(d["w"]), np.array(d["mu"]), np.array(d["sd"]); return c


def choose_threshold(p, right, answered=None):
    p = np.asarray(p, float); r = np.asarray(right, float); a = np.ones_like(r) if answered is None else np.asarray(answered, float)
    order = np.argsort(-p, kind="stable"); gain = np.where(a[order] > 0, 2 * r[order] - 1, 0.0)
    cum = np.cumsum(gain); n = len(r)
    best_i, best = -1, 0.0
    for i in range(n):
        if i + 1 < n and p[order[i + 1]] == p[order[i]]: continue
        if cum[i] / n > best + 1e-12: best_i, best = i, cum[i] / n
    return (float(p[order[best_i]]) if best_i >= 0 else 1.01), float(best)


def conformal_check(p, right, theta, dev_err, slack=0.02):
    p = np.asarray(p, float); r = np.asarray(right, float); keep = p >= theta
    err = float((1 - r[keep]).mean()) if keep.any() else 0.0
    return err, err <= dev_err + slack


def decide(rec, cal, theta):
    f = np.array([[features(rec)[k] for k in FEATS]])
    p = float(cal.predict(f)[0])
    fin = top_answer(rec) or {"type": "abstain"}
    return (fin if p >= theta else {"type": "abstain"}), p


def top_answer(rec):
    lg = rec.get("samples") if isinstance(rec.get("samples"), dict) else {}
    groups = {}
    for s in lg.get("samples") or []:
        if s.get("ok"): groups.setdefault(canon_key(s["answer"]), []).append(s["answer"])
    if not groups: return None
    return max(groups.values(), key=len)[0]
