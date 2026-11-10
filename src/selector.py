#!/usr/bin/env python3
"""Learned answer selector (docs/12_PLAN2.md §6, track M, WP-M6; LEVER / CHASE-SQL style): instead of taking the most-voted
answer group, a small model scores EVERY candidate group of a question from general evidence about the samples behind it and
picks the best one, or abstains when even the best one is unlikely to be right. It never writes an answer: it only chooses among
the answers the model's own programs produced.

    groups(rec) -> [(answer, [samples])]                 candidate answers of one question log (hybrid flow logs.jsonl record)
    group_features(rec, ans, smp, all_groups) -> vector  general features (vote share, support by reasoning samples, distinct
                                                         programs, flags, repairs, hinted answer type, raw pandas vs toolkit …)
    Selector.fit(records, refs) / .decide(rec) / .save / .load
Training data: logs of the training shops (not held out) and their paraphrases only — never the public or private test."""
import re, json
from collections import defaultdict
import numpy as np

FEATS = ["share", "share_w", "rank0", "gap_top", "n_support", "n_think", "think_frac", "n_direct", "n_programs", "flag_frac",
         "repair_frac", "raw_frac", "t_hint_agree", "unans", "zero", "empty", "n_valid", "n_samples", "n_groups", "round2",
         "lp_mean"]
_RAW = re.compile(r"\bC\[|\bT\[|\.groupby\(|\.nunique\(|\bpd\.")


def _key(ans):
    import agent
    return agent.canon_of(ans)


def groups(rec):
    lg = rec.get("samples") if isinstance(rec.get("samples"), dict) else {}
    by = defaultdict(list); rep = {}
    for s in lg.get("samples") or []:
        if not s.get("ok"): continue
        k = _key(s["answer"]); by[k].append(s); rep.setdefault(k, s["answer"])
    return [(rep[k], v) for k, v in by.items()]


def _norm_code(c): return re.sub(r"\s+", "", c or "")


def group_features(rec, ans, smp, allg):
    lg = rec.get("samples") if isinstance(rec.get("samples"), dict) else {}
    S = lg.get("samples") or []; h = lg.get("hints") or {}
    n = max(1, len(S)); valid = sum(len(g[1]) for g in allg)
    w = lambda ss: sum(0.5 if s.get("flags") else 1.0 for s in ss)
    shares = sorted((w(g[1]) for g in allg), reverse=True)
    mine = w(smp)
    t = (ans or {}).get("type"); v = (ans or {}).get("answer")
    lps = [s.get("lp") for s in smp if isinstance(s.get("lp"), (int, float))]
    f = {"share": len(smp) / n, "share_w": mine / n, "rank0": float(mine >= shares[0]), "gap_top": (shares[0] - mine) / n,
         "n_support": len(smp), "n_think": sum(1 for s in smp if s.get("variant") == "think"),
         "n_direct": sum(1 for s in smp if s.get("variant") != "think"),
         "n_programs": len({_norm_code((s.get("turns") or [{}])[-1].get("code")) for s in smp}),
         "flag_frac": sum(1 for s in smp if s.get("flags")) / len(smp),
         "repair_frac": sum(1 for s in smp if len(s.get("turns") or []) > 1) / len(smp),
         "raw_frac": sum(1 for s in smp if _RAW.search((s.get("turns") or [{}])[-1].get("code") or "")) / len(smp),
         "t_hint_agree": float(bool(h.get("answer_type")) and t == h.get("answer_type")), "unans": float(t == "unanswerable"),
         "zero": float(isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0), "empty": float(v in ([], "", None) and t != "unanswerable"),
         "n_valid": valid, "n_samples": len(S), "n_groups": len(allg), "round2": float(len(S) > 2),
         "lp_mean": float(np.mean(lps)) if lps else 0.0}
    f["think_frac"] = f["n_think"] / len(smp)
    return np.array([f[k] for k in FEATS], dtype=float)


class Selector:
    def __init__(self, l2=1.0): self.l2, self.w, self.mu, self.sd, self.theta = l2, None, None, None, 0.5

    def _z(self, X): return (X - self.mu) / self.sd

    def fit(self, records, refs, judge, iters=400, lr=0.5):
        X, y = [], []
        for r in records:
            ref = refs.get(r["id"].split("~")[0])
            if ref is None: continue
            G = groups(r)
            for ans, smp in G:
                try: ok = judge(ans, ref) == "correct"
                except Exception: ok = False
                X.append(group_features(r, ans, smp, G)); y.append(float(ok))
        X = np.array(X); y = np.array(y)
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-6
        Z = np.c_[np.ones(len(X)), self._z(X)]; w = np.zeros(Z.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-Z @ w)); g = Z.T @ (p - y) / len(y) + self.l2 * np.r_[0, w[1:]] / len(y)
            w -= lr * g
        self.w = w
        return self

    def score(self, rec):
        G = groups(rec)
        if not G: return []
        X = np.array([group_features(rec, a, s, G) for a, s in G])
        p = 1 / (1 + np.exp(-(np.c_[np.ones(len(X)), self._z(X)] @ self.w)))
        return sorted(zip(p.tolist(), [a for a, _ in G]), key=lambda x: -x[0])

    def decide(self, rec, theta=None):
        sc = self.score(rec)
        th = self.theta if theta is None else theta
        if not sc or sc[0][0] < th: return {"type": "abstain"}, (sc[0][0] if sc else 0.0)
        return sc[0][1], sc[0][0]

    def rerank(self, rec, theta_vote=0.5):
        """the selector chooses the group; the abstain decision stays the flow's own (vote share of that group >= theta_vote):
        robust when the training logs are easier than the test (in-sample training questions)."""
        G = groups(rec)
        if not G: return {"type": "abstain"}
        lg = rec.get("samples") if isinstance(rec.get("samples"), dict) else {}
        n = max(1, len(lg.get("samples") or []))
        X = np.array([group_features(rec, a, s, G) for a, s in G])
        p = 1 / (1 + np.exp(-(np.c_[np.ones(len(X)), self._z(X)] @ self.w)))
        k = int(np.argmax(p)); share = sum(0.5 if s.get("flags") else 1.0 for s in G[k][1]) / n
        return G[k][0] if share >= theta_vote else {"type": "abstain"}

    def save(self, path):
        json.dump({"feats": FEATS, "w": self.w.tolist(), "mu": self.mu.tolist(), "sd": self.sd.tolist(), "theta": self.theta, "l2": self.l2},
                  open(path, "w"), indent=1)

    @classmethod
    def load(cls, path):
        d = json.load(open(path)); s = cls(d.get("l2", 1.0))
        s.w, s.mu, s.sd, s.theta = np.array(d["w"]), np.array(d["mu"]), np.array(d["sd"]), d["theta"]
        return s
