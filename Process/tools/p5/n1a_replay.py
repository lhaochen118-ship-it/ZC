#!/usr/bin/env python3
"""New7 N1-A (metamorphic data testing, permutation invariance) — offline replay on a run's logged candidates, CPU only.

Every distinct (question, program) candidate of the solver log is executed again (agent.run_code, the same sandbox / toolkit /
model-written units as the run) on the shop's canonical tables and on a copy whose rows are shuffled (fixed seed, every table,
index reset). A candidate whose answer changes under the shuffle depends on row order (first/last of an unordered frame, iloc,
head without a sort): flagged. Policies replayed (vote rule of hybrid.finish: flagged samples dropped):
    base      the run's vote
    n1a_drop  vote without the flagged candidates (abstain if none left / share below theta)
Fidelity: share of candidates whose re-executed answer equals the logged one (canonical tables rebuilt offline must match the run).
Usage: n1a_replay.py <solver_logs.jsonl> <split_dir> <ingest_cache> <canon_dir> <units.jsonl|-> [--workers 4] [--out f.json]"""
import os, sys, json, pickle, argparse, random, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, "/home/user/ZC/src")

G = {}


def init(split, cache, units):
    import agent
    os.environ["ZC_HIDE_API"] = "industry"
    if units and units != "-": os.environ["ZC_UNITS"] = units
    agent._init_worker(cache, split); G["agent"] = agent


def job(t):
    qid, bid, as_of, code, p0, p1 = t
    ag = G["agent"]
    _, r0 = ag.run_code((qid, bid, as_of, code, 30, p0))
    _, r1 = ag.run_code((qid, bid, as_of, code, 30, p1))
    a0 = ag.canon_of(r0["answer"]) if r0.get("ok") else None
    a1 = ag.canon_of(r1["answer"]) if r1.get("ok") else None
    return qid, code, a0, a1


def shuffled(src, dst, seed=1103):
    d = pickle.load(open(src, "rb")); d["v2"] = True
    pickle.dump(d, open(dst[0], "wb"))
    rng = random.Random(seed); C2 = {}
    for k, df in d["C"].items():
        try:
            idx = list(range(len(df))); rng.shuffle(idx); C2[k] = df.iloc[idx].reset_index(drop=True)
        except Exception: C2[k] = df
    d2 = dict(d, C=C2); pickle.dump(d2, open(dst[1], "wb"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("logs"); ap.add_argument("split"); ap.add_argument("cache"); ap.add_argument("canon")
    ap.add_argument("units"); ap.add_argument("--workers", type=int, default=os.cpu_count() or 4); ap.add_argument("--out", default="")
    ap.add_argument("--theta", type=float, default=0.5); ap.add_argument("--gt", default="/home/user/ZC/data/raw/public_gt.jsonl")
    a = ap.parse_args(); t0 = time.time()
    import agent, score
    work = a.canon + "_n1a"; os.makedirs(work, exist_ok=True)
    rows = [json.loads(l) for l in open(a.logs, encoding="utf-8") if l.strip()]
    paths = {}
    for b in sorted({r["business_id"] for r in rows}):
        p = (os.path.join(work, b + ".orig.pkl"), os.path.join(work, b + ".perm.pkl"))
        if not os.path.isfile(p[1]): shuffled(os.path.join(a.canon, b + ".pkl"), p)
        paths[b] = p
    tasks, logged = {}, {}
    for r in rows:
        for s in (r.get("samples") or {}).get("samples") or []:
            code = next((t.get("code") for t in reversed(s.get("turns") or []) if t.get("code")), None)
            if not code or not s.get("ok"): continue
            k = (r["id"], code)
            logged[k] = agent.canon_of(s["answer"])
            tasks[k] = (r["id"], r["business_id"], r.get("as_of_date"), code, *paths[r["business_id"]])
    print(f"{len(rows)} questions, {len(tasks)} distinct candidates", flush=True)
    res = {}
    with ProcessPoolExecutor(a.workers, initializer=init, initargs=(a.split, a.cache, a.units)) as ex:
        for i, (qid, code, a0, a1) in enumerate(ex.map(job, tasks.values(), chunksize=8)):
            res[(qid, code)] = (a0, a1)
            if i % 1000 == 0: print(f"  {i}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    fid = sum(1 for k, (a0, _) in res.items() if a0 == logged[k]) / max(1, len(res))
    flagged = {k for k, (a0, a1) in res.items() if a0 is not None and a1 != a0}
    gt = score.load(a.gt); preds = {"base": {}, "n1a_drop": {}}
    nflag_q = 0
    for r in rows:
        smp = (r.get("samples") or {}).get("samples") or []
        for pol in preds:
            votes, rep, n = Counter(), {}, 0
            for s in smp:
                n += 1
                if not s.get("ok") or not isinstance(s.get("answer"), dict): continue
                code = next((t.get("code") for t in reversed(s.get("turns") or []) if t.get("code")), None)
                if pol == "n1a_drop" and (r["id"], code) in flagged: continue
                c = agent.canon_of(s["answer"]); votes[c] += 0.5 if s.get("flags") else 1.0; rep.setdefault(c, s["answer"])
            if votes:
                top, v = votes.most_common(1)[0]
                preds[pol][r["id"]] = rep[top] if v / max(1, n) >= a.theta else {"type": "abstain"}
            else: preds[pol][r["id"]] = {"type": "abstain"}
        nflag_q += any((r["id"], next((t.get("code") for t in reversed(s.get("turns") or []) if t.get("code")), None)) in flagged for s in smp)
    out = {"candidates": len(res), "fidelity": round(fid, 4), "flagged": len(flagged), "questions_with_flag": nflag_q, "s": round(time.time() - t0)}
    for pol, p in preds.items():
        rr, _ = score.score({k: v for k, v in gt.items() if k in p}, p); out[pol] = {k: rr.get(k) for k in ("PA", "correct", "wrong", "abstain")}
    print(json.dumps(out), flush=True)
    if a.out: json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
