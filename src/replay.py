#!/usr/bin/env python3
"""Offline replay of a v3 run on CPU (no LLM, no GPU): compare mapping-selection and voting strategies.

1. Mapping candidates of every business: the LLM mapping programs logged in compile_logs.jsonl (last ok attempt of each
   sample) and the reference programs whose tables resolve (leave-one-out on a split with answers). Each is executed
   and checked again; the canonical tables are pickled to <work>/canon/.
2. Programs of every question: the last ok attempt of each LLM sample in logs.jsonl + the rule-based template program.
   Each distinct program is executed on each of the top-M candidates.
3. Strategies (scored against answers.jsonl when present):
     run          the run's own final answers
     top1         candidates ranked like the run (coverage, then agreement); vote on the first
     agree1       ranked by signature agreement only
     voteM        votes of the top-M candidates added (each candidate weight 1/M)
     oracle       best candidate per business (upper bound of mapping selection)
Usage: replay.py <split_dir> <run_dir> <cache_dir> <work_dir> [--m 3] [--qw 2] [--tau 0.5] [--biz REGEX] [--workers 4]"""
import os, re, sys, json, pickle, argparse, time
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent, canon as canonmod, score


def load_jsonl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.isfile(p) else []


def candidates(split, run_dir, bids, exclude_own):
    """[(bid, src, code, rid)] for every mapping candidate."""
    progs, meta = canonmod.ref_programs(), canonmod.load_meta()
    out, llm = [], defaultdict(dict)
    for r in load_jsonl(os.path.join(run_dir, "compile_logs.jsonl")):
        for s, atts in (r.get("llm_attempts") or {}).items():
            ok = agent.last_ok(atts)
            if ok and ok.get("code"): llm[r["business_id"]][f"llm:{s}"] = ok["code"]
    for b in bids:
        for rid, code in progs.items():
            out.append((b, "ref:" + rid, code, rid))
        for src, code in sorted(llm[b].items()): out.append((b, src, code, None))
    return [(b, s, c, r) for b, s, c, r in out if not (exclude_own and r == b)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("run"); ap.add_argument("cache"); ap.add_argument("work")
    ap.add_argument("--m", type=int, default=3); ap.add_argument("--qw", type=float, default=2.0); ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--biz", default=""); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--timeout", type=int, default=60)
    a = ap.parse_args(); t0 = time.time()
    os.makedirs(os.path.join(a.work, "canon"), exist_ok=True)
    logs = [r for r in load_jsonl(os.path.join(a.run, "logs.jsonl")) if not a.biz or re.search(a.biz, r["business_id"])]
    bids = sorted({r["business_id"] for r in logs})
    refs = score.load(os.path.join(a.split, "answers.jsonl")) if os.path.isfile(os.path.join(a.split, "answers.jsonl")) else {}
    import multiprocessing as mp
    pool = mp.get_context("fork").Pool(a.workers, initializer=agent._init_worker, initargs=(a.cache, a.split), maxtasksperchild=200)

    # 1. mapping candidates (references are kept only if their tables resolve, as in the run)
    meta = canonmod.load_meta(); tasks = []
    for b, src, code, rid in candidates(a.split, a.run, bids, bool(refs)):
        if rid is not None:
            biz = agent.load_biz(a.cache, a.split, b)
            if not canonmod.resolvable(biz, "generic" if rid.startswith("fmt:") else meta.get(rid, {})): continue
        tasks.append(((b, src), b, code, rid, os.path.join(a.work, "canon", f"{b}.{src.replace(':', '_')}.pkl")))
    cands = defaultdict(list)
    for (b, src), r in agent.pmap(pool, agent.run_mapping, tasks):
        if r.get("ok"): cands[b].append((src, r))
    rank = {}
    for b in bids:
        cs = sorted(cands[b], key=lambda x: x[0])
        if not cs: rank[b] = {"top1": [], "agree1": []}; continue
        _, sc = agent.agreement([r["sig"] for _, r in cs])
        rank[b] = {"top1": [cs[i] for i in sorted(range(len(cs)), key=lambda i: (-cs[i][1]["coverage"], -sc[i], i))],
                   "agree1": [cs[i] for i in sorted(range(len(cs)), key=lambda i: (-sc[i], i))]}
        print(f"{b}: {len(cs)} candidates; top1 {[s for s, _ in rank[b]['top1'][:a.m]]} agree1 {[s for s, _ in rank[b]['agree1'][:a.m]]}", flush=True)
    print(f"[replay] mappings {time.time() - t0:.0f}s", flush=True)

    # 2. execute every distinct program of every question on the candidates that any strategy can pick
    use = {b: {s: r for s, r in (rank[b]["top1"][:a.m] + rank[b]["agree1"][:1] + rank[b]["top1"])} for b in bids}
    jobs, progs = [], {}
    for rec in logs:
        b = rec["business_id"]; ps = []
        for name, att in rec["samples"].items():
            if name == "escalated": continue                            # list of reasons, not a sample
            r = att if name == "rule" else agent.last_ok(att)
            if r and r.get("ok") and r.get("code"): ps.append((name, r["code"]))
        progs[rec["id"]] = ps
        for code in {c for _, c in ps}:
            for src, cr in use[b].items():
                jobs.append(((rec["id"], src, code), b, rec["as_of_date"], code, a.timeout, cr["path"]))
    print(f"[replay] {len(jobs)} executions", flush=True)
    res = {}
    for k, r in agent.pmap(pool, agent.run_code, jobs, per_task=a.timeout):
        res[k] = agent.canon_of(r["answer"]) if r.get("ok") else None
        if r.get("ok"): res[("A",) + k] = r["answer"]
    pool.close()
    print(f"[replay] executed {time.time() - t0:.0f}s", flush=True)

    # 3. strategies
    def vote(qid, srcs, weights):
        v, rep = Counter(), {}
        for src, w in zip(srcs, weights):
            for name, code in progs[qid]:
                c = res.get((qid, src, code))
                if c is not None: v[c] += w * (a.qw if name == "rule" else 1.0); rep.setdefault(c, res[("A", qid, src, code)])
        n_llm = sum(1 for n, _ in progs[qid] if n != "rule")
        total = sum(weights) * (n_llm + a.qw) if srcs else 0
        if not v or not total: return {"type": "abstain"}
        top, n = v.most_common(1)[0]
        return rep[top] if n / total >= a.tau else {"type": "abstain"}

    def tally(preds):
        c = Counter()
        for qid, p in preds.items():
            c["abstain" if p.get("type") == "abstain" else ("correct" if score.judge(p, refs[qid]) == "correct" else "wrong")] += 1
        n = sum(c.values()); return {"PA": round((c["correct"] - c["wrong"]) / max(n, 1), 4), **c}

    strat = {"run": {r["id"]: r["final"] for r in logs}}
    by_b = defaultdict(list)
    for r in logs: by_b[r["business_id"]].append(r["id"])
    for name, pick in [("top1", lambda b: rank[b]["top1"][:1]), ("agree1", lambda b: rank[b]["agree1"][:1])] + \
                      [(f"vote{m}", (lambda m: lambda b: rank[b]["top1"][:m])(m)) for m in range(2, a.m + 1)]:
        strat[name] = {}
        for b, qids in by_b.items():
            cs = pick(b)
            for qid in qids: strat[name][qid] = vote(qid, [s for s, _ in cs], [1.0 / max(len(cs), 1)] * len(cs))
    if refs:
        strat["oracle"] = {}
        for b, qids in by_b.items():
            best = max(rank[b]["top1"] or [(None, None)], key=lambda x: tally({q: vote(q, [x[0]], [1.0]) for q in qids})["PA"] if x[0] else -9)
            for qid in qids: strat["oracle"][qid] = vote(qid, [best[0]], [1.0]) if best[0] else {"type": "abstain"}
        out = {k: tally(v) for k, v in strat.items()}
        for k, v in out.items(): print(f"{k:8s} {v}", flush=True)
        per_b = {b: {k: tally({q: strat[k][q] for q in qids})["PA"] for k in strat} for b, qids in by_b.items()}
        for b in sorted(per_b): print(f"  {b} " + " ".join(f"{k}={v:+.2f}" for k, v in per_b[b].items()), flush=True)
        json.dump({"overall": out, "per_business": per_b}, open(os.path.join(a.work, "replay_score.json"), "w"), indent=1)
    for k, v in strat.items():
        with open(os.path.join(a.work, f"submission_{k}.jsonl"), "w", encoding="utf-8") as f:
            for r in logs: f.write(json.dumps({"id": r["id"], **v[r["id"]]}, ensure_ascii=False) + "\n")
    print(f"[replay] done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
