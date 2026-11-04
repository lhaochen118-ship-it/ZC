#!/usr/bin/env python3
"""Consensus re-answer over several finished runs (CPU only; after the GPU runs, in docker/predict.sh or offline).

Why: each run chooses its mapping among ITS OWN samples, so the runs of one system disagree by luck (public v24: dev_014
seed 1 chose a mapping without the instalment book, dev_013 seed 1 one that copied the whole table's voucher total to every
order) and the vote splits or follows a wrong majority. Here the mapping of every business is chosen ONCE among the candidate
mappings of ALL runs (pooled), with the current soft checks and the rule programs, and every logged program of every run is
executed again on that mapping, then voted as the pipeline does:

 1. candidates of every run (LLM samples: last ok attempt; --ref-adapt rewrites; reference transfers named in the run's
    candidate list) are rebuilt with the current src/ (coverage, warnings, signature) -> pooled list per business;
 2. the rule programs of the business's questions (current qparse, cleaned text) run on every candidate: fail share and
    answers; the ranking is Pipeline.choose's (coverage - warning penalties - 30 x rule failures + 30 x answer agreement,
    signature tie-break) over the pooled list -> best and runner-up;
 3. per run and question: the rule program (re-parsed on the best mapping) + every logged LLM sample (escalations and raw
    samples included; samples the run logged as ungrounded stay out) are executed on the best mapping; the weighted vote
    (rule 4, industry / new templates 2, subjective 4, samples 1; tau 0.5); unanswerable / abstain -> the same programs on
    the runner-up (map fallback); unanswerable on an LLM mapping that left raw tables unused -> abstain (not for the
    subjective rule);
 4. the runs' new answers are voted across runs (vote.py, min share 0.4).
Writes <out_dir>/run_<i>/logs.jsonl (per run, the redone answers) and <out_submission>; prints per business the pooled choice
and how many runs had chosen a mapping of the same rule-answer cluster.
Usage: consensus.py <split> <ingest_cache> <work_dir> <out_submission> <run_dir>... [--biz REGEX] [--workers N]"""
import os, re, sys, json, glob, time, argparse, collections, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent, canon as canonmod, qnorm, pipeline, vote as votemod

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refs", "compile")
FMT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refs", "formats")
SKIP = ("escalated", "ungrounded", "fallback", "unans_abstain", "rule")


def load_jsonl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.isfile(p) else []


def run_compile_logs(d):
    files = sorted(glob.glob(os.path.join(d, "shard*", "compile_logs.jsonl"))) or [os.path.join(d, "compile_logs.jsonl")]
    return [r for f in files for r in load_jsonl(f)]


def candidates(d):
    """{business: {src: (code, rid)}}, {business: chosen src}, {business: src of the reference an adapt round rewrote}."""
    cands, chosen, gaps = collections.defaultdict(dict), {}, {}
    for r in run_compile_logs(d):
        b = r["business_id"]
        for s, atts in (r.get("llm_attempts") or {}).items():
            ok = agent.last_ok(atts)
            if ok and ok.get("code"): cands[b][f"llm:{s}"] = (ok["code"], None)
        for s, atts in (r.get("adapt_attempts") or {}).items():
            ok = agent.last_ok(atts)
            if ok and ok.get("code"): cands[b][f"adapt:{str(r.get('adapt_of', ''))[4:]}:{s}"] = (ok["code"], None)
        if r.get("adapt_of"): gaps[b] = r["adapt_of"]
        if "chosen" in r:
            chosen[b] = r["chosen"]
            for c in r["candidates"]:
                src = str(c[0])
                p = os.path.join(FMT, src[8:] + ".py") if src.startswith("ref:fmt:") else os.path.join(REF, src[4:] + ".py") if src.startswith("ref:") else None
                if p and os.path.isfile(p): cands[b][src] = (open(p, encoding="utf-8").read(), src[4:])
    return cands, chosen, gaps


def rule_texts(split, b, qs):
    """(qid, as_of, text the rule reader parses) with the pipeline's question cleaning."""
    try: nz = qnorm.Normalizer.for_split(split, exclude_business=b, extra_texts=qnorm.Normalizer.entity_texts(agent.load_biz(agent._CTX["cache"], split, b)))
    except Exception: nz = None
    out = []
    for q in qs:
        c = None
        if nz is not None:
            try: c = nz.clean(q["question"], q["as_of_date"])
            except Exception: c = None
        out.append((q["id"], q["as_of_date"], (c or {}).get("clean") or q["question"]))
    return out


def rank(pool_b):
    """Pipeline.choose's order over [(src, rec)]: rec has coverage, warnings, sig, fail_share, rule_answers, gaps."""
    if not pool_b: return []
    _, sc = agent.agreement([r["sig"] for _, r in pool_b])
    ag = pipeline.Pipeline.answer_agreement([r.get("rule_answers") or {} for _, r in pool_b])
    W, F = 30.0, pipeline.Pipeline.SEL_FAIL_W
    return sorted(range(len(pool_b)), key=lambda i: (-(canonmod.rank_score(pool_b[i][1]) - F * pool_b[i][1].get("fail_share", 0.0) + W * ag[i]
                                                     - (pipeline.Pipeline.REF_GAP_W if pool_b[i][1].get("gaps") else 0.0)), -sc[i], i))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("cache"); ap.add_argument("work"); ap.add_argument("out"); ap.add_argument("runs", nargs="+")
    ap.add_argument("--biz", default=""); ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--timeout", type=int, default=60)
    a = ap.parse_args(); t0 = time.time()
    os.makedirs(os.path.join(a.work, "canon"), exist_ok=True)
    agent._init_worker(a.cache, a.split)
    runs = [r for r in a.runs if os.path.isfile(os.path.join(r, "logs.jsonl"))]
    logs = [{x["id"]: x for x in load_jsonl(os.path.join(r, "logs.jsonl"))} for r in runs]
    meta = [candidates(r) for r in runs]
    qs_all = [json.loads(l) for l in open(os.path.join(a.split, "questions.jsonl"), encoding="utf-8")]
    bids = sorted({q["business_id"] for q in qs_all if not a.biz or re.search(a.biz, q["business_id"])})
    pool = mp.get_context("fork").Pool(a.workers, initializer=agent._init_worker, initargs=(a.cache, a.split), maxtasksperchild=100)
    # 1. rebuild every distinct candidate (a reference once; an LLM sample per run)
    tasks, owners = [], collections.defaultdict(list)
    for i, (cands, chosen, gaps) in enumerate(meta):
        for b in bids:
            for src, (code, rid) in cands.get(b, {}).items():
                key = (b, src if src.startswith("ref:") else f"r{i}:{src}"); owners[key].append(i)
                if len(owners[key]) > 1: continue
                tasks.append((key, b, code, rid, os.path.join(a.work, "canon", f"{b}.{key[1].replace(':', '_')}.pkl")))
    built = {}
    for key, r in agent.pmap(pool, agent.run_mapping, tasks):
        if r.get("ok"): built[key] = r
    print(f"[consensus] {len(built)}/{len(tasks)} candidate mappings rebuilt ({time.time() - t0:.0f}s)", flush=True)
    # 2. rule programs on every candidate -> fail share, answers; pooled ranking
    rq = {b: rule_texts(a.split, b, [q for q in qs_all if q["business_id"] == b]) for b in bids}
    evs = [((k, "eval"), k[0], r["path"], rq[k[0]], 150) for k, r in built.items()]
    for (k, _), res in agent.pmap(pool, agent.run_rule_eval, evs):
        built[k]["fail_share"] = res.get("fail_share", 0.0) if res.get("ok") else 0.0
        built[k]["rule_answers"] = res.get("answers") or {} if res.get("ok") else {}
    best = {}
    for b in bids:
        pb = sorted([(k[1], r) for k, r in built.items() if k[0] == b], key=lambda x: (not x[0].startswith("ref:"), x[0]))
        for src, r in pb: r["gaps"] = any(meta[i][2].get(b) == src for i in range(len(meta)))   # a reference an adapt round rewrote
        order = rank(pb)
        if not order: continue
        best[b] = [pb[j] for j in order[:2]]
        sig = lambda r: json.dumps(sorted((r.get("rule_answers") or {}).items()))
        top = sig(best[b][0][1])
        agree_runs = sum(1 for i, (c, ch, g) in enumerate(meta) if b in ch and
                         (built.get((b, ch[b] if ch[b].startswith("ref:") else f"r{i}:{ch[b]}")) or {}).get("rule_answers") is not None and
                         sig(built[(b, ch[b] if ch[b].startswith("ref:") else f"r{i}:{ch[b]}")]) == top)
        pen = sum(canonmod.warning_penalty(w) for w in best[b][0][1].get("warnings") or [])
        print(f"  {b}: {len(pb)} pooled candidates -> {best[b][0][0]} (penalty {pen:.0f}); {agree_runs}/{len(runs)} runs had chosen that cluster", flush=True)
    # 3. every program of every run on the pooled best (and runner-up) mapping
    jobs = []
    for i, L in enumerate(logs):
        for qid, rec in L.items():
            b = rec["business_id"]
            if b not in best: continue
            for j, (src, r) in enumerate(best[b]):
                for name, v in rec["samples"].items():
                    if name in SKIP or not isinstance(v, list) or not all(isinstance(x, dict) for x in v): continue
                    ok = agent.last_ok(v)
                    if ok and ok.get("code"): jobs.append(((i, qid, j, name), b, rec["as_of_date"], ok["code"], a.timeout, r["path"]))
    rule_jobs = []
    rtext = {qid: t for b in rq for qid, _, t in rq[b]}
    for qid in {q for L in logs for q in L}:
        b = next(L[qid]["business_id"] for L in logs if qid in L)
        if b not in best: continue
        as_of = next(L[qid]["as_of_date"] for L in logs if qid in L)
        for j, (src, r) in enumerate(best[b]): rule_jobs.append(((qid, j), b, as_of, rtext.get(qid, ""), r["path"], a.timeout))
    res = dict(agent.pmap(pool, agent.run_code, jobs, per_task=a.timeout))
    rres = dict(agent.pmap(pool, agent.run_qparse, rule_jobs, per_task=a.timeout))
    pool.close()
    print(f"[consensus] {len(jobs)} sample programs + {len(rule_jobs)} rule programs executed ({time.time() - t0:.0f}s)", flush=True)
    NEW, WN = pipeline.Pipeline.NEW_RULES, pipeline.Pipeline.NEW_RULE_W

    def vote_q(i, qid, rec, j):
        votes, rep, total = collections.Counter(), {}, 0.0
        rr = rres.get((qid, j)) or {}
        if rr.get("ok") and rr.get("answer") is not None:
            tpl = str(rr.get("tpl") or ""); w = WN if (tpl.startswith("ind_") or tpl in NEW) else 4.0
            c = agent.canon_of(rr["answer"]); votes[c] += w; rep.setdefault(c, rr["answer"]); total += w
        ung = set(rec["samples"].get("ungrounded") or [])
        for name, v in rec["samples"].items():
            if name in SKIP or not isinstance(v, list): continue
            sid = name.rstrip("LBE")
            if sid.isdigit() and int(sid) in ung: continue
            r = res.get((i, qid, j, name))
            if r is None: continue
            total += 1.0
            if r.get("ok") and r.get("answer") is not None:
                c = agent.canon_of(r["answer"]); votes[c] += 1.0; rep.setdefault(c, r["answer"])
        if not votes or not total: return {"type": "abstain"}, 0.0, str(rr.get("tpl") or "")
        top, n = votes.most_common(1)[0]
        return (rep[top] if n / total >= 0.5 else {"type": "abstain"}), n / total, str(rr.get("tpl") or "")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    new_dirs = []
    for i, L in enumerate(logs):
        d = os.path.join(a.work, f"run_{i}"); os.makedirs(d, exist_ok=True); new_dirs.append(d)
        with open(os.path.join(d, "logs.jsonl"), "w", encoding="utf-8") as f:
            for qid, rec in L.items():
                b = rec["business_id"]; rec = dict(rec)
                if b in best:
                    final, conf, tpl = vote_q(i, qid, rec, 0)
                    if final.get("type") in ("abstain", "unanswerable") and len(best[b]) > 1:
                        f2, c2, _ = vote_q(i, qid, rec, 1)
                        if f2.get("type") not in ("abstain", "unanswerable"): final, conf = f2, c2
                    if final.get("type") == "unanswerable" and tpl != "subjective" and "llm:" in best[b][0][0] and best[b][0][1].get("unused"):
                        final = {"type": "abstain"}                     # (pipeline --unans-unused)
                    rec["final_run"] = rec["final"]; rec["final"] = final; rec["conf"] = conf; rec["consensus_map"] = best[b][0][0]
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    out = votemod.vote([votemod.load(d) for d in new_dirs], 0.4, None)
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for q, (final, _) in out.items(): f.write(json.dumps({"id": q, **final}, ensure_ascii=False) + "\n")
    os.replace(tmp, a.out)
    n_ab = sum(1 for f_, _ in out.values() if f_.get("type") == "abstain")
    print(f"[consensus] {len(out)} answers, {n_ab} abstain -> {a.out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
