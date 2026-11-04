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
    signature tie-break) over the pooled list, the penalties of topic warnings (debt / instalment / vehicle / staff) left out -> best and
    runner-up; a question of a topic (a third of its programs use that topic's toolkit functions / columns) uses instead the best two
    candidates with no heavy warning (>= 10) on the topic, and abstains when every candidate has one;
 3. per run and question: the rule program (re-parsed on the question's mapping) + every logged LLM sample (escalations and
    raw samples included; samples the run logged as ungrounded stay out) are executed on that mapping; the weighted vote
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


def general_score(r):
    """canon.rank_score without the warnings of a topic (debt / instalment / vehicle / staff): the questions of a topic choose their own
    candidate (slots in main), so a debt-only fault must not move every revenue question to another mapping (public dev_014)."""
    return float(r.get("coverage") or 0) - sum(canonmod.warning_penalty(w) for w in r.get("warnings") or []
                                               if not any(rx.search(str(w)) for rx in TOPIC_WARN.values()))


def rank(pool_b):
    """Pipeline.choose's order over [(src, rec)]: rec has coverage, warnings, sig, fail_share, rule_answers, gaps."""
    if not pool_b: return []
    _, sc = agent.agreement([r["sig"] for _, r in pool_b])
    ag = pipeline.Pipeline.answer_agreement([r.get("rule_answers") or {} for _, r in pool_b])
    W, F = 30.0, pipeline.Pipeline.SEL_FAIL_W
    return sorted(range(len(pool_b)), key=lambda i: (-(general_score(pool_b[i][1]) - F * pool_b[i][1].get("fail_share", 0.0) + W * ag[i]
                                                     - (pipeline.Pipeline.REF_GAP_W if pool_b[i][1].get("gaps") else 0.0)), -sc[i], i))


# question topics whose answers depend on one part of the mapping: such a question is answered on the best candidate WITHOUT a
# heavy warning (penalty >= 10) on that part; no clean candidate -> abstain (a likely wrong answer costs a point, an abstain
# nothing). Public v23 runs: every dev_008 candidate ignored the debt notes (debts 0 or 233M for 4.5M), every dev_014 one
# had unpaid orders / no debt book (22.9 bn) while its revenue was right.
TOPIC_FN = {
    "debt": re.compile(r"\b(?:debts?|debt_\w+|n_debtors|top_debtors|paid_until|n_due_orders)\s*\("),      # book + instalments
    "instalment": re.compile(r"\b(?:instalment_\w+|terms_due|n_overdue_customers)\s*\("),                # the instalment book only
    "vehicle": re.compile(r"\b(?:n_vehicles|top_plates|vehicle_sales|top_vehicles)\s*\(|\bvehicle\s*=|\[\s*[\"'](?:plate|vehicle)[\"']\s*\]"),
    "staff": re.compile(r"\b(?:staff_sales|staff_stats|top_staff|commission)\s*\(|\bstaff\s*=|\[\s*[\"']staff[\"']\s*\]"),
}
TOPIC_WARN = {
    "debt": re.compile(r"C counts \d+ of \d+ orders .* as still owing|the notes \S+ say \d+ orders|the data has a debt book|"
                       r"the data has an instalment book|C\['instalments'\] lacks|C\['instalment_payments'\] lacks"),
    "instalment": re.compile(r"the data has an instalment book|C\['instalments'\] lacks|C\['instalment_payments'\] lacks"),
    "vehicle": re.compile(r"the data records number plates|the data writes the vehicle model"),
    "staff": re.compile(r"orders\.staff (?:holds given names|does not follow)"),
}


def question_topics(rec):
    """topics of a logged question: those whose toolkit functions / columns at least a third of its programs (rule + samples)
    use (raw-table samples read T, not C: public dev_012 'bao nhiêu chiếc xe Wave Alpha' had 2 of 12 programs on n_vehicles and 3
    with vehicle=)."""
    codes = []
    for name, v in (rec.get("samples") or {}).items():
        if name == "rule" and isinstance(v, dict) and v.get("code"): codes.append(str(v["code"]))
        elif name not in SKIP and isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            ok = agent.last_ok(v)
            if ok and ok.get("code"): codes.append(str(ok["code"]))
    if not codes: return set()
    return {t for t, rx in TOPIC_FN.items() if sum(1 for c in codes if rx.search(c)) >= len(codes) / 3}


def topic_bad(r, topics):
    return any(TOPIC_WARN[t].search(str(w)) and canonmod.warning_penalty(w) >= 10 for t in topics for w in r.get("warnings") or [])


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
        best[b] = [pb[j] for j in order]                                  # full pooled ranking
        sig = lambda r: json.dumps(sorted((r.get("rule_answers") or {}).items()))
        top = sig(best[b][0][1])
        agree_runs = sum(1 for i, (c, ch, g) in enumerate(meta) if b in ch and
                         (built.get((b, ch[b] if ch[b].startswith("ref:") else f"r{i}:{ch[b]}")) or {}).get("rule_answers") is not None and
                         sig(built[(b, ch[b] if ch[b].startswith("ref:") else f"r{i}:{ch[b]}")]) == top)
        pen = sum(canonmod.warning_penalty(w) for w in best[b][0][1].get("warnings") or [])
        print(f"  {b}: {len(pb)} pooled candidates -> {best[b][0][0]} (penalty {pen:.0f}); {agree_runs}/{len(runs)} runs had chosen that cluster", flush=True)
    # 3. every program of every run on the mappings the question uses: the pooled best and runner-up, or for a debt / vehicle /
    #    staff question the best two candidates without a heavy warning on that topic (none -> abstain)
    def slots(b, rec):
        T = question_topics(rec)
        if not T: return best[b][:2], T
        clean = [m for m in best[b] if not topic_bad(m[1], T)]
        return clean[:2], T
    qslots, qtop = {}, {}
    for L in logs:
        for qid, rec in L.items():
            b = rec["business_id"]
            if b in best and qid not in qslots: qslots[qid], qtop[qid] = slots(b, rec)
    jobs, uniq, alias = [], {}, {}
    for i, L in enumerate(logs):
        for qid, rec in L.items():
            b = rec["business_id"]
            if b not in best: continue
            for src, r in qslots[qid]:
                for name, v in rec["samples"].items():
                    if name in SKIP or not isinstance(v, list) or not all(isinstance(x, dict) for x in v): continue
                    ok = agent.last_ok(v)
                    if not (ok and ok.get("code")): continue
                    ck = (qid, src, ok["code"])                     # the same program logged by several runs runs once
                    if ck not in uniq: uniq[ck] = len(jobs); jobs.append(((len(jobs),), b, rec["as_of_date"], ok["code"], a.timeout, r["path"]))
                    alias[(i, qid, src, name)] = uniq[ck]
    rule_jobs = []
    rtext = {qid: t for b in rq for qid, _, t in rq[b]}
    for qid in qslots:
        b = next(L[qid]["business_id"] for L in logs if qid in L)
        as_of = next(L[qid]["as_of_date"] for L in logs if qid in L)
        for src, r in qslots[qid]: rule_jobs.append(((qid, src), b, as_of, rtext.get(qid, ""), r["path"], a.timeout))
    n_top = collections.Counter((next(L[q]["business_id"] for L in logs if q in L), "clean" if qslots[q] else "abstain")
                                for q in qslots if qtop[q])
    print(f"[consensus] topic questions (debt / instalment / vehicle / staff) per business: {dict(sorted(n_top.items()))}", flush=True)
    rj = dict(agent.pmap(pool, agent.run_code, jobs, per_task=a.timeout))
    res = {k: rj.get((j,)) for k, j in alias.items()}
    rres = dict(agent.pmap(pool, agent.run_qparse, rule_jobs, per_task=a.timeout))
    pool.close()
    print(f"[consensus] {len(jobs)} distinct sample programs ({len(alias)} logged) + {len(rule_jobs)} rule programs executed "
          f"({time.time() - t0:.0f}s)", flush=True)
    NEW, WN = pipeline.Pipeline.NEW_RULES, pipeline.Pipeline.NEW_RULE_W

    def vote_q(i, qid, rec, src):
        votes, rep, total = collections.Counter(), {}, 0.0
        rr = rres.get((qid, src)) or {}
        if rr.get("ok") and rr.get("answer") is not None:
            tpl = str(rr.get("tpl") or ""); w = WN if (tpl.startswith("ind_") or tpl in NEW) else 4.0
            c = agent.canon_of(rr["answer"]); votes[c] += w; rep.setdefault(c, rr["answer"]); total += w
        ung = set(rec["samples"].get("ungrounded") or [])
        for name, v in rec["samples"].items():
            if name in SKIP or not isinstance(v, list): continue
            sid = name.rstrip("LBE")
            if sid.isdigit() and int(sid) in ung: continue
            r = res.get((i, qid, src, name))
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
                    sl = qslots.get(qid) or []
                    if not sl:                                          # every candidate is flagged on the question's topic
                        final, conf, tpl = {"type": "abstain"}, 0.0, ""; rec["topic_abstain"] = sorted(qtop.get(qid) or [])
                    else:
                        final, conf, tpl = vote_q(i, qid, rec, sl[0][0])
                        if final.get("type") in ("abstain", "unanswerable") and len(sl) > 1:
                            f2, c2, _ = vote_q(i, qid, rec, sl[1][0])
                            if f2.get("type") not in ("abstain", "unanswerable"): final, conf = f2, c2
                        if final.get("type") == "unanswerable" and tpl != "subjective" and "llm:" in sl[0][0] and sl[0][1].get("unused"):
                            final = {"type": "abstain"}                 # (pipeline --unans-unused)
                    rec["final_run"] = rec["final"]; rec["final"] = final; rec["conf"] = conf
                    rec["consensus_map"] = sl[0][0] if sl else None; rec["topics"] = sorted(qtop.get(qid) or [])
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
