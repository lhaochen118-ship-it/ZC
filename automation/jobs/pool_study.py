#!/usr/bin/env python3
"""pool_study.py <split_dir> <ingest_cache> <work_dir> <results_dir>... [--biz REGEX] [--workers 3] [--w 30]

Offline study of the mapping choice: does choosing among the candidate mappings of SEVERAL runs (pooled) pick a better
mapping than each run choosing among its own? No LLM, no GPU.

1. Every candidate mapping each run had (LLM samples: last ok attempt; --ref-adapt rewrites; reference transfers named
   in the run's candidate list) is rebuilt with the current src/ (coverage, soft checks, signature).
2. The rule programs of the business's questions run on every candidate (fail share, answers; with answers.jsonl also
   correct / wrong against the labels).
3. Pipeline.choose is replayed (coverage - penalties - 30 x fail share + W x answer agreement, signature tie-break) on
   each run's own candidates and on the pool of all runs; the chosen candidates are compared: rule PA (labelled
   split), and the rule-answer cluster each choice falls into (cluster = identical rule answers).
Writes <work_dir>/pool_study.json (every candidate's scores + the choices)."""
import os, re, sys, json, glob, pickle, argparse, time, collections, multiprocessing as mp
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
import agent, canon as canonmod, score, dm, qparse, qnorm

REF = os.path.join(ROOT, "src", "refs", "compile"); FMT = os.path.join(ROOT, "src", "refs", "formats")


def run_candidates(d):
    """{business: {src: code}} and {business: chosen src}, {business: src of the reference an adapt round rewrote} of one
    results dir (shard*/compile_logs.jsonl, else compile_logs.jsonl)."""
    files = sorted(glob.glob(os.path.join(d, "shard*", "compile_logs.jsonl"))) or [os.path.join(d, "compile_logs.jsonl")]
    cands, chosen, gaps = collections.defaultdict(dict), {}, {}
    for f in files:
        if not os.path.isfile(f): continue
        for l in open(f, encoding="utf-8"):
            r = json.loads(l); b = r["business_id"]
            for s, atts in (r.get("llm_attempts") or {}).items():
                ok = agent.last_ok(atts)
                if ok and ok.get("code"): cands[b][f"llm:{s}"] = ok["code"]
            for s, atts in (r.get("adapt_attempts") or {}).items():
                ok = agent.last_ok(atts)
                if ok and ok.get("code"): cands[b][f"adapt:{str(r.get('adapt_of', ''))[4:]}:{s}"] = ok["code"]
            if r.get("adapt_of"): gaps[b] = r["adapt_of"]
            if "chosen" in r:
                chosen[b] = r["chosen"]
                for c in r["candidates"]:
                    src = str(c[0])
                    if src.startswith("ref:fmt:"): p = os.path.join(FMT, src[8:] + ".py")
                    elif src.startswith("ref:"): p = os.path.join(REF, src[4:] + ".py")
                    else: continue
                    if os.path.isfile(p): cands[b][src] = open(p, encoding="utf-8").read()
    return cands, chosen, gaps


def rule_job(t):
    """(key, b, pkl, [(qid, as_of, text)], refs{qid: ref}) -> (key, {qid: canon answer or None}, {qid: verdict})."""
    key, b, path, qs, refs = t
    ans, ver = {}, {}
    try: can = pickle.load(open(path, "rb"))
    except Exception: return key, ans, ver
    meta = {"data_end": can.get("data_end")}
    for qid, as_of, text in qs:
        shop = dm.Shop(can["C"], as_of, meta)
        try: pr = qparse.parse(text, as_of, shop)
        except Exception: pr = None
        if not pr: continue
        try:
            agent.arm(20)
            try: a = agent.fmt_answer(dm.run_program(pr[1], shop, as_of))
            finally: agent.disarm()
            bad = a.get("type") == "unanswerable" or a.get("answer") in (0, 0.0, [], None)
            ans[qid] = None if bad else repr(agent.canon_of(a))
            if qid in refs: ver[qid] = score.judge(a, refs[qid])
        except BaseException:
            ans[qid] = None
            if qid in refs: ver[qid] = "error"
    return key, ans, ver


def choose(cs, W):
    """cs: [(src, rec)] -> index chosen by Pipeline.choose's ranking (rec: coverage, warnings, fail_share, answers, sig, gaps)."""
    if not cs: return None
    _, sc = agent.agreement([r["sig"] for _, r in cs])
    import pipeline
    ag = pipeline.Pipeline.answer_agreement([r["answers"] for _, r in cs])
    def key(i):
        r = cs[i][1]
        return (-(canonmod.rank_score(r) - 30.0 * r["fail_share"] + W * ag[i] - (5.0 if r.get("gaps") else 0.0)), -sc[i], i)
    return min(range(len(cs)), key=key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("cache"); ap.add_argument("work"); ap.add_argument("runs", nargs="+")
    ap.add_argument("--biz", default=""); ap.add_argument("--workers", type=int, default=3); ap.add_argument("--w", type=float, default=30.0)
    a = ap.parse_args(); t0 = time.time()
    os.makedirs(os.path.join(a.work, "canon"), exist_ok=True)
    refs = score.load(os.path.join(a.split, "answers.jsonl")) if os.path.isfile(os.path.join(a.split, "answers.jsonl")) else {}
    qs_all = [json.loads(l) for l in open(os.path.join(a.split, "questions.jsonl"), encoding="utf-8")]
    runs = [run_candidates(d) for d in a.runs]
    bids = sorted({b for c, ch, g in runs for b in ch if not a.biz or re.search(a.biz, b)})
    # 1. rebuild every distinct candidate (references once; LLM samples per run)
    tasks, owner = [], {}
    for i, (cands, chosen, gaps) in enumerate(runs):
        for b in bids:
            for src, code in cands.get(b, {}).items():
                key = (b, src if src.startswith("ref:") else f"r{i}:{src}")
                if key in owner: owner[key].append(i); continue
                owner[key] = [i]; rid = src[4:] if src.startswith("ref:") else None
                tasks.append((key, b, code, rid, os.path.join(a.work, "canon", f"{b}.{key[1].replace(':', '_')}.pkl")))
    pool = mp.get_context("fork").Pool(a.workers, initializer=agent._init_worker, initargs=(a.cache, a.split), maxtasksperchild=50)
    built = {}
    for key, r in agent.pmap(pool, agent.run_mapping, tasks):
        if r.get("ok"): built[key] = {"coverage": r["coverage"], "warnings": [str(w)[:160] for w in r.get("warnings") or []],
                                      "sig": r["sig"], "path": r["path"]}
    print(f"[pool] {len(built)}/{len(tasks)} candidates rebuilt ({time.time() - t0:.0f}s)", flush=True)
    # 2. rule programs on every candidate (cleaned question text as in the pipeline)
    rq = {}
    for b in bids:
        biz = agent.load_biz(a.cache, a.split, b)
        try: nz = qnorm.Normalizer.for_split(a.split, exclude_business=b, extra_texts=qnorm.Normalizer.entity_texts(biz))
        except Exception: nz = None
        out = []
        for q in qs_all:
            if q["business_id"] != b: continue
            c = None
            if nz is not None:
                try: c = nz.clean(q["question"], q["as_of_date"])
                except Exception: c = None
            out.append((q["id"], q["as_of_date"], (c or {}).get("clean") or q["question"]))
        rq[b] = out
    jobs = [(key, key[0], v["path"], rq[key[0]], {q: refs[q] for q, _, _ in rq[key[0]] if q in refs}) for key, v in built.items()]
    for key, ans, ver in pool.imap_unordered(rule_job, jobs):
        n = len(ans); built[key].update(answers=ans, fail_share=sum(1 for x in ans.values() if x is None) / max(n, 1),
                                         parsed=n, correct=sum(1 for x in ver.values() if x == "correct"), nlab=len(ver))
    pool.close()
    print(f"[pool] rule programs done ({time.time() - t0:.0f}s)", flush=True)
    # 3. choices
    report = {"runs": a.runs, "w": a.w, "business": {}}
    tot = collections.Counter()
    for b in bids:
        mine = {i: [] for i in range(len(runs))}
        for key, v in built.items():
            if key[0] != b: continue
            v["gaps"] = any(key[1] == runs[i][2].get(b) for i in owner[key])
            for i in owner[key]: mine[i].append((key[1], v))
        pooled = sorted({k: v for i in mine for k, v in mine[i]}.items(), key=lambda x: (not x[0].startswith("ref:"), x[0]))
        for i in mine: mine[i].sort(key=lambda x: (not x[0].startswith("ref:"), x[0]))
        # clusters by identical rule answers
        cl, cid = {}, {}
        for src, v in pooled:
            sig = json.dumps(sorted(v["answers"].items()))
            cid[src] = cl.setdefault(sig, len(cl))
        size = collections.Counter(cid.values())
        def desc(src, v):
            pa = (2 * v["correct"] - v["nlab"]) / max(v["nlab"], 1) if v["nlab"] else None
            return {"src": src, "cluster": cid[src], "cl_size": size[cid[src]], "rank": round(canonmod.rank_score(v), 1),
                    "fail": round(v["fail_share"], 3), "rule_pa": None if pa is None else round(pa, 3), "warn": len(v["warnings"])}
        row = {"n_cands": len(pooled), "clusters": len(cl), "runs": [], "logged": [runs[i][1].get(b) for i in range(len(runs))]}
        for i in mine:
            j = choose(mine[i], a.w); row["runs"].append(desc(*mine[i][j]) if j is not None else None)
        j = choose(pooled, a.w); row["pooled"] = desc(*pooled[j]) if j is not None else None
        best = max(pooled, key=lambda x: x[1]["correct"]) if pooled and refs else None
        row["oracle"] = desc(*best) if best else None
        row["all"] = [desc(s, v) for s, v in pooled]
        report["business"][b] = row
        rs = [x for x in row["runs"] if x]
        if refs and rs and row["pooled"] is not None:
            nl = max(1, pooled[0][1]["nlab"])
            tot["runs_pa_x"] += sum(x["rule_pa"] or 0 for x in rs) / len(rs) * nl; tot["pooled_pa_x"] += (row["pooled"]["rule_pa"] or 0) * nl
            tot["oracle_pa_x"] += (row["oracle"]["rule_pa"] or 0) * nl; tot["nl"] += nl
        print(f"{b}: {len(pooled)} cands, {len(cl)} clusters | runs " +
              " ".join(f"{x['src']}(c{x['cluster']}/{x['cl_size']} pa {x['rule_pa']})" if x else "-" for x in row["runs"]) +
              f" | pooled {row['pooled']['src'] if row['pooled'] else '-'}(c{row['pooled']['cluster'] if row['pooled'] else '-'} pa {row['pooled']['rule_pa'] if row['pooled'] else '-'})" +
              (f" | oracle pa {row['oracle']['rule_pa']}" if row["oracle"] else ""), flush=True)
    if tot["nl"]:
        print(f"[pool] labelled rule PA (question-weighted): runs mean {tot['runs_pa_x'] / tot['nl']:.4f} | pooled {tot['pooled_pa_x'] / tot['nl']:.4f}"
              f" | oracle {tot['oracle_pa_x'] / tot['nl']:.4f}", flush=True)
    json.dump(report, open(os.path.join(a.work, "pool_study.json"), "w"), ensure_ascii=False, indent=0, default=str)
    print(f"[pool] {time.time() - t0:.0f}s -> {a.work}/pool_study.json", flush=True)


if __name__ == "__main__":
    main()
