#!/usr/bin/env python3
"""replay_biz.py <split> <ingest_cache> <work> <biz[,biz]> <results_dir> <out_results_dir>

Re-answer the questions of some businesses of a finished run with the CURRENT src/ (toolkit / finalize fixes), without the
LLM: the run's chosen mapping program (compile log) is executed again, every logged program of each question (rule program +
every LLM sample's last ok attempt, escalations and raw-table samples included) runs again on it, and the vote is redone with
the pipeline's weights (rule: 4, industry / new templates 2, samples 1; samples logged as ungrounded stay out; tau 0.5). The
other businesses are copied unchanged. Writes <out>/logs.jsonl (+ compile logs, for vote.py / map_penalties.py).
Self-check printed: on the replayed questions whose programs give the same answers as logged, the redone vote must equal
the logged final answer."""
import os, sys, json, glob, shutil, textwrap, pickle, collections, multiprocessing as mp
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
import agent, canon, pipeline

split, cache, work, bizs, src_dir, out_dir = sys.argv[1:7]
bizs = set(bizs.split(","))
os.makedirs(work, exist_ok=True); os.makedirs(out_dir, exist_ok=True)
logs = [json.loads(l) for l in open(os.path.join(src_dir, "logs.jsonl"), encoding="utf-8")]
clogs = [l for f in sorted(glob.glob(os.path.join(src_dir, "shard*", "compile_logs.jsonl")) or [os.path.join(src_dir, "compile_logs.jsonl")])
         for l in open(f, encoding="utf-8")]
att, chosen = {}, {}
for l in clogs:
    r = json.loads(l); b = r["business_id"]
    if r.get("llm_attempts"): att[b] = r["llm_attempts"]
    if "chosen" in r: chosen[b] = r["chosen"]
agent._init_worker(cache, split)
paths = {}
for b in bizs:
    ch = chosen.get(b)
    if not ch or not ch.startswith("llm:"): print(f"{b}: chosen {ch}: not an LLM mapping, kept"); continue
    code = agent.last_ok(att[b][ch[4:]])["code"]
    _, r = agent.run_mapping(((b, ch), b, code, None, os.path.join(work, f"{b}.{ch.replace(':', '_')}.pkl")))
    if not r.get("ok"): print(f"{b}: mapping failed now: {r.get('error')}"); continue
    paths[b] = r["path"]; print(f"{b}: {ch} rebuilt")


def progs(rec):
    out = []
    for name, v in rec["samples"].items():
        if name in ("escalated", "ungrounded", "fallback", "unans_abstain"): continue      # lists of reasons / sample ids
        if name == "rule" and isinstance(v, dict) and v.get("code"): out.append(("rule", v["code"], v.get("tpl")))
        elif isinstance(v, list) and all(isinstance(x, dict) for x in v):
            ok = agent.last_ok(v)
            if ok and ok.get("code"): out.append((name, ok["code"], None))
    return out


jobs = []
for rec in logs:
    if rec["business_id"] in paths:
        for name, code, tpl in progs(rec):
            jobs.append(((rec["id"], name), rec["business_id"], rec["as_of_date"], code, 60, paths[rec["business_id"]]))
with mp.get_context("fork").Pool(3, initializer=agent._init_worker, initargs=(cache, split)) as pool:
    res = dict(agent.pmap(pool, agent.run_code, jobs, per_task=60))
W_NEW = pipeline.Pipeline.NEW_RULE_W; NEW = pipeline.Pipeline.NEW_RULES
same_n = same_ok = changed = 0
out = []
for rec in logs:
    b = rec["business_id"]
    if b not in paths: out.append(rec); continue
    ung = set(rec["samples"].get("ungrounded") or [])
    votes, rep, total, any_diff = collections.Counter(), {}, 0.0, False
    for name, code, tpl in progs(rec):
        r = res.get((rec["id"], name)) or {}
        old = rec["samples"][name] if name == "rule" else agent.last_ok(rec["samples"][name])
        if (old or {}).get("answer") is not None and r.get("answer") is not None and agent.canon_of(old["answer"]) != agent.canon_of(r["answer"]): any_diff = True
        if name == "rule":
            w = W_NEW if (tpl or "").startswith("ind_") or tpl in NEW else 4.0
        else:
            sid = name.rstrip("LBE")
            if sid.isdigit() and int(sid) in ung: continue
            w = 1.0
        total += w
        if r.get("ok") and r.get("answer") is not None:
            c = agent.canon_of(r["answer"]); votes[c] += w; rep.setdefault(c, r["answer"])
    if votes and total:
        top, n = votes.most_common(1)[0]; final = rep[top] if n / total >= 0.5 else {"type": "abstain"}; conf = n / total
    else: final, conf = {"type": "abstain"}, 0.0
    if not any_diff:
        same_n += 1; same_ok += int(agent.canon_of(final) == agent.canon_of(rec["final"]))
    elif agent.canon_of(final) != agent.canon_of(rec["final"]): changed += 1
    rec = dict(rec)
    if any_diff: rec["final_logged"] = rec["final"]; rec["final"] = final; rec["conf"] = conf   # unaffected: the run's own vote
    out.append(rec)                                                                              # (map fallback, confirm phase)
print(f"self-check: {same_ok}/{same_n} unaffected questions re-voted to the logged answer; {changed} affected questions changed")
with open(os.path.join(out_dir, "logs.jsonl"), "w", encoding="utf-8") as f:
    for rec in out: f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
os.makedirs(os.path.join(out_dir, "shard0"), exist_ok=True)
open(os.path.join(out_dir, "shard0", "compile_logs.jsonl"), "w", encoding="utf-8").writelines(clogs)
if os.path.isfile(os.path.join(src_dir, "map_penalties.json")): shutil.copy(os.path.join(src_dir, "map_penalties.json"), out_dir)
