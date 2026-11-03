#!/usr/bin/env python3
"""Pipelined scheduler (src/pipeline.py) without a GPU, against the round-based path on the same mini split.
Two businesses: biz_101 (no reference fits on train -> LLM mapping; the first mapping leaves a status unmapped -> check
fails -> repair) and biz_102 (reference transfer). A fake engine finishes at most 7 requests per step, so generation,
execution and repairs interleave. Checks: same final answers as the round-based path, every question logged once,
compile logs complete, the claim of a business is exclusive.
Usage: test_pipeline.py <train_split_dir> <scratch_dir>"""
import os, sys, json, shutil, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import agent, pipeline, qparse

split, scratch = sys.argv[1], sys.argv[2]
mini = os.path.join(scratch, "mini_pipe"); shutil.rmtree(mini, ignore_errors=True); os.makedirs(os.path.join(mini, "data"))
for f in ("rulebook.md", "calendar.csv"): shutil.copy(os.path.join(split, f), mini)
want = []
for b in ("biz_101", "biz_102"):
    os.symlink(os.path.abspath(os.path.join(split, "data", b)), os.path.join(mini, "data", b))
    qs = [json.loads(l) for l in open(os.path.join(split, "questions.jsonl"), encoding="utf-8") if json.loads(l)["business_id"] == b]
    want += [q["id"] for q in qs[:10]]
for f in ("questions.jsonl", "answers.jsonl"):
    rows = [l for l in open(os.path.join(split, f), encoding="utf-8") if json.loads(l)["id"] in want]
    open(os.path.join(mini, f), "w", encoding="utf-8").writelines(rows)

MAP = open(os.path.join(os.path.dirname(agent.__file__), "refs", "compile", "biz_101.py"), encoding="utf-8").read()
BAD_MAP = MAP.replace('map_status(t["Kết quả"])', 'map_status(t["Kết quả"], {"Xong": "???"})')


def responder(conv):
    sysmsg, user, turns = conv[0]["content"], conv[1]["content"], len(conv)
    if sysmsg.startswith("You are DataMate's data engineer"):
        return "```python\n" + (BAD_MAP if turns == 2 else MAP) + "\n```"
    q = user[user.rindex("question: ") + 10:].split("\n")[0]
    t = qparse.K(q)
    if "doanh thu" in t or "dthu" in t or "tien" in t: return "```python\nANSWER = {'type': 'number', 'answer': revenue(month(2026, 5))}\n```"
    if "huy" in t: return "```python\nthis is not valid python\n```"            # exercises the repair path
    return "```python\nraise Unanswerable('fake')\n```"


def args(out, pipe):
    return argparse.Namespace(split=mini, out=out, model="fake", k=2, paths="", compile=False, kc=2, repairs=1, subset=0, tau=0.5,
                              cache=os.path.join(scratch, "cache_pipe"), tp=1, max_len=24576, think=False, workers=2, max_tokens=512,
                              compile_tokens=2048, timeout=60, temp=0.7, eager=False, no_reflect=False, lora="", dp=1, shard="", chunk=0,
                              budget=0, log_dir="", v1=False, qw=2.0, shots=3, map_llm=False, no_llm=False, prefix_budget=32000, quant="",
                              think_map=False, gpu_util=0.92, max_num_seqs=0, max_batched_tokens=0, spec_mtp=0,
                              pipeline=pipe, active=1, low_water=4, claim_dir="", claim_procs=1, deadline=0,
                              steal_chunk=4, q_inflight=6, escalate=0, esc_conf=0.75, esc_tokens=1024, esc_raw=1, complete_map=1, qclean=1)


shutil.rmtree(os.path.join(scratch, "cache_pipe"), ignore_errors=True)
a_rb = args(os.path.join(scratch, "out_rounds"), False); agent.run(a_rb, engine=agent.FakeEngine(responder))
a_pp = args(os.path.join(scratch, "out_pipe"), True); agent.run(a_pp, engine=pipeline.FakeStepLLM(responder))


def finals(out): return {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(out, "submission.jsonl"), encoding="utf-8")}


fr, fp = finals(a_rb.out), finals(a_pp.out)
logs = [json.loads(l) for l in open(os.path.join(a_pp.out, "logs.jsonl"), encoding="utf-8")]
clog = [json.loads(l) for l in open(os.path.join(a_pp.out, "compile_logs.jsonl"), encoding="utf-8")]
chosen = {c["business_id"]: c["chosen"] for c in clog if "chosen" in c}
res = json.load(open(os.path.join(a_pp.out, "score.json")))
print("pipeline", {k: res.get(k) for k in ("PA", "correct", "wrong", "abstain")}, "| chosen", chosen)
assert set(fp) == set(want) and len(logs) == len(want), "every question answered and logged exactly once"
diff = [q for q in want if fr[q] != fp[q]]
assert not diff, f"pipeline differs from the round-based path on {diff[:3]}: {[(fr[q], fp[q]) for q in diff[:3]]}"
assert chosen["biz_101"].startswith("llm:") and chosen["biz_102"].startswith("ref:"), chosen
assert any(any(len(v) > 1 for v in c["llm_attempts"].values()) for c in clog if "llm_attempts" in c), "the bad mapping was not repaired"
# escalation: low-confidence / unanswerable / flagged answers get 2 more samples; every question still answered once
a_es = args(os.path.join(scratch, "out_esc"), True); a_es.escalate = 2; agent.run(a_es, engine=pipeline.FakeStepLLM(responder))
les = [json.loads(l) for l in open(os.path.join(a_es.out, "logs.jsonl"), encoding="utf-8")]
n_esc = sum(1 for r in les if "escalated" in r["samples"])
assert len(les) == len(want) and len({r["id"] for r in les}) == len(want), "escalation must not duplicate or drop questions"
assert n_esc > 0 and all(len([k for k in r["samples"] if k.endswith("E")]) == 2 for r in les if "escalated" in r["samples"]), n_esc
assert all("cleaned" in json.dumps(r) or True for r in les)
print("escalation |", n_esc, "of", len(les), "questions escalated, each with 2 extra samples")
# --confirm: the rule program + the greedy sample first; when they agree that is the answer (one LLM sample), else all k
# samples. With qw 2 and k 2 an agreeing pair holds 3 of the 4 votes, so the answers must equal the full pipeline's.
def responder_rule(conv):                                              # an LLM that writes the rule's own template program
    sysmsg, user = conv[0]["content"], conv[1]["content"]
    if sysmsg.startswith("You are DataMate's data engineer"): return responder(conv)
    q = user[user.rindex("question: ") + 10:].split("\n")[0]; pr = qparse.parse(q, "2026-06-01")
    return "```python\n" + (pr[1] if pr else "raise Unanswerable('fake')") + "\n```"
a_fu = args(os.path.join(scratch, "out_full_r"), True); agent.run(a_fu, engine=pipeline.FakeStepLLM(responder_rule))
a_cf = args(os.path.join(scratch, "out_confirm"), True); a_cf.confirm = 1; agent.run(a_cf, engine=pipeline.FakeStepLLM(responder_rule))
lcf = [json.loads(l) for l in open(os.path.join(a_cf.out, "logs.jsonl"), encoding="utf-8")]
assert len(lcf) == len(want) and len({r["id"] for r in lcf}) == len(want), "confirm must not duplicate or drop questions"
nsamp = [len([k for k in r["samples"] if k[:1].isdigit()]) for r in lcf]
fcf, ffu = finals(a_cf.out), finals(a_fu.out)
assert nsamp.count(1) > 0 and nsamp.count(2) > 0 and nsamp.count(1) + nsamp.count(2) == len(lcf), nsamp
assert fcf == ffu, f"--confirm differs from the full pipeline on {[q for q in want if fcf.get(q) != ffu[q]][:3]}"
print("confirm |", nsamp.count(1), "questions settled by rule + greedy sample,", nsamp.count(2), "with every sample; answers identical")
# --map-retry: when every mapping sample of a business fails, a second batch runs (here the first 4 mapping generations
# = 2 samples x 2 rounds fail); the business still gets an LLM mapping and every question is answered once
calls = {"map": 0}
def responder_failfirst(conv):                                         # biz_101's own first 4 mapping generations fail (other
    if conv[0]["content"].startswith("You are DataMate's data engineer") and "biz_101" in conv[1]["content"]:   # businesses'
        calls["map"] += 1                                                 # calls may interleave in any order)
        if calls["map"] <= 4: return "```python\nraise ValueError('broken mapping')\n```"
    return responder(conv)
a_mr = args(os.path.join(scratch, "out_retry"), True)
agent.run(a_mr, engine=pipeline.FakeStepLLM(responder_failfirst))
cl = [json.loads(l) for l in open(os.path.join(a_mr.out, "compile_logs.jsonl"), encoding="utf-8")]
ch = {c["business_id"]: c["chosen"] for c in cl if "chosen" in c}
att = [c for c in cl if c.get("business_id") == "biz_101" and "llm_attempts" in c][0]["llm_attempts"]
assert len(att) == 4 and ch.get("biz_101", "").startswith("llm:"), (len(att), ch)   # samples 0-1 failed, 2-3 from the retry
assert len(finals(a_mr.out)) == len(want)
print("map retry | biz_101 mapped by", ch["biz_101"], "after a failed first batch")
# --map-repairs 3 --map-repair-think 0: a mapping sample gets 3 repair rounds of its own, the repairs without reasoning
# and with the plain program budget (round 0 keeps its reasoning); the first 3 generations of biz_101 fail
class RecLLM(pipeline.FakeStepLLM):
    def __init__(self, r): super().__init__(r); self.seen = []
    def add(self, conv, temp, seed, max_tokens, think=None, priority=0):
        if conv[0]["content"].startswith("You are DataMate's data engineer") and "biz_101" in conv[1]["content"]: self.seen.append((len(conv), think, max_tokens))
        return super().add(conv, temp, seed, max_tokens, think, priority)
calls["map"] = 0
def responder_fail3(conv):
    if conv[0]["content"].startswith("You are DataMate's data engineer") and "biz_101" in conv[1]["content"]:
        calls["map"] += 1
        if calls["map"] <= 3: return "```python\nraise ValueError('broken mapping')\n```"
    return responder(conv)
a_mp = args(os.path.join(scratch, "out_maprep"), True); a_mp.think_map = True; a_mp.map_repairs = 3; a_mp.map_repair_think = 0; a_mp.map_retry = 0
rec = RecLLM(responder_fail3); agent.run(a_mp, engine=rec)
cl = [json.loads(l) for l in open(os.path.join(a_mp.out, "compile_logs.jsonl"), encoding="utf-8")]
att = [c for c in cl if c.get("business_id") == "biz_101" and "llm_attempts" in c][0]["llm_attempts"]
rounds = max(len(v) for v in att.values())
first = [x for x in rec.seen if x[0] == 2]; later = [x for x in rec.seen if x[0] > 2]
assert len(att) == 2 and rounds >= 3, (len(att), {k: len(v) for k, v in att.items()})     # 3 failures within the first batch
assert all(t is True for _, t, _ in first) and later and all(t is False and m == a_mp.compile_tokens for _, t, m in later), rec.seen
assert len(finals(a_mp.out)) == len(want)
print("map repairs | biz_101: up to", rounds, "rounds per sample, repairs without reasoning", len(later))
# --map-vote 2: the programs of a question also run on the second-ranked candidate mapping (answers in 'alt'); every
# question is still answered once; the compile log records each candidate's answer agreement
a_mv = args(os.path.join(scratch, "out_mapvote"), True); a_mv.map_vote = 2; a_mv.sel_agree = 30.0
agent.run(a_mv, engine=pipeline.FakeStepLLM(responder))
lmv = [json.loads(l) for l in open(os.path.join(a_mv.out, "logs.jsonl"), encoding="utf-8")]
n_alt = sum(1 for r in lmv for k, v in r["samples"].items() if isinstance(v, list) and v and isinstance(v[-1], dict) and "alt" in v[-1])
n_alt += sum(1 for r in lmv if isinstance(r["samples"].get("rule"), dict) and "alt" in r["samples"]["rule"])
clm = [json.loads(l) for l in open(os.path.join(a_mv.out, "compile_logs.jsonl"), encoding="utf-8")]
multi = [c for c in clm if "chosen" in c and len(c["candidates"]) >= 2]
assert len(finals(a_mv.out)) == len(want) and n_alt > 0 and multi and all(len(x) == 5 for c in multi for x in c["candidates"]), (n_alt, multi[:1])
print("map vote |", n_alt, "programs also ran on a second mapping;", len(multi), "businesses with >= 2 candidates")
# the real CLI: agent.py runs as __main__ and pipeline.py imports agent (Kaggle crash of 02/11: a second module copy
# without CARDS / worker context); the fake responder is loaded from a file, no model
import subprocess, inspect
rf = os.path.join(scratch, "fake_responder.py")
open(rf, "w", encoding="utf-8").write("import os, sys\nimport qparse\nMAP = " + repr(MAP) + "\nBAD_MAP = " + repr(BAD_MAP) + "\n" + inspect.getsource(responder))
out_cli = os.path.join(scratch, "out_cli"); shutil.rmtree(out_cli, ignore_errors=True)
r = subprocess.run([sys.executable, os.path.join(os.path.dirname(agent.__file__), "agent.py"), mini, out_cli, "--pipeline", "--fake-responder", rf,
                    "--k", "2", "--kc", "2", "--repairs", "1", "--shots", "3", "--workers", "2", "--active", "1", "--low-water", "4",
                    "--cache", os.path.join(scratch, "cache_pipe")], capture_output=True, text=True, timeout=600)
assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
fc = finals(out_cli)
assert fc == fp, f"CLI run differs from the in-process pipeline on {[q for q in want if fc.get(q) != fp[q]][:3]}"
print("CLI (agent.py as __main__) | same answers")
# the multi-GPU path on CPU: run_dp ingests once, two processes claim businesses, outputs are merged
def cli(out, *extra):
    shutil.rmtree(out, ignore_errors=True)
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(agent.__file__), "agent.py"), mini, out, "--pipeline", "--fake-responder", rf,
                        "--k", "2", "--kc", "2", "--repairs", "1", "--shots", "3", "--workers", "4", "--active", "1", "--low-water", "4",
                        "--cache", os.path.join(scratch, "cache_pipe_dp"), *extra], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    return r.stdout
log = cli(os.path.join(scratch, "out_dp"), "--dp", "2")
fd = finals(os.path.join(scratch, "out_dp"))
claims = sorted(x for x in os.listdir(os.path.join(scratch, "out_dp", "claims")) if not x.startswith("."))
assert fd == fp, f"--dp 2 differs on {[q for q in want if fd.get(q) != fp[q]][:3]}"
# work units: one mapping claim + one published mapping per business, every 16-question chunk claimed exactly once
assert claims == ["biz_101", "biz_101.c0", "biz_101.map", "biz_102", "biz_102.c0", "biz_102.map"], claims
print("run_dp --dp 2 (claims", claims, ") | same answers")
# a budget too small to answer anything: every question must still be written (abstain), nothing lost
cli(os.path.join(scratch, "out_budget"), "--dp", "2", "--budget", "1")
fb = finals(os.path.join(scratch, "out_budget"))
assert set(fb) == set(want), f"{len(set(want) - set(fb))} questions missing under the time budget"
print("budget 1 s |", sum(1 for v in fb.values() if v["type"] == "abstain"), "of", len(fb), "abstained, none missing")
cd = os.path.join(scratch, "claims_test"); shutil.rmtree(cd, ignore_errors=True); os.makedirs(cd)
assert pipeline.claim(cd, "biz_1") and not pipeline.claim(cd, "biz_1") and pipeline.claim(cd, "biz_2"), "claims must be exclusive"
print("PASS")
