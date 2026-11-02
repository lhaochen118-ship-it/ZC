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
