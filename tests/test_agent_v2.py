#!/usr/bin/env python3
"""Local test of the v2 (toolkit) pipeline without a GPU: a FakeEngine plays the LLM.
Checks: reference transfer is skipped for a shop's own program on train (leave-one-out) -> the LLM mapping path runs
(first mapping leaves a status unmapped -> check fails -> repair), toolkit answer programs + rule-based programs are
executed and voted, Unanswerable becomes an `unanswerable` answer, and the run is scored.
Usage: test_agent_v2.py <train_split_dir> <scratch_dir>"""
import os, sys, json, shutil, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import agent

split, scratch = sys.argv[1], sys.argv[2]
mini = os.path.join(scratch, "mini_v2"); shutil.rmtree(mini, ignore_errors=True); os.makedirs(os.path.join(mini, "data"))
for f in ("rulebook.md", "calendar.csv"): shutil.copy(os.path.join(split, f), mini)
os.symlink(os.path.abspath(os.path.join(split, "data", "biz_101")), os.path.join(mini, "data", "biz_101"))
qs = [json.loads(l) for l in open(os.path.join(split, "questions.jsonl"), encoding="utf-8") if json.loads(l)["business_id"] == "biz_101"]
want = [q["id"] for q in qs[:12]]
for f in ("questions.jsonl", "answers.jsonl"):
    rows = [l for l in open(os.path.join(split, f), encoding="utf-8") if json.loads(l)["id"] in want]
    open(os.path.join(mini, f), "w", encoding="utf-8").writelines(rows)

MAP = open(os.path.join(os.path.dirname(agent.__file__), "refs", "compile", "biz_101.py"), encoding="utf-8").read()
BAD_MAP = MAP.replace('map_status(t["Kết quả"])', 'map_status(t["Kết quả"], {"Xong": "???"})')
assert BAD_MAP != MAP
calls = {"map": 0, "answer": 0}


def responder(conv):
    sysmsg, user, turns = conv[0]["content"], conv[1]["content"], len(conv)
    if sysmsg.startswith("You are DataMate's data engineer"):
        calls["map"] += 1
        assert "# EXAMPLES" in user and "Its mapping program" in user, "mapping prompt lacks reference examples"
        return "```python\n" + (BAD_MAP if turns == 2 else MAP) + "\n```"
    calls["answer"] += 1
    assert "CANONICAL TABLES" in user and "# EXAMPLES" in user, "answer prompt lacks the canonical card / examples"
    q = user[user.rindex("question: ") + 10:].split("\n")[0]
    import qparse, pandas as pd
    # the fake LLM answers like the rule-based reader when it can, otherwise says unanswerable
    t = qparse.K(q)
    if "doanh thu" in t or "dthu" in t or "tien" in t: return "```python\nANSWER = {'type': 'number', 'answer': revenue(month(2026, 5))}\n```"
    return "```python\nraise Unanswerable('fake')\n```"


ap = argparse.Namespace(split=mini, out=os.path.join(scratch, "out_v2"), model="fake", k=2, paths="", compile=False, kc=2, repairs=1, subset=0,
                        tau=0.5, cache=os.path.join(scratch, "cache_v2"), tp=1, max_len=24576, think=False, workers=2, max_tokens=512,
                        compile_tokens=2048, timeout=60, temp=0.7, eager=False, no_reflect=False, lora="", dp=1, shard="", chunk=0, budget=0,
                        log_dir=os.path.join(scratch, "out_v2", "logs"), v1=False, qw=2.0, shots=3, map_llm=False, no_llm=False)
shutil.rmtree(ap.cache, ignore_errors=True)
agent.run(ap, engine=agent.FakeEngine(responder))
res = json.load(open(os.path.join(ap.out, "score.json")))
logs = [json.loads(l) for l in open(os.path.join(ap.out, "logs.jsonl"), encoding="utf-8")]
clog = [json.loads(l) for l in open(os.path.join(ap.out, "compile_logs.jsonl"), encoding="utf-8")]
chosen = [c for c in clog if "chosen" in c]
print("calls", calls, "| chosen mapping", [c["chosen"] for c in chosen], "| score", {k: res.get(k) for k in ("PA", "correct", "wrong", "abstain")})
assert calls["map"] >= 3, "the bad mapping was not repaired"
assert chosen and chosen[0]["chosen"].startswith("llm:"), "own reference program must not be transferred on train"
assert len(logs) == len(want) and len(os.listdir(ap.log_dir)) == len(want)
assert res["correct"] >= 8, "rule-based programs should answer most of these questions"
print("PASS")
