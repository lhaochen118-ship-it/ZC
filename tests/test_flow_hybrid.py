#!/usr/bin/env python3
"""The hybrid flow (src/hybrid.py through agent.py --flow hybrid --pipeline) end to end on CPU with a fake model:
- every question gets exactly one answer line, every answer comes from a model sample (no rule answer, no 'rule' vote);
- a program that fails the precheck gets a repair turn and the repaired program is the one that runs;
- the prompt carries the compact sections (# BUSINESS, # HINTS, # QUESTION) and no rule program;
- --dp 2 (two processes with claims) answers every question once;
- a tiny --budget still writes a line for every question (abstain when out of time);
- the <sql> escape (WP12) runs through precheck and DuckDB;
- --review 1 (M11): a flagged result gets exactly one review turn and keeps a valid answer;
- --calibrator (QĐ14): the online abstain decision equals confidence.decide on the same log;
- --think-rescue: a sample cut at the token cap without a program is re-run once in thinking mode.
Usage: test_flow_hybrid.py <train_split> <scratch>"""
import os, sys, json, subprocess, shutil
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAKE = r'''
import re, hashlib
def responder(conv):
    sysm, user = conv[0]["content"], conv[-1]["content"]
    if sysm.startswith("You are DataMate's data engineer"):            # mapping prompts (should not be needed: automap)
        return "```python\nraise ValueError('fake mapping')\n```"
    assert "# QUESTION" in conv[1]["content"] and "# BUSINESS" in conv[1]["content"], "compact prompt sections missing"
    assert "qparse" not in conv[1]["content"] and "rule program" not in conv[1]["content"].lower()
    if "The program has problems" in user or "The program failed" in user:
        return "<plan>orders | month | -</plan>\n<code>\nANSWER = {\"type\": \"number\", \"answer\": 7}\n</code>"
    h = int(hashlib.md5(conv[1]["content"][-200:].encode()).hexdigest(), 16)
    if h % 3 == 0:                                                      # an unknown function: precheck error -> repair turn
        return "<plan>x</plan>\n<code>\nANSWER = {\"type\": \"number\", \"answer\": revenu(month(2025, 1))}\n</code>"
    return "<plan>orders | month | -</plan>\n<code>\nANSWER = {\"type\": \"number\", \"answer\": 7}\n</code>"
'''

FAKE_SQL = r'''
def responder(conv):
    assert "<sql>" in conv[0]["content"], "the <sql> escape is not described in the system prompt"
    if "The program has problems" in conv[-1]["content"] or "The program failed" in conv[-1]["content"]:
        raise AssertionError("the SQL program was refused: " + conv[-1]["content"][:300])
    return "<plan>orders | all | -</plan>\n<sql>\nSELECT COUNT(*) AS number FROM orders WHERE valid\n</sql>"
'''


FAKE_CONT = '''def responder(conv, think=False):
    return "n | - | -</plan>\\n<code>\\nANSWER = {\\"type\\": \\"number\\", \\"answer\\": 7}\\n</code>"
'''
FAKE_RAMBLE = r'''
def responder(conv, think=None):
    if not think:                                                       # plain mode: reasons in prose until the token cap
        return ("Let me think about which toolkit function fits this question. " * 40, "MAX")
    return "<plan>orders | month | -</plan>\n<code>\nANSWER = {\"type\": \"number\", \"answer\": 7}\n</code>"
'''


def run(args, timeout=1800):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "src", "agent.py")] + args, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0: print((r.stdout + r.stderr)[-4000:]); raise SystemExit("agent.py failed")
    return r.stdout


def check(out, n_expected, label):
    sub = [json.loads(l) for l in open(os.path.join(out, "submission.jsonl"), encoding="utf-8")]
    logs = [json.loads(l) for l in open(os.path.join(out, "logs.jsonl"), encoding="utf-8")]
    ids = [x["id"] for x in sub]
    assert len(ids) == n_expected == len(set(ids)) == len(logs), (label, len(ids), n_expected, len(logs))
    for rec in logs:
        assert "rule" not in rec["samples"], "a rule program voted"
        if rec["final"].get("type") != "abstain":
            assert rec["samples"]["samples"], "an answer without a model sample"
    return sub, logs


def main():
    split, S = sys.argv[1], sys.argv[2]
    shutil.rmtree(S, ignore_errors=True); os.makedirs(S)
    fake = os.path.join(S, "fake.py"); open(fake, "w").write(FAKE)
    cache = os.environ.get("CACHE") or os.path.join(S, "cache")
    base = ["--flow", "hybrid", "--pipeline", "--fake-responder", fake, "--prefer-map", "fmt:automap", "--qw", "0", "--k-fast", "2", "--k-max", "4",
            "--cache", cache, "--workers", "4", "--subset", "24"]
    qs = [json.loads(l) for l in open(os.path.join(split, "questions.jsonl"), encoding="utf-8")]
    sys.path.insert(0, os.path.join(ROOT, "src")); import agent
    n = len(agent.select_subset(qs, 24))
    out1 = os.path.join(S, "o1"); run([split, out1] + base)
    sub, logs = check(out1, n, "dp1")
    rep = sum(1 for r in logs for s in r["samples"]["samples"] if len(s["turns"]) > 1)
    assert rep > 0, "no repair turn happened"
    for r in logs:
        for s in r["samples"]["samples"]:
            if len(s["turns"]) > 1:
                assert any("error:name" in x for x in s["turns"][0]["precheck"]), s["turns"][0]["precheck"]
                assert s["ok"] and s["answer"] == {"type": "number", "answer": 7}, s
    answered = [x for x in sub if x.get("type") == "number"]
    assert len(answered) >= n - 2, f"only {len(answered)} of {n} answered"
    early = sum(1 for r in logs if r["samples"]["stop"] == "early")
    print(f"dp1: {n} questions, {len(answered)} answered, {rep} repaired samples, {early} early stops")
    out2 = os.path.join(S, "o2"); run([split, out2] + base + ["--dp", "2"])
    check(out2, n, "dp2"); print("dp2: every question answered once")
    out3 = os.path.join(S, "o3"); run([split, out3] + base + ["--budget", "20"])
    check(out3, n, "budget"); print("budget: every question has a line")
    import time as _t                                                     # Kaggle 12 h session: --hard-deadline (eval_flow sets it
    out3h = os.path.join(S, "o3h"); run([split, out3h] + base + ["--hard-deadline", str(int(_t.time()) - 1)])   # from ZC_DEADLINE)
    subh, _ = check(out3h, n, "hard deadline past")
    assert all(x.get("type") == "abstain" for x in subh), "hard deadline in the past: every question must be abstained"
    out3f = os.path.join(S, "o3f"); run([split, out3f] + base + ["--hard-deadline", str(int(_t.time()) + 86400)])
    subf, _ = check(out3f, n, "hard deadline far")
    assert {x["id"]: x.get("answer") for x in subf} == {x["id"]: x.get("answer") for x in sub}, "a far hard deadline changed answers"
    print("hard deadline: past -> all abstained, far -> same answers as without it")
    fs = os.path.join(S, "fake_sql.py"); open(fs, "w").write(FAKE_SQL)              # WP12: <sql> -> precheck -> DuckDB -> answer
    out4 = os.path.join(S, "o4"); run([split, out4] + [x if x != fake else fs for x in base] + ["--escape", "python,sql", "--subset", "6"])
    _, logs4 = check(out4, len(agent.select_subset(qs, 6)), "sql")
    sm = [s for r in logs4 for s in r["samples"]["samples"]]
    assert sm and all(s["ok"] and s.get("escape") == "sql" and s["answer"]["type"] == "number" and s["answer"]["answer"] > 0 for s in sm), sm[:2]
    print(f"sql: {len(sm)} samples answered through DuckDB")
    out5 = os.path.join(S, "o5"); run([split, out5] + base + ["--review", "1"])              # M11: suspicious result -> one review turn
    _, logs5 = check(out5, n, "review")
    sm = [s for r in logs5 for s in r["samples"]["samples"]]
    rev = [s for s in sm if s.get("reviewed")]
    assert rev, "no review turn although the fake answer (always a number) contradicts typed hints"
    for s in rev:
        t = [x for x in s["turns"] if x.get("review")]
        assert len(t) == 1 and (t[0].get("post") or "Check before answering" in t[0]["review"]), t
        assert s["ok"], "a reviewed sample lost its answer"
        assert s["turns"].index(t[0]) < len(s["turns"]) - 1 or len(s["turns"]) > 2, "review turn not generated"
    print(f"review: {len(rev)} of {len(sm)} samples re-read once")
    import confidence as C                                              # WP8 / QĐ14: --calibrator decides online like offline
    X = C.matrix(logs); y = [float((r["final"] or {}).get("answer") == 7) for r in logs]
    cp = os.path.join(S, "cal.json"); C.Calibrator().fit(X, y).save(cp)
    out6 = os.path.join(S, "o6"); run([split, out6] + base + ["--calibrator", cp, "--theta-abstain", "0.5"])
    _, logs6 = check(out6, n, "calibrator"); cal = C.Calibrator.load(cp); m = 0
    for r in logs6:
        if r["samples"].get("best") is None: continue
        fin, p = C.decide(r, cal, 0.5); m += 1
        assert abs(p - r["samples"]["conf_cal"]) < 1e-9 and (fin.get("type") == "abstain") == (r["final"].get("type") == "abstain"), r["id"]
    assert m, "no calibrated decision"
    print(f"calibrator: {m} decisions equal to confidence.decide")
    fr = os.path.join(S, "fake_ramble.py"); open(fr, "w").write(FAKE_RAMBLE)    # think rescue: no program at the cap -> rerun thinking
    out7 = os.path.join(S, "o7"); run([split, out7] + [x if x != fake else fr for x in base] + ["--subset", "6", "--think-rescue", "1"])
    _, logs7 = check(out7, len(agent.select_subset(qs, 6)), "rescue")
    sm = [s for r in logs7 for s in r["samples"]["samples"]]
    assert sm and all(s["ok"] and s.get("rescue") and s["turns"][0].get("rescued") and s["answer"]["answer"] == 7 for s in sm), sm[:1]
    out8 = os.path.join(S, "o8"); run([split, out8] + [x if x != fake else fr for x in base] + ["--subset", "6", "--think-rescue", "0"])
    _, logs8 = check(out8, len(agent.select_subset(qs, 6)), "no rescue")
    assert not any(s.get("ok") for r in logs8 for s in r["samples"]["samples"]), "without the rescue the rambling model has no program"
    print(f"think rescue: {len(sm)} samples recovered (0 without --think-rescue)")
    fp = os.path.join(S, "fake_cont.py"); open(fp, "w").write(FAKE_CONT)        # P2: --prefill-plan: the model continues '<plan>'
    out9 = os.path.join(S, "o9"); run([split, out9] + [x if x != fake else fp for x in base] + ["--subset", "6", "--prefill-plan", "1", "--hint-counts", "1"])
    _, logs9 = check(out9, len(agent.select_subset(qs, 6)), "prefill")
    sm = [s for r in logs9 for s in r["samples"]["samples"]]
    assert sm and all(s["ok"] and s["answer"]["answer"] == 7 and s["turns"][0]["plan"].startswith("n |") for s in sm), sm[:1]
    assert all("counted" in (r["samples"]["hints"] or {}) for r in logs9), "--hint-counts: no 'counted' key in the hints"
    print(f"prefill: {len(sm)} samples parsed from a continuation of '<plan>'")
    print("test_flow_hybrid PASS")


if __name__ == "__main__":
    main()
