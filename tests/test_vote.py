#!/usr/bin/env python3
"""Multi-run vote (src/vote.py): plurality over runs, abstaining runs give no vote, the top answer needs >= 0.4 of the
runs, ties by confidence; the submission is written atomically and every question of any run is in it.
Usage: test_vote.py <scratch_dir>"""
import os, sys, json, shutil, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import vote

scratch = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_vote"
shutil.rmtree(scratch, ignore_errors=True)
N = lambda x: {"type": "number", "answer": x}
AB = {"type": "abstain"}
runs = {"r1": {"q1": (N(1), 0.9), "q2": (AB, 0.0), "q3": (N(5), 0.6), "q4": (N(7), 0.5)},
        "r2": {"q1": (N(1), 0.8), "q2": (AB, 0.0), "q3": (N(6), 0.9), "q4": (N(7), 0.5)},
        "r3": {"q1": (N(2), 1.0), "q2": (N(3), 0.7), "q3": (AB, 0.0)}}
dirs = []
for name, qs in runs.items():
    d = os.path.join(scratch, name); os.makedirs(d); dirs.append(d)
    with open(os.path.join(d, "logs.jsonl"), "w", encoding="utf-8") as f:
        for q, (final, conf) in qs.items(): f.write(json.dumps({"id": q, "final": final, "conf": conf}) + "\n")
bad = []
res = vote.vote([vote.load(d) for d in dirs])
want = {"q1": N(1), "q2": AB, "q3": AB, "q4": N(7)}            # q2: 1 vote of 3 < 0.4; q3: 1-1 tie, each 1 of 3 < 0.4
for q, w in want.items():
    if res[q][0] != w: bad.append((q, res[q][0], w))
two = vote.vote([vote.load(d) for d in dirs[1:]])              # 2 runs: one vote is enough, ties by confidence
if two["q3"][0] != N(6) or two["q2"][0] != N(3): bad.append(("two runs", two["q3"][0], two["q2"][0]))
# runs cut short (no record of q4: r3 and a run with q1 only) have no say on q4: 1 vote of the 1 run that reached it
cut = vote.vote([vote.load(dirs[0]), vote.load(dirs[2]), {"q1": {"id": "q1", "final": N(1), "conf": 0.5}}])
if cut["q4"][0] != N(7) or cut["q1"][0] != N(1): bad.append(("cut run", cut["q4"], cut["q1"]))
out = os.path.join(scratch, "submission.jsonl")
r = subprocess.run([sys.executable, os.path.join(os.path.dirname(vote.__file__), "vote.py"), out] + dirs, capture_output=True, text=True)
rows = [json.loads(l) for l in open(out, encoding="utf-8")] if r.returncode == 0 else []
if r.returncode != 0 or {x["id"] for x in rows} != set(want) or os.path.exists(out + ".tmp"): bad.append(("cli", r.stderr[-300:], rows))
for b in bad: print("FAIL", b)
print(f"{len(bad)} failures")
print("PASS" if not bad else "FAIL")
