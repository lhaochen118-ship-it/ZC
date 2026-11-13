#!/usr/bin/env python3
"""Rebuild data/raw/public_gt.jsonl (public reference answers, SCORING ONLY: never read by src/ or prompts) without asking the
user for score_1.jsonl: the rule-first submission v30 (commit 30fee95, submissions/v30/submission.jsonl) was scored 1000 / 1000
against the organisers' reference (EXPERIMENT_LOG entry 27), so its answers equal the reference under the judge (aliases are
not kept). Check: final_submission/submission.jsonl must score exactly its recorded final_submission/score.json.
Usage: rebuild_public_gt.py [--commit 30fee95]"""
import os, sys, json, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import score

commit = sys.argv[sys.argv.index("--commit") + 1] if "--commit" in sys.argv else "30fee95"
txt = subprocess.run(["git", "-C", ROOT, "show", f"{commit}:submissions/v30/submission.jsonl"], capture_output=True, text=True)
if txt.returncode != 0:
    sys.exit(f"git show failed (shallow clone? run: git fetch --unshallow origin): {txt.stderr[-300:]}")
rows = [json.loads(l) for l in txt.stdout.splitlines() if l.strip()]
assert len(rows) == 1000, len(rows)
out = os.path.join(ROOT, "data", "raw", "public_gt.jsonl"); os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(dict(r, alias=[]), ensure_ascii=False) + "\n")
fs = os.path.join(ROOT, "final_submission")
if os.path.isfile(os.path.join(fs, "score.json")):
    want = json.load(open(os.path.join(fs, "score.json")))
    res, _ = score.score(score.load(out), score.load(os.path.join(fs, "submission.jsonl")))
    ok = round(res.get("PA", 0), 4) == round(want["pa"], 4) and res.get("correct", 0) == want["right"]
    print(f"check: final_submission scores PA {res.get('PA')} ({res.get('correct')} right) vs recorded {want['pa']} ({want['right']}): {'OK' if ok else 'MISMATCH'}")
    if not ok: sys.exit(1)
print(f"wrote {out} ({len(rows)} answers)")
