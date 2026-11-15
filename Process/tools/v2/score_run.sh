#!/bin/bash
# score_run.sh <run> [...]: pub_full / pub_ind / pub_noind of each run's submission + the units line of its log
S=/tmp/claude-0/-home-user-ZC/53d4fe9e-6712-52f3-bd72-e042dbdf5ede/scratchpad
cd /home/user/ZC
for r in "$@"; do
  d=$(dirname $(ls $S/runs/$r/zc-$r-1_out/results/$r/*/submission.jsonl | head -1))
  echo "== $r ($(basename $d))"
  python3 Process/tools/score_sets.py $d/submission.jsonl 2>/dev/null | grep -E "pub_full|pub_ind|pub_noind" | head -3
  grep -ah "units: rc\|units: reused" $d/eval.log $S/runs/$r/zc-$r-1_out/logs/*.log 2>/dev/null | tail -1
done
