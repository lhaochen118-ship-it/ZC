#!/bin/bash
# P5 16/11 MS3 chain v2 (after_ms3.sh launched evals even when a merge failed, and the first merge of an account hit STALE CODE:
# the 2.6 GB adapter made the code dataset slow to process). Per account: launch the 3 FP8 merges (MS3, ML1, MJ2) one after the
# other; a merge that ends STALE CODE / error is relaunched (up to 3 times, 3 min apart); the half-split evaluation is launched only
# when all 3 merges of that account ended DONE:complete with "modules merged".
cd /home/user/ZC
merge_ok(){ grep -q "STATUS=DONE:complete" Process/runs/p5/$1/zc-$1-1.log 2>/dev/null; }
launch(){ acc=$1; slug=$2; name=$3; ad=$4
  echo "run_kaggle.py public_test OUT --script training/merge_fp8.py --name Qwen3.6-35B-A3B-E88-$name-FP8 --adapter $ad --weight 1.0 --keep configs/moe/keep_e88.json" > automation/jobs/p5/$slug.jobs
  [ -d Process/runs/p5/$slug ] && mv Process/runs/p5/$slug Process/runs/p5/_${slug}_$(date +%H%M) && mv Process/runs/p5/$slug.log Process/runs/p5/_${slug}_$(date +%H%M).log
  ZC_OUT_PATTERN='.*\.(log|json)$' python3 automation/fanout.py Process/runs/p5/$slug zc-$slug automation/jobs/p5/$slug.jobs 1 --accounts $acc --cpu --internet --train-data p5ad --no-preflight > Process/runs/p5/$slug.log 2>&1 &
  until grep -qE "launched|rror" Process/runs/p5/$slug.log 2>/dev/null; do sleep 5; done
  grep -qx $slug Process/runs/p5/active_runs.txt || echo $slug >> Process/runs/p5/active_runs.txt
  echo "$(date -u +%H:%M) $slug: $(grep -E 'launched|rror' Process/runs/p5/$slug.log | head -1)"; }
wait_done(){ until grep -q "STATUS=" Process/runs/p5/$1/zc-$1-1.log 2>/dev/null; do sleep 30; done; }
account(){ acc=$1; i=$2
  for pair in "m3s$i MS3 MS3" "m3l$i ML1 ML1" "m3j$i MJ2 MJ2"; do
    set -- $pair; slug=$1; name=$2; ad=$3
    merge_ok $slug && { echo "$(date -u +%H:%M) $slug already done"; continue; }
    [ -f Process/runs/p5/$slug.log ] || launch $acc $slug $name $ad      # a launch already in flight (old chain) is reused
    for try in 1 2 3; do
      wait_done $slug
      merge_ok $slug && break
      echo "$(date -u +%H:%M) $slug try $try failed: $(grep -hoE 'STALE CODE[^\\]*' Process/runs/p5/$slug/zc-$slug-1_out/*.log 2>/dev/null | head -1)"
      [ $try -lt 3 ] && { sleep 180; launch $acc $slug $name $ad; }
    done
    merge_ok $slug || { echo "$(date -u +%H:%M) $slug FAILED 3 times: no evaluation for half $i"; return 1; }
  done
  echo "$(date -u +%H:%M) half $i merges OK: $(grep -rhoE 'modules merged into [^;]*' Process/runs/p5/m3?$i 2>/dev/null | sed 's#.*/##' | sort -u | tr '\n' ' ')"
  python3 automation/launch_many.py Process/runs/p5 automation/plans_p5/wave18_e8ms3p$i.json
  grep -qx e8ms3p$i Process/runs/p5/active_runs.txt || echo e8ms3p$i >> Process/runs/p5/active_runs.txt; }
account loinguyentran 0 &
sleep 120
account namhunhh 1 &
wait
