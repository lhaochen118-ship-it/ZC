#!/bin/bash
# P5 16/11 alpha sweep (A: MS2 x 0.50, B: MS2 x 0.35) on flow_e8rw7's framework, each variant = 2 shop halves on 2 accounts in parallel.
# Per account: FP8 merges it lacks (merge_fp8.py; launched one after the other, each waits until its kernel RUNS so that the next
# code-dataset upload cannot replace the version it mounts), retried on failure (3x); then the half's evaluation, only when every
# merge it needs ended DONE:complete. Usage: after_alpha.sh <acc> <half i> <variant tag w5|w35> <solver name MS2W5|MS2W35> <weight> <need ML1/MJ2 0|1>
cd /home/user/ZC
acc=$1; i=$2; tag=$3; sname=$4; w=$5; needw=$6; sad=${7:-MS2}; td=${8:-p5ad_w}
log(){ echo "$(date -u +%H:%M) [$acc $tag p$i] $*"; }
ok(){ grep -q "STATUS=DONE:complete" Process/runs/p5/$1/zc-$1-1.log 2>/dev/null; }
launch_merge(){ slug=$1; name=$2; ad=$3; wt=$4
  echo "run_kaggle.py public_test OUT --script training/merge_fp8.py --name Qwen3.6-35B-A3B-E88-$name-FP8 --adapter $ad --weight $wt --keep configs/moe/keep_e88.json" > automation/jobs/p5/$slug.jobs
  if [ -d Process/runs/p5/$slug ]; then s=$(date +%H%M%S); mv Process/runs/p5/$slug Process/runs/p5/_${slug}_$s; mv Process/runs/p5/$slug.log Process/runs/p5/_${slug}_$s.log 2>/dev/null; fi
  ZC_OUT_PATTERN='.*\.(log|json)$' python3 automation/fanout.py Process/runs/p5/$slug zc-$slug automation/jobs/p5/$slug.jobs 1 --accounts $acc --cpu --internet --train-data $td --no-preflight > Process/runs/p5/$slug.log 2>&1 &
  until grep -qE "launched|rror" Process/runs/p5/$slug.log 2>/dev/null; do sleep 5; done
  grep -qx $slug Process/runs/p5/active_runs.txt || echo $slug >> Process/runs/p5/active_runs.txt
  log "$slug: $(grep -E 'launched|rror' Process/runs/p5/$slug.log | head -1)"
  until grep -qE "RUNNING|STATUS=" Process/runs/p5/$slug/zc-$slug-1.log 2>/dev/null; do sleep 10; done; }
merges="s:$sname:$sad:$w"; [ "$needw" = 1 ] && merges="$merges l:ML1:ML1:1.0 j:MJ2:MJ2:1.0"
slugs=""
for m in $merges; do IFS=: read k name ad wt <<< "$m"; slug="a${tag}${k}$i"; slugs="$slugs $slug"; [ -f Process/runs/p5/$slug/zc-$slug-1.log ] && continue; launch_merge $slug $name $ad $wt; done   # resume-safe
for slug in $slugs; do
  for try in 1 2 3; do
    until grep -q "STATUS=" Process/runs/p5/$slug/zc-$slug-1.log 2>/dev/null; do sleep 30; done
    ok $slug && break
    log "$slug try $try failed: $(grep -hoE 'STALE CODE[^\\]{0,60}|Error[^\\]{0,80}' Process/runs/p5/$slug/zc-$slug-1_out/*.log 2>/dev/null | head -1)"
    [ $try -lt 3 ] || { log "$slug FAILED: no evaluation"; exit 1; }
    sleep 60; m=$(echo $merges | tr ' ' '\n' | grep "^${slug:$((${#tag}+1)):1}:"); IFS=: read k name ad wt <<< "$m"; launch_merge $slug $name $ad $wt
  done
done
log "merges OK: $(grep -rhoE 'modules merged into [^;]*' $(for s in $slugs; do echo Process/runs/p5/$s; done) 2>/dev/null | sed 's#.*/##' | sort -u | tr '\n' ' ')"
ks="nhatlong1103/zc-assets$(for s in $slugs; do printf ',%s/zc-%s-1' $acc $s; done)"
if [ "$needw" = 0 ]; then ks="$ks,$(cat Process/tools/p5/w_kernels_$acc.txt)"; fi
python3 - "$acc" "e8r${tag}p$i" "$ks" <<'PY'
import json, sys
acc, name, ks = sys.argv[1:4]
json.dump([{"acc": acc, "name": name, "jobs": f"automation/jobs/p5/{name}.jobs", "est_h": 1.3, "args": ["--ks", ks, "--no-preflight"]}],
          open(f"automation/plans_p5/wave21_{name}.json", "w"), indent=1)
PY
[ -d Process/runs/p5/e8r${tag}p$i ] && { log "evaluation already launched"; exit 0; }
python3 automation/launch_many.py Process/runs/p5 automation/plans_p5/wave21_e8r${tag}p$i.json
grep -qx e8r${tag}p$i Process/runs/p5/active_runs.txt || echo e8r${tag}p$i >> Process/runs/p5/active_runs.txt
log "evaluation launched"
