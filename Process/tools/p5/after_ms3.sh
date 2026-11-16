#!/bin/bash
# P5 16/11 MS3 chain: training done -> adapter to training/data/out/p5ad/MS3 -> FP8 merges (MS3, ML1, MJ2) on the two eval accounts
# -> evaluation of the two shop halves (public_test@i/2) with flow_e8r (= flow_p5_final_h with the MoE + LoRAs).
cd /home/user/ZC
until grep -q "STATUS=" Process/runs/p5/fts3/zc-fts3-1.log 2>/dev/null; do sleep 60; done
grep -q "STATUS=DONE" Process/runs/p5/fts3/zc-fts3-1.log || { echo "$(date -u +%H:%M) fts3 not DONE: $(grep STATUS= Process/runs/p5/fts3/zc-fts3-1.log)"; exit 1; }
d=/tmp/ms3dl; rm -rf $d; mkdir -p $d
for t in 1 2 3; do KAGGLE_API_TOKEN=$(cat ~/.kaggle/tok_nguynduylm) kaggle kernels output nguynduylm/zc-fts3-1 -p $d -o --file-pattern '.*MS3/(adapter/.*|result\.json|train_metrics\.json)$' > $d.log 2>&1 && break; sleep 30; done
a=$(dirname $(find $d -name adapter_model.safetensors | head -1))
[ -f "$a/adapter_config.json" ] || { echo "$(date -u +%H:%M) MS3 adapter missing"; exit 1; }
mkdir -p training/data/out/p5ad/MS3; rm -rf training/data/out/p5ad/MS3/adapter; cp -r $a training/data/out/p5ad/MS3/adapter; cp $(find $d -name result.json | head -1) training/data/out/p5ad/MS3/ 2>/dev/null
echo "$(date -u +%H:%M) adapter $(du -sh training/data/out/p5ad/MS3/adapter | cut -f1); $(cat training/data/out/p5ad/MS3/result.json 2>/dev/null | tr -d '\n' | cut -c1-200)"
rm -rf $d
mg(){ acc=$1; slug=$2; name=$3; ad=$4; echo "run_kaggle.py public_test OUT --script training/merge_fp8.py --name Qwen3.6-35B-A3B-E88-$name-FP8 --adapter $ad --weight 1.0 --keep configs/moe/keep_e88.json" > automation/jobs/p5/$slug.jobs
  (ZC_OUT_PATTERN='.*\.(log|json)$' python3 automation/fanout.py Process/runs/p5/$slug zc-$slug automation/jobs/p5/$slug.jobs 1 --accounts $acc --cpu --internet --train-data p5ad --no-preflight > Process/runs/p5/$slug.log 2>&1 &)
  until grep -qE "launched|rror" Process/runs/p5/$slug.log; do sleep 5; done; echo "$(date -u +%H:%M) $slug: $(grep -E 'launched|rror' Process/runs/p5/$slug.log | head -1)"; echo $slug >> Process/runs/p5/active_runs.txt; sleep 60; }
mg loinguyentran m3s0 MS3 MS3; mg loinguyentran m3l0 ML1 ML1; mg loinguyentran m3j0 MJ2 MJ2
mg namhunhh m3s1 MS3 MS3; mg namhunhh m3l1 ML1 ML1; mg namhunhh m3j1 MJ2 MJ2
for i in 0 1; do
  ( for s in m3s$i m3l$i m3j$i; do until grep -q "STATUS=" Process/runs/p5/$s/zc-$s-1.log 2>/dev/null; do sleep 60; done; done
    echo "$(date -u +%H:%M) merges $i: $(grep -rhoE 'modules merged into [^;]*' Process/runs/p5/m3?$i 2>/dev/null | sort -u | tr '\n' ' ')"
    python3 automation/launch_many.py Process/runs/p5 automation/plans_p5/wave18_e8ms3p$i.json; echo e8ms3p$i >> Process/runs/p5/active_runs.txt ) &
  sleep 150
done
wait
