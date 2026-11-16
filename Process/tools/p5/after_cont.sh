#!/bin/bash
# P5 16/11 F2 / E2: continued training done -> adapter downloaded into training/data/out/<dir>/<NAME>/adapter -> the 2 shop-half
# evaluations (after_alpha.sh: FP8 merges + flow_e8r*_half) on 2 accounts. Usage: after_cont.sh <slug> <train acc> <NAME> <dir> <alpha> <tag> <acc0> <acc1> <lora_experts 0|1>
cd /home/user/ZC
slug=$1; tacc=$2; name=$3; dir=$4; w=$5; tag=$6; a0=$7; a1=$8
log(){ echo "$(date -u +%H:%M) [$slug] $*"; }
until grep -q "STATUS=" Process/runs/p5/$slug/zc-$slug-1.log 2>/dev/null; do sleep 60; done
grep -q "STATUS=DONE:complete" Process/runs/p5/$slug/zc-$slug-1.log || { log "training not complete: $(grep STATUS= Process/runs/p5/$slug/zc-$slug-1.log)"; exit 1; }
d=/tmp/dl_$slug; rm -rf $d; mkdir -p $d
for t in 1 2 3; do KAGGLE_API_TOKEN=$(cat ~/.kaggle/tok_$tacc) kaggle kernels output $tacc/zc-$slug-1 -p $d -o --file-pattern ".*$name/(adapter/.*|result\.json|train_metrics\.json)$" > $d.log 2>&1 && break; sleep 30; done
ad=$(dirname "$(find $d -name adapter_model.safetensors | head -1)")
[ -f "$ad/adapter_config.json" ] || { log "adapter missing in the kernel output"; exit 1; }
rm -rf training/data/out/$dir/$name; mkdir -p training/data/out/$dir/$name; mv $ad training/data/out/$dir/$name/adapter
cp $(find $d -name train_metrics.json | head -1) $(find $d -name result.json | head -1) training/data/out/$dir/$name/ 2>/dev/null; rm -rf $d
log "adapter $(du -sh training/data/out/$dir/$name/adapter | cut -f1); $(python3 -c "import json;m=json.load(open('training/data/out/$dir/$name/train_metrics.json'));print({k:m.get(k) for k in ('tokens','tok_s','loss_first','loss_last','partial','steps')})" 2>/dev/null)"
sname=$name; [ "$w" != "1.0" ] && sname="${name}W${w#0.}"
Process/tools/p5/after_alpha.sh $a0 0 $tag $sname $w 1 $name $dir > Process/runs/p5/after_alpha_$a0.log 2>&1 &
sleep 240
Process/tools/p5/after_alpha.sh $a1 1 $tag $sname $w 1 $name $dir > Process/runs/p5/after_alpha_$a1.log 2>&1 &
wait
