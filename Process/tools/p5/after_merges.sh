#!/bin/bash
# wait until the named merge kernels are DONE with every LoRA module merged, then launch the plan (P5 16/11)
cd /home/user/ZC; plan=$1; shift
for s in "$@"; do
  until grep -q "STATUS=DONE" Process/runs/p5/$s/zc-$s-1.log 2>/dev/null; do sleep 60; done
  if ! grep -rqs "modules merged into" Process/runs/p5/$s/; then echo "$(date -u +%H:%M) $s: merge not confirmed in pulled logs (continuing: weights stay on Kaggle)"; fi
  echo "$(date -u +%H:%M) $s done"
done
python3 automation/launch_many.py Process/runs/p5 $plan
for n in $(python3 -c "import json,sys;print(' '.join(x['name'] for x in json.load(open('$plan'))))"); do grep -qx $n Process/runs/p5/active_runs.txt || echo $n >> Process/runs/p5/active_runs.txt; done
