#!/bin/bash
# 16/11 (container restart lost Process/runs/p5): wait for a kernel launched by the old container, then write the runner-style
# log the chain scripts read (STATUS=DONE:complete|error) and pull its small outputs. Usage: poll_lost.sh <acc> <slug> [file regex]
cd /home/user/ZC
acc=$1; slug=$2; pat=${3:-'.*\.(log|json|jsonl|csv|txt)$'}; d=Process/runs/p5/$slug; mkdir -p $d
while true; do
  s=$(KAGGLE_API_TOKEN=$(cat ~/.kaggle/tok_$acc) timeout 120 kaggle kernels status $acc/zc-$slug-1 2>&1 | grep -oE 'KernelWorkerStatus\.[A-Z_]+' | head -1)
  case "$s" in *COMPLETE|*ERROR|*CANCEL*) break;; esac
  sleep 180
done
for t in 1 2 3; do KAGGLE_API_TOKEN=$(cat ~/.kaggle/tok_$acc) timeout 1200 kaggle kernels output $acc/zc-$slug-1 -p $d/zc-$slug-1_out -o --file-pattern "$pat" > $d/dl.log 2>&1 && break; sleep 30; done
st=error; [ "$s" = KernelWorkerStatus.COMPLETE ] && ! grep -q "STALE CODE" $d/zc-$slug-1_out/*.log 2>/dev/null && st=complete
echo "$(date -u +%H:%M) $acc/zc-$slug-1 $s (polled)" > $d/zc-$slug-1.log; echo "STATUS=DONE:$st" >> $d/zc-$slug-1.log
