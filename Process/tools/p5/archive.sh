#!/bin/bash
# archive a finished P5 run: submission / result / judge ensemble / logs (gz) -> Process/runs/p5_results/<run>/
cd /home/user/ZC; r=$1; src=$(ls -d Process/runs/p5/$r/*_out/results/$r 2>/dev/null | head -1); [ -z "$src" ] && { echo "no results for $r"; exit 1; }
dst=Process/runs/p5_results/$r; mkdir -p $dst
cp $src/gpu_summary.json $dst/ 2>/dev/null
for c in $src/*/; do n=$(basename $c); [ -f $c/submission.jsonl ] || continue; mkdir -p $dst/$n
  for f in submission.jsonl result.json; do [ -f $c/$f ] && cp $c/$f $dst/$n/; done
  [ -f $c/judge/ensemble/submission.jsonl ] && cp $c/judge/ensemble/submission.jsonl $dst/$n/judge_ensemble_submission.jsonl
  for f in solver_logs.jsonl cards_unit.jsonl links.jsonl; do [ -f $c/$f ] && gzip -c $c/$f > $dst/$n/$f.gz; done
  [ -f $c/logs/run.log ] && cp $c/logs/run.log $dst/$n/run.log
  [ -d $c/lrl ] && tar czf $dst/$n/lrl_cards.tgz -C $c lrl
done; du -sh $dst
