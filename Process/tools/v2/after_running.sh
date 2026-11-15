#!/bin/bash
# after_running.sh <runs_dir> <run_name> <plan.json>: wait until <run_name>'s kernel reports RUNNING (or finished), then launch <plan>
R=$1; N=$2; P=$3
for i in $(seq 1 240); do
  if grep -qhE 'KernelWorkerStatus.(RUNNING|COMPLETE|ERROR)|STATUS=DONE' $R/$N/*.log 2>/dev/null; then break; fi; sleep 15
done
cd /home/user/ZC && python3 automation/launch_many.py $R $P >> $R/launch_$(basename $P .json).log 2>&1
