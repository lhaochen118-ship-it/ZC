#!/bin/bash
# when_ready.sh <acc> <dataset slug> <runs_dir> <plan.json> [after_run]: wait for <acc>/<slug> ready (and optionally for <after_run> RUNNING), then launch
A=$1; D=$2; R=$3; P=$4; AFTER=$5
for i in $(seq 1 120); do KAGGLE_API_TOKEN=$(cat ~/.kaggle/tok_$A) kaggle datasets status $A/$D 2>&1 | grep -q ready && break; sleep 30; done
if [ -n "$AFTER" ]; then for i in $(seq 1 240); do grep -qhE 'KernelWorkerStatus.(RUNNING|COMPLETE|ERROR)|STATUS=DONE' $R/$AFTER/*.log 2>/dev/null && break; sleep 15; done; fi
cd /home/user/ZC && ZC_POLL_S=180 python3 automation/launch_many.py $R $P >> $R/launch_$(basename $P .json).log 2>&1
