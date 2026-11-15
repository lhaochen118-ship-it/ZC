#!/bin/bash
# after_done.sh <run> <plan>: when <run> is reported DONE in watch.out, launch <plan>
S=/tmp/claude-0/-home-user-ZC/53d4fe9e-6712-52f3-bd72-e042dbdf5ede/scratchpad
until grep -q "^$1/.*DONE" $S/watch.out; do sleep 60; done
cd /home/user/ZC && ZC_POLL_S=180 python3 automation/launch_many.py $S/runs $2 >> $S/runs/launch_$(basename $2 .json).log 2>&1
