#!/bin/bash
# P5 plan watcher: one line per finished kernel of Process/runs/p5 -> Process/runs/p5/watch.out (Monitor tails it)
while true; do python3 /home/user/ZC/automation/watch_done.py /home/user/ZC/Process/runs/p5 --max 110 >> /home/user/ZC/Process/runs/p5/watch.out 2>&1; sleep 5; done
