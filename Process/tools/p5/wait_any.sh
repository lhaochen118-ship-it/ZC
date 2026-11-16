#!/bin/bash
# exit as soon as one of the given runs gets a STATUS line it did not have at start (prints which)
cd /home/user/ZC
declare -A had; for r in "$@"; do had[$r]=$(grep -hs "STATUS=" Process/runs/p5/$r/*.log | wc -l); done
while true; do
  for r in "$@"; do n=$(grep -hs "STATUS=" Process/runs/p5/$r/*.log | wc -l); [ "$n" -gt "${had[$r]}" ] && { echo "$(date -u +%H:%M) $r: $(grep -hs STATUS= Process/runs/p5/$r/*.log | tail -1)"; exit 0; }; done
  sleep 60
done
