#!/bin/bash
# mksub5.sh vNN [vCONS] : the 5 public runs of vNN (kernels 2..6 = pub, pub_s1..pub_s4)
#   vNN   = multi-run vote (src/vote.py) with the mapping tie-break recomputed by the current src/ (automation/map_penalties.py)
#   vCONS = consensus re-answer over the same runs (src/consensus.py: mapping chosen once among every run's candidates)
# -> validated submissions/<v>/submission.jsonl. Env S = scratch dir holding runs/ and cache_pub/ (public ingest cache).
set -e
V=$1; VC=${2:-}; S=${S:?set S to the scratch dir that holds runs/}; R=$S/runs/$V; cd /home/user/ZC
SPLIT=data/raw/public_test/public_test
DIRS="$R/zc-$V-2_out/results/pub $R/zc-$V-3_out/results/pub_s1 $R/zc-$V-4_out/results/pub_s2 $R/zc-$V-5_out/results/pub_s3 $R/zc-$V-6_out/results/pub_s4"
DIRS=$(for d in $DIRS; do [ -s "$d/logs.jsonl" ] && echo "$d"; done | tr '\n' ' ')
echo "runs: $DIRS"
[ -d "$S/cache_pub" ] || python3 src/ingest.py $SPLIT $S/cache_pub 4 > /dev/null
python3 automation/map_penalties.py $S/cache_pub $DIRS > $S/mappen_$V.log 2>&1
T=$S/vote_$V; mkdir -p $T
python3 src/vote.py $T/submission.jsonl $DIRS
python3 automation/make_submission.py $T $SPLIT tmp_$V
mkdir -p submissions/$V && mv submissions/tmp_$V.jsonl submissions/$V/submission.jsonl
if [ -n "$VC" ]; then
    W=$S/cons_$V; mkdir -p $W
    python3 src/consensus.py $SPLIT $S/cache_pub $W $W/submission.jsonl $DIRS --workers 3 2>&1 | tee $S/cons_$V.log
    python3 automation/make_submission.py $W $SPLIT tmp_$VC
    mkdir -p submissions/$VC && mv submissions/tmp_$VC.jsonl submissions/$VC/submission.jsonl
fi
for v in $V $VC; do python3 -c "
import json,collections; c=collections.Counter(json.loads(l)['type'] for l in open('submissions/$v/submission.jsonl')); print('$v', dict(c))"; done
