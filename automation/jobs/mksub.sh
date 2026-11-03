#!/bin/bash
# mksub.sh vNN : vote the 3 public runs of vNN (kernels 2/3/4) -> validate -> submissions/vNN/submission.jsonl
set -e
V=$1; S=${S:?set S to the scratch dir that holds runs/}; R=$S/runs/$V; cd /home/user/ZC
T=$S/vote_$V; mkdir -p $T
python3 src/vote.py $T/submission.jsonl $R/zc-$V-2_out/results/pub $R/zc-$V-3_out/results/pub_s1 $R/zc-$V-4_out/results/pub_s2
python3 automation/make_submission.py $T data/raw/public_test/public_test tmp_$V
mkdir -p submissions/$V && mv submissions/tmp_$V.jsonl submissions/$V/submission.jsonl
python3 -c "
import json,collections; c=collections.Counter(json.loads(l)['type'] for l in open('submissions/$V/submission.jsonl')); print(dict(c))"
