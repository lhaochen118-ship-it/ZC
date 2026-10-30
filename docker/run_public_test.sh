#!/bin/bash
# Run the image exactly like the organisers' re-run: no network, test set read-only at /data, outputs in /result.
# Usage: docker/run_public_test.sh <test_split_dir> [result_dir] [tag]
#   test_split_dir = the folder that contains questions.jsonl, rulebook.md, calendar.csv and data/
#   extra docker options via DOCKER_ARGS, e.g. DOCKER_ARGS="-e K=2 -e MAX_LEN=16384"
set -euo pipefail
DATA=$(realpath "$1"); RESULT=$(realpath -m "${2:-./result_public}"); TAG=${3:-datamate:latest}
mkdir -p "$RESULT"
time docker run --rm --gpus all --network none --shm-size 16g \
    ${DOCKER_ARGS:-} -v "$DATA":/data:ro -v "$RESULT":/result "$TAG"
echo "answers: $(wc -l < "$RESULT/submission.jsonl")  ->  upload $RESULT/submission.jsonl to the portal"
