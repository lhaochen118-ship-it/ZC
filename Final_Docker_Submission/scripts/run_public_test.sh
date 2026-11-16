#!/bin/bash
set -euo pipefail
DATA=$(realpath "$1"); RESULT=$(realpath -m "${2:-./result_public}"); TAG=${3:-datamate:latest}
mkdir -p "$RESULT"
time docker run --rm --gpus all --network none --shm-size 16g --ipc host ${DOCKER_ARGS:-} -v "$DATA":/data:ro -v "$RESULT":/result "$TAG"
echo "answers: $(wc -l < "$RESULT/submission.jsonl") -> $RESULT/submission.jsonl"
