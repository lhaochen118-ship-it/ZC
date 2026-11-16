#!/bin/bash
set -euo pipefail
DATA=${DATA:-/data}; RESULT=${RESULT:-/result}
SPLIT=$(dirname "$(find "$DATA" -name questions.jsonl | head -1)")
[ -f "$SPLIT/questions.jsonl" ] || { echo "no questions.jsonl under $DATA"; exit 1; }
mkdir -p "$RESULT/logs" /tmp/zc
exec python3 /app/src/predict_flow.py "$SPLIT" "$RESULT" --model "${MODEL:-/models/solver}" --budget "${BUDGET:-20400}" \
    --runs 1 --dp "${DP:-0}" --work /tmp/zc --config "${CONFIG:-/app/configs/solution.json}"
