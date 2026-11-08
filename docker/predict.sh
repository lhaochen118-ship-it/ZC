#!/bin/bash
# DataMate v3 entry point (organisers' re-run): reads the test set from /data (read-only), writes
# /result/submission.jsonl and one JSON log per question in /result/logs/ (id, final answer, elapsed time, programs,
# execution output / votes). Overridable with docker run -e NAME=value (defaults = the submitted configuration).
#
# src/predict_flow.py: the hybrid flow (docs/08_MASTER_PLAN.md §4.1): every question is answered by a program the model writes
# (schema by automap, hints, retrieved checked examples, precheck, tools, execution, vote, abstain); settings in CONFIG.
#   MODEL   model directory                 (default /models/Qwen3.5-9B: the fine-tuned merged weights when the image has them)
#   CONFIG  flow settings                   (default /app/configs/flow_final.json)
#   BUDGET  seconds for the whole run       (default 20400 = 5 h 40 min of the 6 h limit)
#   RUNS    independent runs voted at most  (default 1)
#   DP      model copies (one per GPU)      (default: every GPU)
#   EXTRA   extra agent.py flags
set -euo pipefail
DATA=${DATA:-/data}; RESULT=${RESULT:-/result}
SPLIT=$(dirname "$(find "$DATA" -name questions.jsonl | head -1)")
mkdir -p "$RESULT/logs" /tmp/zc
# shellcheck disable=SC2086
exec python3 /app/src/predict_flow.py "$SPLIT" "$RESULT" --model "${MODEL:-/models/Qwen3.5-9B}" --budget "${BUDGET:-20400}" \
    --runs "${RUNS:-1}" --dp "${DP:-0}" --work /tmp/zc --config "${CONFIG:-/app/configs/flow_final.json}" -- ${EXTRA:-}
