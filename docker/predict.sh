#!/bin/bash
# DataMate v3 entry point (organisers' re-run): reads the test set from /data (read-only), writes
# /result/submission.jsonl and one JSON log per question in /result/logs/ (id, final answer, elapsed time, programs,
# execution output / votes). Overridable with docker run -e NAME=value (defaults = the submitted configuration).
#
# src/predict_flow.py: stage 1 = rules on CPU (src/rules_first.py: automap mapping + qnorm cleaning + qparse rule programs
# over the dm / industry toolkit; every answer final; train 3000 / 3000, public 1000 / 1000); stage 2 = Qwen3.5-9B
# (agent.py, the only model) on the questions the rules leave, multi-run vote while time allows, re-merged every 20 s.
#   MODEL   model directory                 (default /models/Qwen3.5-9B)
#   BUDGET  seconds for both stages         (default 20400 = 5 h 40 min of the 6 h limit)
#   RUNS    model runs voted at most        (default 3)
#   LLM     0 = rules only                  (default 1)
#   DP      model copies (one per GPU)      (default: every GPU)
#   EXTRA   extra agent.py flags
set -euo pipefail
DATA=${DATA:-/data}; RESULT=${RESULT:-/result}
SPLIT=$(dirname "$(find "$DATA" -name questions.jsonl | head -1)")
mkdir -p "$RESULT/logs" /tmp/zc
# shellcheck disable=SC2086
exec python3 /app/src/predict_flow.py "$SPLIT" "$RESULT" --model "${MODEL:-/models/Qwen3.5-9B}" --budget "${BUDGET:-20400}" \
    --runs "${RUNS:-3}" --llm "${LLM:-1}" --dp "${DP:-0}" --work /tmp/zc -- ${EXTRA:-}
