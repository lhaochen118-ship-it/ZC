#!/bin/bash
# DataMate entry point (organisers' re-run): reads the test set from /data (read-only), writes /result/submission.jsonl (appended
# while running: every line already written counts if the run is stopped) and one JSON log per question in /result/logs/<id>.json
# (id, final answer, elapsed time, the programs the model wrote, execution output, votes, judge verdict) + /result/logs/run.log.
# Overridable with docker run -e NAME=value (defaults = the submitted configuration):
#   MODEL   served weights            (default /models/solver = Qwen3.5-9B + LoRA V3A x 0.70, merged at build time)
#   CONFIG  flow settings             (default /app/configs/flow_p5c4.json: the submitted configuration, public 0.939)
#   BUDGET  seconds for the whole run (default 20400 = 5 h 40 min of the 6 h limit, model loading included)
#   DP      model copies, one per GPU (default 0 = every visible GPU)
set -euo pipefail
DATA=${DATA:-/data}; RESULT=${RESULT:-/result}
SPLIT=$(dirname "$(find "$DATA" -name questions.jsonl | head -1)")
[ -f "$SPLIT/questions.jsonl" ] || { echo "no questions.jsonl under $DATA"; exit 1; }
mkdir -p "$RESULT/logs" /tmp/zc
exec python3 /app/src/predict_flow.py "$SPLIT" "$RESULT" --model "${MODEL:-/models/solver}" --budget "${BUDGET:-20400}" \
    --runs 1 --dp "${DP:-0}" --work /tmp/zc --config "${CONFIG:-/app/configs/flow_p5c4.json}"
