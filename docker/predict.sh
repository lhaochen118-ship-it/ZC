#!/bin/bash
# DataMate solution entry point (organisers' re-run): reads the test set from /data (read-only), writes
# /result/submission.jsonl (appended chunk by chunk, so a time-out keeps every finished answer) and one log per
# question in /result/logs/ (id, final answer, elapsed time, generated programs, execution output, votes).
set -euo pipefail
DATA=${DATA:-/data}; RESULT=${RESULT:-/result}
SPLIT=$(dirname "$(find "$DATA" -name questions.jsonl | head -1)")
MODEL=${MODEL:-/models/Qwen3-14B-FP8}
NGPU=$(nvidia-smi -L | wc -l)
mkdir -p "$RESULT/logs" /tmp/zc
echo "split: $SPLIT  model: $MODEL  gpus: $NGPU  start: $(date -u +%FT%TZ)"
# 6 h limit including model loading: budget 5 h 40 min for the agent, chunks of 4 businesses.
# overridable with docker run -e NAME=value (defaults = the submitted configuration)
K=${K:-4}; KC=${KC:-4}; MAX_LEN=${MAX_LEN:-32768}; BUDGET=${BUDGET:-20400}; DP=${DP:-$NGPU}; EXTRA=${EXTRA:-}
python3 /app/src/agent.py "$SPLIT" "$RESULT" --model "$MODEL" --dp "$DP" --compile --k "$K" --kc "$KC" --repairs 2 \
    --chunk 4 --budget "$BUDGET" --cache /tmp/zc/cache --log-dir "$RESULT/logs" --max-len "$MAX_LEN" $EXTRA 2>&1 | tee "$RESULT/logs/run.log"
echo "end: $(date -u +%FT%TZ)  answers: $(wc -l < "$RESULT/submission.jsonl")"
