#!/bin/bash
# DataMate v3 entry point (organisers' re-run): reads the test set from /data (read-only), writes
# /result/submission.jsonl (every answer appended as soon as it is decided; merged from the GPU processes every few
# seconds) and one JSON log per question in /result/logs/ (id, final answer, elapsed time, generated programs,
# execution output, votes). Overridable with docker run -e NAME=value (defaults = the submitted configuration).
set -euo pipefail
DATA=${DATA:-/data}; RESULT=${RESULT:-/result}
SPLIT=$(dirname "$(find "$DATA" -name questions.jsonl | head -1)")
MODEL=${MODEL:-/models/Qwen3.5-9B}
NGPU=$(nvidia-smi -L | wc -l)
mkdir -p "$RESULT/logs" /tmp/zc
# 6 h limit includes start-up and model loading: 5 h 40 min budget for the agent (fast mode / abstain when short of time)
# defaults = the best configuration measured on train leave-one-out (v4b 02/11: PA 0.955; mapping fixes after it)
K=${K:-4}; KC=${KC:-6}; QW=${QW:-4}; ESC=${ESC:-4}; BUDGET=${BUDGET:-20400}; DP=${DP:-$NGPU}; QUANT=${QUANT:-fp8}; PIPE=${PIPE:-1}; EXTRA=${EXTRA:-}
[ "$PIPE" = "1" ] && EXTRA="--pipeline $EXTRA"
[ -n "$QUANT" ] && EXTRA="--quant $QUANT $EXTRA"
echo "split: $SPLIT  model: $MODEL  gpus: $NGPU  start: $(date -u +%FT%TZ)"
python3 /app/src/agent.py "$SPLIT" "$RESULT" --model "$MODEL" --dp "$DP" --k "$K" --kc "$KC" --repairs 2 --shots 4 --qw "$QW" \
    --think-map --escalate "$ESC" --max-len 24576 --max-tokens 1024 --compile-tokens 6144 --budget "$BUDGET" --cache /tmp/zc/cache \
    --log-dir "$RESULT/logs" $EXTRA 2>&1 | tee "$RESULT/logs/run.log"
echo "end: $(date -u +%FT%TZ)  answers: $(wc -l < "$RESULT/submission.jsonl")"
