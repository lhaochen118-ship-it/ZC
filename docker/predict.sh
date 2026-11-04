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
K=${K:-4}; KC=${KC:-8}; QW=${QW:-4}; ESC=${ESC:-4}; BUDGET=${BUDGET:-20400}; DP=${DP:-$NGPU}; QUANT=${QUANT:-fp8}; PIPE=${PIPE:-1}; EXTRA=${EXTRA:-}
[ "$PIPE" = "1" ] && EXTRA="--pipeline $EXTRA"
[ -n "$QUANT" ] && EXTRA="--quant $QUANT $EXTRA"
# Multi-run vote (src/vote.py): run 1 writes /result incrementally (a time-out still leaves its answers); while the budget
# allows another run of the same length (+15 %), run i (seed base i) goes to /tmp/zc/run_i and the plurality over every
# finished run replaces /result/submission.jsonl atomically (train, 4 runs: PA 0.980 vs <= 0.969 for one run).
RUNS=${RUNS:-3}; T0=$(date +%s)
echo "split: $SPLIT  model: $MODEL  gpus: $NGPU  runs: up to $RUNS  start: $(date -u +%FT%TZ)"
run_agent() {   # run_agent <out_dir> <seed_base> <budget_s> <log_dir>
    python3 /app/src/agent.py "$SPLIT" "$1" --model "$MODEL" --dp "$DP" --k "$K" --kc "$KC" --repairs 2 --shots 4 --qw "$QW" \
        --think-map --escalate "$ESC" --confirm 1 --max-len 24576 --max-tokens 1024 --compile-tokens 6144 --budget "$3" \
        --cache /tmp/zc/cache --seed-base "$2" --log-dir "$4" $EXTRA 2>&1 | tee -a "$RESULT/logs/run.log"
}
run_agent "$RESULT" 0 "$BUDGET" "$RESULT/logs"
cp "$RESULT/submission.jsonl" "$RESULT/logs/submission_run1.jsonl" 2>/dev/null || true
DIRS="$RESULT"; LAST=$(( $(date +%s) - T0 ))
for i in $(seq 1 $(( RUNS - 1 ))); do
    LEFT=$(( BUDGET - ($(date +%s) - T0) ))
    if [ $(( LAST * 115 / 100 )) -ge "$LEFT" ]; then echo "run $((i + 1)) skipped: ${LEFT}s left, a run takes ${LAST}s"; break; fi
    S0=$(date +%s); mkdir -p "/tmp/zc/run_$i/logs"
    run_agent "/tmp/zc/run_$i" "$i" "$LEFT" "/tmp/zc/run_$i/logs"; LAST=$(( $(date +%s) - S0 ))
    DIRS="$DIRS /tmp/zc/run_$i"
    python3 /app/src/vote.py "$RESULT/submission.jsonl" $DIRS --log-dir "$RESULT/logs" 2>&1 | tee -a "$RESULT/logs/run.log"
done
# Consensus re-answer (src/consensus.py, CPU): the mapping of each business chosen once among the candidates of ALL runs, every
# logged program of every run executed again on it, votes redone, then the multi-run vote. CONSENSUS=1 to use it; it replaces
# the submission only when it finishes in the time left (the plain vote stays otherwise).
if [ "${CONSENSUS:-0}" = "1" ] && [ "$(echo $DIRS | wc -w)" -ge 2 ]; then
    LEFT=$(( BUDGET - ($(date +%s) - T0) ))
    if [ "$LEFT" -gt 900 ]; then
        mkdir -p /tmp/zc/consensus
        if timeout $(( LEFT - 300 )) python3 /app/src/consensus.py "$SPLIT" /tmp/zc/cache /tmp/zc/consensus /tmp/zc/consensus/submission.jsonl $DIRS \
                2>&1 | tee -a "$RESULT/logs/run.log" && [ -s /tmp/zc/consensus/submission.jsonl ]; then
            cp "$RESULT/submission.jsonl" "$RESULT/logs/submission_vote.jsonl"
            cp /tmp/zc/consensus/submission.jsonl "$RESULT/submission.jsonl.tmp" && mv "$RESULT/submission.jsonl.tmp" "$RESULT/submission.jsonl"
        fi
    else echo "consensus skipped: ${LEFT}s left"; fi
fi
echo "end: $(date -u +%FT%TZ)  answers: $(wc -l < "$RESULT/submission.jsonl")"
