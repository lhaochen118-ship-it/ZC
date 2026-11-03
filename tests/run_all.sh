#!/bin/bash
# Full CPU test suite (no GPU); every test must PASS before a Kaggle launch (automation/preflight.py runs this).
# Each test's last line must contain PASS, otherwise the suite stops with exit code 1 (no silent pass).
set -uo pipefail
cd "$(dirname "$0")/.."
if [ -n "${SCRATCH:-}" ]; then S=$SCRATCH; else
    S=$(mktemp -d); trap 'rm -rf "$S"' EXIT; trap 'exit 143' TERM INT   # own temp dir removed on exit / kill (22 GB once piled up)
fi
SPLIT=data/raw/train/train
CACHE=${CACHE:-$S/cache_train}
[ -d "$CACHE" ] || python3 src/ingest.py $SPLIT $CACHE 4 > /dev/null
run() {                                     # run <name> <timeout s> <command...>
    local name=$1 t=$2; shift 2
    local out; out=$(timeout "$t" "$@" 2>&1); local rc=$?
    if [ $rc -ne 0 ] || ! echo "$out" | tail -1 | grep -q PASS; then
        echo "$name FAIL (rc=$rc)"; echo "$out" | tail -25; exit 1
    fi
    echo "$name PASS"
}
run test_ingest_robust 300 python3 tests/test_ingest_robust.py
run test_engine_batching 300 python3 tests/test_engine_batching.py
run test_sandbox 300 python3 tests/test_sandbox.py
run test_agent_v2 900 python3 tests/test_agent_v2.py $SPLIT $S/t_v2
run test_plausibility 900 python3 tests/test_plausibility.py $SPLIT $CACHE
run test_qnorm 900 python3 tests/test_qnorm.py $SPLIT $S/t_qn
run test_mapping_fixes 300 python3 tests/test_mapping_fixes.py
run test_vote 120 python3 tests/test_vote.py $S/t_vote
run test_pipeline 3600 python3 tests/test_pipeline.py $SPLIT $S/t_pipe
echo "ALL PASS"
