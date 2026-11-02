#!/bin/bash
# Full CPU test suite (no GPU); every test must PASS before a Kaggle launch (automation/preflight.py runs this).
set -euo pipefail
cd "$(dirname "$0")/.."
S=${SCRATCH:-$(mktemp -d)}
SPLIT=data/raw/train/train
[ -d "${CACHE:-$S/cache_train}" ] || python3 src/ingest.py $SPLIT ${CACHE:-$S/cache_train} 4 > /dev/null
python3 tests/test_ingest_robust.py | tail -1
python3 tests/test_engine_batching.py | tail -1
timeout 300 python3 tests/test_sandbox.py 2>/dev/null | tail -1
python3 tests/test_agent_v2.py $SPLIT $S/t_v2 2>&1 | tail -1 | grep -q PASS && echo "test_agent_v2 PASS"
python3 tests/test_plausibility.py $SPLIT ${CACHE:-$S/cache_train} 2>&1 | tail -1 | grep -q PASS && echo "test_plausibility PASS"
timeout 1800 python3 tests/test_pipeline.py $SPLIT $S/t_pipe 2>&1 | tail -1 | grep -q PASS && echo "test_pipeline PASS"
echo "ALL PASS"
