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
export CACHE
[ -d "$S/canon" ] || python3 tests/build_canon.py $SPLIT $CACHE $S/canon 3 > /dev/null 2>&1   # automap tables, shared by the hybrid tests
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
run test_mode_h 120 python3 tests/test_mode_h.py
run test_judge_constrained 120 python3 tests/test_judge_constrained.py
run test_force_json 60 python3 tests/test_force_json.py
run test_merge_first_decides 60 python3 tests/test_merge_first_decides.py
run test_units_multi 60 python3 tests/test_units_multi.py
run test_agent_v2 900 python3 tests/test_agent_v2.py $SPLIT $S/t_v2
run test_plausibility 900 python3 tests/test_plausibility.py $SPLIT $CACHE
run test_qnorm 900 python3 tests/test_qnorm.py $SPLIT $S/t_qn
run test_mapping_fixes 300 python3 tests/test_mapping_fixes.py
run test_industry 120 python3 tests/test_industry.py
run test_vote 120 python3 tests/test_vote.py $S/t_vote
run test_rules_first 1800 python3 tests/test_rules_first.py $SPLIT $S/t_rules
run test_canon_flow 1500 python3 tests/test_canon_flow.py $SPLIT $S/t_canon
run test_pipeline 3600 python3 tests/test_pipeline.py $SPLIT $S/t_pipe
# hybrid flow (docs/08 WP2): scratch dirs share the automap tables ($S/canon)
for d in t_spec t_gram t_hint t_tool t_pre t_sql t_wp9; do mkdir -p $S/$d; ln -sfn $S/canon $S/$d/canon; done
run test_api_spec 600 python3 tests/test_api_spec.py $SPLIT $S/t_spec
run test_grammar 1800 python3 tests/test_grammar.py $SPLIT $S/t_gram
run test_hints 1800 python3 tests/test_hints.py $SPLIT $S/t_hint
run test_tools 600 python3 tests/test_tools.py $SPLIT $S/t_tool
run test_precheck 1800 python3 tests/test_precheck.py $SPLIT $S/t_pre
run test_sqlesc 300 python3 tests/test_sqlesc.py $SPLIT $S/t_sql
run test_wp9_fixes 600 python3 tests/test_wp9_fixes.py $SPLIT $S/t_wp9
run test_p2_fixes 600 python3 tests/test_p2_fixes.py $SPLIT
run test_lrl 600 python3 tests/test_lrl.py
run test_predict_args 120 python3 tests/test_predict_args.py
run test_session_limit 300 python3 tests/test_session_limit.py $S/t_sess
run test_units 900 python3 tests/test_units.py data/raw/public_test/public_test $S/t_units
run test_selector_flow 120 python3 tests/test_selector_flow.py
run test_flow_hybrid 2400 python3 tests/test_flow_hybrid.py $SPLIT $S/t_flow
echo "ALL PASS"
