#!/bin/bash
# the rest of tests/run_all.sh from test_grammar on (same commands, same scratch), after installing xgrammar
cd /home/user/ZC; S=$S/rall; SPLIT=data/raw/train/train; export CACHE=$S/cache_train
run() { local name=$1 t=$2; shift 2; local out; out=$(timeout "$t" "$@" 2>&1); local rc=$?
  if [ $rc -ne 0 ] || ! echo "$out" | tail -1 | grep -q PASS; then echo "$name FAIL (rc=$rc)"; echo "$out" | tail -25; exit 1; fi; echo "$name PASS"; }
for d in t_spec t_gram t_hint t_tool t_pre t_sql t_wp9; do mkdir -p $S/$d; ln -sfn $S/canon $S/$d/canon; done
run test_grammar 1800 python3 tests/test_grammar.py $SPLIT $S/t_gram
run test_hints 1800 python3 tests/test_hints.py $SPLIT $S/t_hint
run test_tools 600 python3 tests/test_tools.py $SPLIT $S/t_tool
run test_precheck 1800 python3 tests/test_precheck.py $SPLIT $S/t_pre
run test_sqlesc 300 python3 tests/test_sqlesc.py $SPLIT $S/t_sql
run test_wp9_fixes 600 python3 tests/test_wp9_fixes.py $SPLIT $S/t_wp9
run test_p2_fixes 600 python3 tests/test_p2_fixes.py $SPLIT
run test_predict_args 120 python3 tests/test_predict_args.py
run test_session_limit 300 python3 tests/test_session_limit.py $S/t_sess
run test_flow_hybrid 2400 python3 tests/test_flow_hybrid.py $SPLIT $S/t_flow
echo "ALL PASS"
