#!/bin/bash
# Plan 3 SFT v3b data (docs/16 §3): v3a + learned rule cards in the prompts (DC) + new rulebook terms (DR) + loanword spellings (DT).
# Usage: build_v3b.sh <scratch S> <LRL results dir>[,<dir>...] <train logs.jsonl,...>
#   S holds cache_train/, rall/canon/, ext/vilexnorm/ (handoff/ENVIRONMENT_SUMMARY.md); writes training/data/out/v3b/v3b_mix.jsonl
set -e
S=$1; RES=$2; LOGS=$3
cd "$(dirname "$0")/../.."
T=data/raw/train/train; C=$S/cache_train; K=$S/rall/canon; O=training/data/out; mkdir -p $O/v3b
# 1. card files of the flow (src/refs/lrl) + induced candidates
python3 training/data/lrl_collect.py "$RES" --induced-out $O/v3b/induced_cand.jsonl
# 2. induced CHECK cards accepted on logged training samples (clean signal, other shops only)
if [ -f $O/v3b/induced_cand.jsonl ]; then
  python3 tests/lrl_accept.py scan $O/v3b/induced_cand.jsonl "$LOGS" src/refs/lrl/induced.jsonl --report $O/v3b/scan_report.json
else : > src/refs/lrl/induced.jsonl; fi
CARDS=src/refs/lrl/compile_train.jsonl,src/refs/lrl/alias_train.jsonl,src/refs/lrl/translit.jsonl,src/refs/lrl/induced.jsonl
# 3. DC: the v3a records re-rendered with the cards their question matches (+ distractors)
python3 training/data/dc_cards.py $T $C $K $O/v3a/v3a_mix.jsonl $O/v3b/v3a_dc.jsonl --cards $CARDS
# 4. DR (new rulebook terms; also writes the tr_newterm measurement set)
NT=$(ls $(echo "$RES" | tr ',' ' ') -d 2>/dev/null | xargs -I{} find {} -name 'cards_newterm*.jsonl' | tr '\n' ',' | sed 's/,$//')
if [ -n "$NT" ]; then python3 training/data/dr_newterm.py $T $C $K "$NT" $O/v3b/dr.jsonl; fi
# 5. DT: loanword spellings of the model's own list on D1 / D2 questions
python3 training/data/dp_style.py $T $C $K $O/wg/d1_full.jsonl,$O/wg/d2_full.jsonl $O/v3b/style_t.jsonl --kinds translit --translit src/refs/lrl/translit.jsonl --spec
# 6. mix
P="$O/v3b/v3a_dc.jsonl:0"
[ -f $O/v3b/dr.jsonl ] && P="$P,$O/v3b/dr.jsonl:600"
[ -s $O/v3b/style_t.jsonl ] && P="$P,$O/v3b/style_t.jsonl:500"
python3 training/data/v3_mix.py $O/v3b/v3b_mix.jsonl "$P"
