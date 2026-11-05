#!/bin/bash
# Regression suite of the deterministic stack (CPU, ~15 min on 4 cores): question cleaning of train regenerated, qnorm unit
# test, canonical train / public scores, and stress tests (questions without accents, chat noise, READMEs blanked).
# Usage: S=<scratch with cache_train / cache_pub> bash tests/eval_all.sh [quick]
#   quick: canonical train + public only.
# The public reference file (data/raw/public_gt.jsonl) is used only to SCORE; nothing in src/ reads it.
set -uo pipefail
cd "$(dirname "$0")/.."
: "${S:?set S to the scratch dir holding cache_train / cache_pub}"
mkdir -p "$S/ana"
W=${WORKERS:-2}
run() { python3 tests/eval_split.py "$@" --workers "$W" 2>&1 | grep -v "^finalize" | grep TOTAL; }
if [ "${1:-}" != "quick" ]; then
  timeout 1800 python3 question_cleaning/make.py data/raw/train/train --cache "$S/cache_train" > "$S/ana/qclean_make.log" 2>&1
  echo "question cleaning: $(tail -1 "$S/ana/qclean_make.log" | cut -c1-120) | changed lines vs git: $(git diff question_cleaning/train_questions_clean.jsonl | grep -c '^[-+]{')"
  echo "test_qnorm: $(timeout 900 python3 tests/test_qnorm.py data/raw/train/train "$S/t_qn" 2>&1 | tail -1)"
fi
echo "train  canonical : $(run data/raw/train/train "$S/cache_train" data/raw/train/train/answers.jsonl --map auto --clean question_cleaning/train_questions_clean.jsonl --out "$S/ana/r_train.jsonl")"
echo "public canonical : $(run data/raw/public_test/public_test "$S/cache_pub" data/raw/public_gt.jsonl --map auto --clean qnorm --out "$S/ana/r_pub.jsonl")"
[ "${1:-}" = "quick" ] && exit 0
for pq in accentfree noise:1 noise:2 noise:3; do
  echo "public $pq : $(run data/raw/public_test/public_test "$S/cache_pub" data/raw/public_gt.jsonl --map auto --perturb-q "$pq" --out "$S/ana/r_pub_${pq/:/_}.jsonl")"
done
echo "public noreadme  : $(run data/raw/public_test/public_test "$S/cache_pub" data/raw/public_gt.jsonl --map auto --clean qnorm --perturb-data noreadme --out "$S/ana/r_pub_noreadme.jsonl")"
echo "train  noreadme  : $(run data/raw/train/train "$S/cache_train" data/raw/train/train/answers.jsonl --map auto --clean question_cleaning/train_questions_clean.jsonl --perturb-data noreadme --out "$S/ana/r_train_noreadme.jsonl")"
