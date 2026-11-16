#!/bin/bash
set -euo pipefail
OUT=${1:-./public_test_data}
cd "$(dirname "$0")/.."
python3 -m pip install -q -U kaggle
rm -rf "$OUT" _pt_dl; mkdir -p _pt_dl
kaggle datasets download -d nhatlong1103/zc-2026 -p _pt_dl --unzip -q
SRC=$(dirname "$(find _pt_dl -path '*public_test*' -name questions.jsonl | head -1)")
[ -f "$SRC/questions.jsonl" ] && [ -f "$SRC/rulebook.md" ] && [ -d "$SRC/data" ] || { echo "public_test not found in the dataset"; exit 1; }
mv "$SRC" "$OUT"; rm -rf _pt_dl
echo "$(wc -l < "$OUT/questions.jsonl") questions, $(ls "$OUT/data" | wc -l) shops -> $(realpath "$OUT")"
ls "$OUT"
