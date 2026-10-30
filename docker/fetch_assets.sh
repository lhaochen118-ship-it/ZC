#!/bin/bash
# Download the model weights into ./models/ (run once, from the repo root). Needs internet; ~16 GB for Qwen3-14B-FP8.
set -euo pipefail
MODEL_ID=${1:-Qwen/Qwen3-14B-FP8}
NAME=$(basename "$MODEL_ID")
cd "$(dirname "$0")/.."
python3 -m pip install -q -U "huggingface_hub>=0.25"
python3 - "$MODEL_ID" "models/$NAME" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], local_dir=sys.argv[2]))
PY
rm -rf "models/$NAME/.cache"
du -sh "models/$NAME"
