#!/bin/bash
# Download the model weights into ./models/ (run once, from the repo root, needs internet; ~19.3 GB for Qwen3.5-9B).
set -euo pipefail
MODEL_ID=${1:-Qwen/Qwen3.5-9B}
NAME=$(basename "$MODEL_ID")
cd "$(dirname "$0")/.."
python3 -m pip install -q -U "huggingface_hub>=0.25"
python3 - "$MODEL_ID" "models/$NAME" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], local_dir=sys.argv[2], allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "LICENSE", "README.md"]))
PY
rm -rf "models/$NAME/.cache"
# the files must be the ones Hugging Face publishes (src/model_hashes.json, the same check as the Kaggle runs)
python3 - "$MODEL_ID" "models/$NAME" <<'PY'
import sys, json, hashlib, os
h = json.load(open("src/model_hashes.json"))
if h.get("repo") != sys.argv[1]: print("no reference hashes for", sys.argv[1]); sys.exit(0)
bad = []
for f, ref in h["files"].items():
    d = hashlib.sha256()
    with open(os.path.join(sys.argv[2], f), "rb") as fi:
        for chunk in iter(lambda: fi.read(16 << 20), b""): d.update(chunk)
    if d.hexdigest() != ref["sha256"]: bad.append(f)
print("sha256 check:", "OK" if not bad else f"MISMATCH {bad}"); sys.exit(1 if bad else 0)
PY
du -sh "models/$NAME"
