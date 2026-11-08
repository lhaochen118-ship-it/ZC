#!/bin/bash
# Download the model weights into ./models/ (run once, from the repo root, needs internet; ~19.3 GB for Qwen3.5-9B) and the
# fine-tuned LoRA adapter (WP7, ~0.6 GB) into ./models/adapter/ from the project's Kaggle dataset named in docker/assets.env
# (needs the kaggle CLI and a token in ~/.kaggle/kaggle.json); docker/Dockerfile merges it into the base weights at build time.
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

# --- fine-tuned adapter (docker/assets.env: ADAPTER_DS=<owner>/<dataset>, empty = base model only)
ADAPTER_DS=$(sed -n 's/^ADAPTER_DS=//p' docker/assets.env 2>/dev/null | head -1)
rm -rf models/adapter; mkdir -p models/adapter
if [ -z "$ADAPTER_DS" ]; then echo "no adapter in docker/assets.env: the image serves the base weights"; touch models/adapter/NONE; exit 0; fi
python3 -m pip install -q -U kaggle
kaggle datasets download -d "$ADAPTER_DS" -p models/_adapter_dl --unzip -q
SRC=$(dirname "$(find models/_adapter_dl -name adapter_config.json | head -1)")
cp "$SRC"/adapter_config.json "$SRC"/adapter_model.safetensors models/adapter/ && rm -rf models/_adapter_dl
python3 - <<'PY'
import json, hashlib
ref = json.load(open("src/adapter_hashes.json"))
bad = [f for f, h in ref["files"].items() if hashlib.sha256(open("models/adapter/" + f, "rb").read()).hexdigest() != h]
print("adapter sha256 check:", "OK" if not bad else f"MISMATCH {bad}"); raise SystemExit(1 if bad else 0)
PY
du -sh models/adapter
