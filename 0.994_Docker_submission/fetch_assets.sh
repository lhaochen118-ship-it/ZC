#!/bin/bash
# Download the model weights into ./models/ (run once, from the repo root, needs internet; ~19.3 GB for Qwen3.5-9B) and the
# fine-tuned LoRA adapter (WP7, ~0.6 GB) into ./models/adapter/ from the project's Kaggle dataset named in 0.994_Docker_submission/assets.env
# (needs the kaggle CLI and a token in ~/.kaggle/kaggle.json); 0.994_Docker_submission/Dockerfile merges it into the base weights at build time.
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

# --- system P adapter (0.994_Docker_submission/assets.env ADAPTER_DS: WG1 + soup.json x 0.35), every file checked against
#     0.994_Docker_submission/adapter_hashes.json; into ./models/adapter_p/ (separate from the 0.952 image's models/adapter/)
ADAPTER_DS=$(sed -n 's/^ADAPTER_DS=//p' 0.994_Docker_submission/assets.env | head -1)
python3 -m pip install -q -U kaggle
rm -rf models/adapter_p models/_p_dl; mkdir -p models/adapter_p
kaggle datasets download -d "$ADAPTER_DS" -p models/_p_dl --unzip -q
python3 - <<'PY'
import json, hashlib, os, shutil, glob
ref = json.load(open("0.994_Docker_submission/adapter_hashes.json"))["files"]
bad = []
for rel, h in ref.items():
    c = glob.glob("models/_p_dl/**/" + rel, recursive=True) or glob.glob("models/_p_dl/" + rel)
    if not c or hashlib.sha256(open(c[0], "rb").read()).hexdigest() != h: bad.append(rel); continue
    dst = os.path.join("models/adapter_p", rel); os.makedirs(os.path.dirname(dst) or ".", exist_ok=True); shutil.copy(c[0], dst)
shutil.rmtree("models/_p_dl")
print("adapter sha256 check:", "OK" if not bad else f"MISMATCH {bad}"); raise SystemExit(1 if bad else 0)
PY
du -sh models/adapter_p
