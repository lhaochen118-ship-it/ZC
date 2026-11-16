#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m pip install -q -U "huggingface_hub>=0.25" kaggle
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3.5-9B", local_dir="models/Qwen3.5-9B", allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "LICENSE", "README.md"])
PY
rm -rf models/Qwen3.5-9B/.cache
python3 - <<'PY'
import json, hashlib, os, sys
h = json.load(open("hashes/model.json"))
bad = []
for f, ref in h["files"].items():
    d = hashlib.sha256()
    with open(os.path.join("models/Qwen3.5-9B", f), "rb") as fi:
        for chunk in iter(lambda: fi.read(16 << 20), b""): d.update(chunk)
    if d.hexdigest() != ref["sha256"]: bad.append(f)
print("base model sha256:", "OK" if not bad else f"MISMATCH {bad}"); sys.exit(1 if bad else 0)
PY
fetch() {
  local ds=$1 ref=$2 out=$3
  rm -rf "$out" models/_dl; mkdir -p "$out"
  kaggle datasets download -d "$ds" -p models/_dl --unzip -q
  python3 - "$ref" "$out" <<'PY'
import json, hashlib, os, shutil, glob, sys
ref, out = json.load(open(sys.argv[1]))["files"], sys.argv[2]
bad = []
for rel, h in ref.items():
    c = glob.glob("models/_dl/**/" + rel, recursive=True) or glob.glob("models/_dl/" + rel)
    if not c or hashlib.sha256(open(c[0], "rb").read()).hexdigest() != h: bad.append(rel); continue
    dst = os.path.join(out, rel); os.makedirs(os.path.dirname(dst) or ".", exist_ok=True); shutil.copy(c[0], dst)
shutil.rmtree("models/_dl")
print(out, "sha256:", "OK" if not bad else f"MISMATCH {bad}"); sys.exit(1 if bad else 0)
PY
}
fetch "$(python3 -c "import json; print(json.load(open('hashes/adapter.json'))['dataset'])")" hashes/adapter.json models/adapter
mkdir -p models/adapters
du -sh models/*
