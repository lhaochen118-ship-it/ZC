#!/bin/bash
# Build the solution image from the repo root. Usage: "Docker submission/build.sh" [tag]
set -euo pipefail
TAG=${1:-datamate:latest}; MODEL_NAME=Qwen3.5-9B
cd "$(dirname "$0")/.."
[ -f "models/$MODEL_NAME/config.json" ] || { echo "models/$MODEL_NAME missing: run \"Docker submission/fetch_assets.sh\" first"; exit 1; }
[ -f models/adapter/soup.json ] || [ -f models/adapter/adapter_config.json ] || { echo "models/adapter missing: run fetch_assets.sh first"; exit 1; }
[ -f models/adapters/WL1/adapter/adapter_model.safetensors ] && [ -f models/adapters/WJR1/adapter/adapter_model.safetensors ] || { echo "models/adapters (WL1, WJR1) missing: run fetch_assets.sh first"; exit 1; }
docker build -f "Docker submission/Dockerfile" --build-arg MODEL_NAME="$MODEL_NAME" -t "$TAG" .
docker images "$TAG"
