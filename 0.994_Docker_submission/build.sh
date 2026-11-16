#!/bin/bash
# Build the solution image from the repo root. Usage: 0.994_Docker_submission/build.sh [tag]
set -euo pipefail
TAG=${1:-datamate_p:latest}; MODEL_NAME=Qwen3.5-9B
cd "$(dirname "$0")/.."
[ -f "models/$MODEL_NAME/config.json" ] || { echo "models/$MODEL_NAME missing: run 0.994_Docker_submission/fetch_assets.sh first"; exit 1; }
[ -f models/adapter_p/soup.json ] || { echo "models/adapter_p missing: run fetch_assets.sh first"; exit 1; }
docker build -f 0.994_Docker_submission/Dockerfile --build-arg MODEL_NAME="$MODEL_NAME" -t "$TAG" .
docker images "$TAG"
