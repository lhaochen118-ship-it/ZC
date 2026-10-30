#!/bin/bash
# Build the solution image from the repo root. Usage: docker/build.sh [tag] [model folder name in ./models]
set -euo pipefail
TAG=${1:-datamate:latest}; MODEL_NAME=${2:-Qwen3-14B-FP8}
cd "$(dirname "$0")/.."
[ -f "models/$MODEL_NAME/config.json" ] || { echo "models/$MODEL_NAME missing: run docker/fetch_assets.sh first"; exit 1; }
docker build -f docker/Dockerfile --build-arg MODEL_NAME="$MODEL_NAME" -t "$TAG" .
docker images "$TAG"
