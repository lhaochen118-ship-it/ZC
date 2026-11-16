#!/bin/bash
set -euo pipefail
TAG=${1:-datamate:latest}
cd "$(dirname "$0")/.."
[ -f models/Qwen3.5-9B/config.json ] && [ -f models/adapter/soup.json ] || { echo "models missing: run scripts/fetch_assets.sh first"; exit 1; }
docker build -t "$TAG" .
docker images "$TAG"
