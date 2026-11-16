#!/bin/bash
set -euo pipefail
TAG=${1:-datamate:latest}; OUT=${2:-datamate_solution.tar.gz}
docker save "$TAG" | gzip -1 > "$OUT"
sha256sum "$OUT" | tee "$OUT.sha256"
ls -lh "$OUT"
