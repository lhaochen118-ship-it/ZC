#!/bin/bash
set -euo pipefail
TAG=${1:-datamate:latest}; OUT=${2:-datamate_solution.tar.gz}
docker save "$TAG" | gzip -1 > "$OUT"
md5sum "$OUT" | tee "$OUT.md5"
ls -lh "$OUT"
