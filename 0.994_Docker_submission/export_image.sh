#!/bin/bash
# Save the image as a compressed tarball + SHA-256 checksum for the "Solution Submission" form (Google Drive link).
# Usage: docker/export_image.sh [tag] [out_file]
set -euo pipefail
TAG=${1:-datamate:latest}; OUT=${2:-datamate_solution.tar.gz}
docker save "$TAG" | gzip -1 > "$OUT"
sha256sum "$OUT" | tee "$OUT.sha256"
ls -lh "$OUT"
