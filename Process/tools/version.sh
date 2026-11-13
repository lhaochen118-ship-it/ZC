#!/bin/bash
# version.sh <acc> <src_dir> <slug>: new version of <acc>/<slug> from a staged folder (subfolders zipped)
acc=$1; src=$2; slug=$3
echo "{\"title\":\"$slug\",\"id\":\"$acc/$slug\",\"licenses\":[{\"name\":\"CC0-1.0\"}]}" > $src/dataset-metadata.json
KAGGLE_API_TOKEN=$(cat ~/.kaggle/tok_$acc) kaggle datasets version -p $src -m "keep the <name>/adapter layout (second folder)" -r zip -q 2>&1 | tail -1
