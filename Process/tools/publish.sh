#!/bin/bash
# publish.sh <acc> <src_dir> <slug>: create <acc>/<slug> (private) from a staged folder (subfolders zipped, Kaggle unpacks them)
acc=$1; src=$2; slug=$3; mkdir -p $src/info; echo "keeps the folder layout (Kaggle unwraps a lone top folder)" > $src/info/README.txt
echo "{\"title\":\"$slug\",\"id\":\"$acc/$slug\",\"licenses\":[{\"name\":\"CC0-1.0\"}]}" > $src/dataset-metadata.json
KAGGLE_API_TOKEN=$(cat ~/.kaggle/tok_$acc) kaggle datasets create -p $src -r zip -q 2>&1 | tail -2
