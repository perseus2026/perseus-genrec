#!/usr/bin/env bash
set -euo pipefail
# One shared preparation per category, from original author TFRecords.
raw=${1:?RAW_CATEGORY_DIR}
data=${2:?NEW_CATEGORY_DATA_DIR}
category=${3:?CATEGORY}
grid=${4:?GRID_CHECKOUT}
grid_python=${5:?GRID_PYTHON}
model=${6:?LOCAL_FLAN_T5_XL_SNAPSHOT}
case "$category" in beauty|toys|sports) ;; *) exit 2 ;; esac
for value in "$raw" "$data" "$grid" "$grid_python" "$model"; do
  case "$value" in /*) ;; *) echo 'Use absolute paths' >&2; exit 2 ;; esac
done
test ! -e "$data"
export PYTHONPATH=src OMP_NUM_THREADS=1 POLARS_MAX_THREADS=4 NVIDIA_TF32_OVERRIDE=0
.venv/bin/python -m model_datasets.amazon2014_grid import-amazon \
  --category="$category" --source-dir="$raw" --output-dir="$data/prepared"
bash scripts/amazon2014-grid/grid-embeddings.sh "$raw" "$data" "$grid" "$grid_python" "$model"
.venv/bin/python -m model_datasets.amazon2014_grid build-basis \
  --category="$category" --prepared-source-dir="$data/prepared" \
  --stage01-dir="$data/stage01" --output-dir="$data/basis" \
  --augmentation=none --max-history=20
# build-basis also exports checked, unnormalized, full-catalog embeddings to
# DATA/basis/representations. Both native and GRID methods use this same source.
