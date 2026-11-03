#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root with the existing framework .venv.
# One-category quickstart; the six-path form remains available for custom locations.
if [ "$#" -eq 1 ]; then
  set -- "$PWD/work/amazon_data/$1" "$PWD/work/es1-complete/$1" "$1" \
    "$PWD/work/grid/GRID-main" "$PWD/work/grid/venv/bin/python" \
    "$PWD/work/es1/flan-t5-xl"
fi
if [ "$#" -ne 6 ]; then
  echo "Usage: $0 CATEGORY (beauty|toys|sports)" >&2
  exit 2
fi
raw=${1:?RAW_CATEGORY_DIR}
work=${2:?NEW_EXPERIMENT_CATEGORY_DIR}
category=${3:?CATEGORY: beauty, toys or sports}
grid=${4:?GRID_CHECKOUT}
grid_python=${5:?GRID_PYTHON}
model_path=${6:?FLAN_T5_XL_DIR: existing snapshot or new download destination}
case "$category" in beauty|toys|sports) ;; *) echo "Unknown category: $category" >&2; exit 2 ;; esac
for path in "$raw" "$work" "$grid" "$grid_python" "$model_path"; do
  case "$path" in /*) ;; *) echo 'Use absolute paths' >&2; exit 2 ;; esac
done
test -x .venv/bin/python
test ! -e "$work"
if [ "$(basename "$raw")" != "$category" ]; then
  echo 'RAW_CATEGORY_DIR must end with the selected category' >&2
  exit 2
fi
bash scripts/amazon2014-grid/download-data.sh "$(dirname "$raw")"
model_path=$(bash scripts/amazon2014-grid/download-embedder.sh "$model_path")
bash scripts/amazon2014-grid/prepare-grid.sh "$grid" "$grid_python"
mkdir -p "$(dirname "$work")"
mkdir "$work"
bash scripts/amazon2014-grid/prepare-data.sh \
  "$raw" "$work/data" "$category" "$grid" "$grid_python" "$model_path"

bash scripts/amazon2014-grid/grid-sids.sh es1 \
  "$raw" "$work/data" "$work/data/grid-sids" "$grid" "$grid_python" "$category" rkmeans 42
bash scripts/amazon2014-grid/prepare.sh \
  "$work/data" "$work/prepared" "configs/amazon2014-grid/$category.yaml"
for model in onerec grid; do
  config="configs/amazon2014-grid/$category.yaml"
  if [ "$model" = grid ]; then config="configs/amazon2014-grid/grid-$category.yaml"; fi
  for seed in 42 43 44 45 46; do
    bash scripts/amazon2014-grid/run.sh \
      "$work/prepared" "$work/models/$model/seed$seed" "$category" "$seed" "$config"
  done
done

for seed in 42 43 44 45 46; do
  bash scripts/amazon2014-grid/es1/run-tiger.sh \
    "$raw" "$work/data/grid-sids/inference/pickle/merged_predictions_tensor.pt" \
    "$work/models/tiger-author/seed$seed" "$category" "$seed" "$grid" "$grid_python"
done
