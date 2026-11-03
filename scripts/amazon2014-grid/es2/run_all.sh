#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root with the existing framework .venv.
# One-category quickstart; the six-path form remains available for custom locations.
if [ "$#" -eq 1 ]; then
  set -- "$PWD/work/amazon_data/$1" "$PWD/work/es2-complete/$1" "$1" \
    "$PWD/work/grid/GRID-main" "$PWD/work/grid/venv/bin/python" \
    "$PWD/work/es2/flan-t5-xl"
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

for builder in grid-rkmeans grid-rvq grid-rqvae ours-rq_kmeans_full ours-rq_vae ours-rvq; do
  for seed in 42 43 44 45 46; do
    fit="$work/fits/$builder/sid$seed"
    prepared="$work/prepared/$builder/sid$seed"
    if [[ "$builder" == grid-* ]]; then
      bash scripts/amazon2014-grid/grid-sids.sh es2 \
        "$raw" "$work/data" "$fit" "$grid" "$grid_python" "$category" "${builder#grid-}" "$seed"
    else
      bash scripts/amazon2014-grid/es2/sids.sh \
        "$work/data/basis/representations" "$fit" "$category" "${builder#ours-}" "$seed"
    fi
    if [[ "$builder" == grid-* ]]; then
      mapping="$fit/imported/item_to_sid.pq"
      report="$fit/imported/sid_import_report.json"
    else
      mapping="$fit/inference/item_to_sid.parquet"
      report=""
    fi
    bash scripts/amazon2014-grid/prepare.sh "$work/data" "$prepared" \
      "configs/amazon2014-grid/$category.yaml" "$mapping" "$report"
    for model in onerec grid; do
      config="configs/amazon2014-grid/$category.yaml"
      if [ "$model" = grid ]; then config="configs/amazon2014-grid/grid-$category.yaml"; fi
      bash scripts/amazon2014-grid/run.sh \
        "$prepared" "$work/models/$model/$builder/sid$seed-model42" "$category" 42 "$config"
    done
  done
done
