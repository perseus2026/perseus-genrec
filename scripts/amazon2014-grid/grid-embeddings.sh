#!/usr/bin/env bash
set -euo pipefail
# Run from the genrecsys repository root. All arguments must be absolute paths.
raw=${1:?RAW_CATEGORY_DIR}
data=${2:?NEW_CATEGORY_DATA_DIR containing prepared/}
grid=${3:?GRID_CHECKOUT}
grid_python=${4:?GRID_PYTHON}
model=${5:?LOCAL_FLAN_T5_XL_SNAPSHOT}
repo="$PWD"
for value in "$raw" "$data" "$grid" "$grid_python" "$model"; do
  case "$value" in /*) ;; *) echo 'Use absolute paths' >&2; exit 2 ;; esac
done
test -f "$grid/configs/experiment/reproduce_embeddings.yaml"
test -d "$model"
test ! -e "$data/stage01"
mkdir -p "$data/stage01/runtime"
export OMP_NUM_THREADS=1 POLARS_MAX_THREADS=4 NVIDIA_TF32_OVERRIDE=0
PYTHONPATH="$repo/src" "$repo/.venv/bin/python" -m model_datasets.amazon2014_grid check-source-inputs \
  --prepared-source-dir="$data/prepared" --source-dir="$raw" --scope=items \
  --output-path="$data/stage01/runtime/source-inputs.json"
cp "$grid/requirements.txt" "$data/stage01/runtime/grid-requirements.txt"
"$grid_python" -m pip freeze > "$data/stage01/runtime/grid-packages.txt"
cd "$grid"
PYTHONPATH="$grid" "$grid_python" -m src.inference \
  experiment=reproduce_embeddings data_dir="$raw" embedding_model="$model" \
  hydra.run.dir="$data/stage01/grid"
