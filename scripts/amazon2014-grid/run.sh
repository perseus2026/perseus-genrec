#!/usr/bin/env bash
set -euo pipefail

prepared=${1:?Usage: scripts/amazon2014-grid/run.sh PREPARED_DIR WORKDIR CATEGORY SEED [CONFIG]}
workdir=${2:?WORKDIR is required}
category=${3:?CATEGORY is required}
seed=${4:?SEED is required}
case "$category" in beauty|toys|sports) ;; *) echo "Unknown category: $category" >&2; exit 2 ;; esac
config=${5:-configs/amazon2014-grid/$category.yaml}
export PYTHONPATH=src
export POLARS_MAX_THREADS=4
export OMP_NUM_THREADS=1

prepared=$(cd "$prepared" && pwd)
test ! -e "$prepared/checkpoint/model"
mkdir -p "$(dirname "$workdir")"
mkdir "$workdir"
# Each run owns its checkpoint; all compatible models/seeds share the dataset.
cp -R "$prepared/checkpoint" "$workdir/checkpoint"
ln -s "$prepared/dataset" "$workdir/dataset"
.venv/bin/python - "$config" "$workdir/config.yaml" "$seed" <<'PY_CONFIG'
import sys
from pathlib import Path
from genrecsys.core import Config

seed = int(sys.argv[3])
if not 0 <= seed < 2**32:
    raise ValueError("SEED must be in [0, 2**32)")
recipe = Config.load_data(sys.argv[1])
recipe.training.dataloader.seed = seed
recipe.validate()
recipe.save(Path(sys.argv[2]))
PY_CONFIG
# Reproduction uses one GPU: changing process count changes the optimizer batch.
scripts/train.sh "$workdir" "$workdir/config.yaml" 1
scripts/predict.sh "$workdir" 1
scripts/evaluate.sh "$workdir"
