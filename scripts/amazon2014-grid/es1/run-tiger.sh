#!/usr/bin/env bash
set -euo pipefail
# Author TIGER train/test on the same raw splits and SID mapping as native models.
raw=${1:?RAW_CATEGORY_DIR}
sids=${2:?SID_TENSOR}
work=${3:?NEW_RUN_DIR}
category=${4:?CATEGORY: beauty, toys or sports}
seed=${5:?MODEL_SEED}
grid=${6:?GRID_CHECKOUT}
grid_python=${7:?GRID_PYTHON}
repo="$PWD"
case "$category" in beauty|toys|sports) ;; *) echo "Unknown category: $category" >&2; exit 2 ;; esac
for path in "$raw" "$sids" "$work" "$grid" "$grid_python"; do
  case "$path" in /*) ;; *) echo 'Use absolute paths' >&2; exit 2 ;; esac
done
test -s "$sids"
test -f "$grid/configs/experiment/reproduce_tiger_train.yaml"
mkdir -p "$(dirname "$work")"
mkdir "$work"
export OMP_NUM_THREADS=1 POLARS_MAX_THREADS=4 NVIDIA_TF32_OVERRIDE=0
cd "$grid"
PYTHONPATH="$grid" "$grid_python" -m src.train \
  experiment=reproduce_tiger_train category="$category" \
  data_dir="$raw" semantic_id_path="$sids" seed="$seed" \
  hydra.run.dir="$work/train"

# GRID evaluates the best validation checkpoint and writes test metrics to CSV.
"$repo/.venv/bin/python" - "$work" <<'PY_METRICS'
import csv
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
names = {
    "test/recall@5": "Recall@5",
    "test/recall@10": "Recall@10",
    "test/ndcg@5": "NDCG@5",
    "test/ndcg@10": "NDCG@10",
}
metrics = {}
for path in sorted((root / "train/csv").rglob("metrics.csv")):
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            for field, name in names.items():
                if row.get(field):
                    metrics[name] = float(row[field])
best = list((root / "train/checkpoints").glob("best_*.ckpt"))
if len(best) != 1 or set(metrics) != set(names.values()):
    raise ValueError("GRID must produce one best checkpoint and all four test metrics")
if any(not math.isfinite(value) or not 0 <= value <= 1 for value in metrics.values()):
    raise ValueError("Invalid GRID test metrics")
with (root / "metrics.json").open("x") as stream:
    json.dump({"metrics": metrics, "best_checkpoint": str(best[0].relative_to(root))}, stream, indent=2)
    stream.write("\n")
PY_METRICS
