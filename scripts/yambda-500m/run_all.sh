#!/usr/bin/env bash
set -euo pipefail

data_dir=${1:-work/yambda-500m/data}
run_dir=${2:-work/yambda-500m/run}
export PYTHONPATH=src:.
exec .venv/bin/python -m scripts.experiments run-all yambda-500m \
  --data-dir="$data_dir" --run-dir="$run_dir"
