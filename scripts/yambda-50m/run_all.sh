#!/usr/bin/env bash
set -euo pipefail

data_dir=${1:-work/yambda-50m/data}
run_dir=${2:-work/yambda-50m/run}
export PYTHONPATH=src:.
exec .venv/bin/python -m scripts.experiments run-all yambda-50m \
  --data-dir="$data_dir" --run-dir="$run_dir"
