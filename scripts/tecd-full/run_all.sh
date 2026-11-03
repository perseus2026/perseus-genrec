#!/usr/bin/env bash
set -euo pipefail

data_dir=${1:-work/tecd-full/data}
run_dir=${2:-work/tecd-full/run}
export PYTHONPATH=src:.
exec .venv/bin/python -m scripts.experiments run-all tecd-full \
  --data-dir="$data_dir" --run-dir="$run_dir"
