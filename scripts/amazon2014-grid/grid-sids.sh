#!/usr/bin/env bash
set -euo pipefail
# ES1|ES2 RAW DATA NEW_SID_DIR GRID_CHECKOUT GRID_PYTHON CATEGORY METHOD SID_SEED
export PYTHONPATH=src OMP_NUM_THREADS=1 POLARS_MAX_THREADS=4 NVIDIA_TF32_OVERRIDE=0
exec .venv/bin/python scripts/amazon2014-grid/grid-quantizers.py fit "$@"
