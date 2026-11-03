#!/usr/bin/env bash
set -euo pipefail

workdir=${1:?Usage: scripts/evaluate.sh WORKDIR}
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export POLARS_MAX_THREADS=4
export OMP_NUM_THREADS=1
exec .venv/bin/python -m genrecsys inference evaluate --workdir="$workdir"
