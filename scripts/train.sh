#!/usr/bin/env bash
set -euo pipefail

workdir=${1:?Usage: scripts/train.sh WORKDIR CONFIG [NPROC]}
config=${2:?CONFIG is required}
nproc=${3:-1}
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export POLARS_MAX_THREADS=4
export OMP_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

if [ "$nproc" -eq 1 ]; then
  exec .venv/bin/python -m genrecsys train fit-model \
    --workdir="$workdir" --config="$config" --memory-map-samples
fi
exec .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node="$nproc" \
  --module genrecsys train fit-model --workdir="$workdir" --config="$config" --memory-map-samples
