#!/usr/bin/env bash
set -euo pipefail

workdir=${1:?Usage: scripts/predict.sh WORKDIR [NPROC] [CONFIG]}
nproc=${2:-4}
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export POLARS_MAX_THREADS=4
export OMP_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ $# -ge 3 ]]; then
  .venv/bin/python -c '
import sys
from pathlib import Path
from genrecsys.core import Config, DataContract
checkpoint = Path(sys.argv[1]) / "checkpoint"
config = Config.load(sys.argv[2])
DataContract.load(checkpoint / "data_contract.json").require_compatible(config)
config.save(checkpoint / "config.yaml")
' "$workdir" "$3"
fi

POLARS_MAX_THREADS=8 .venv/bin/python -m genrecsys inference distribute-samples \
  --workdir="$workdir" --history-source=event_store
if [ "$nproc" -eq 1 ]; then
  exec .venv/bin/python -m genrecsys inference make-predictions \
    --workdir="$workdir" --history-source=event_store \
    --input-batch-size=1024 --prediction-write-batch-size=1024
fi
exec .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node="$nproc" \
  --module genrecsys inference make-predictions \
  --workdir="$workdir" --history-source=event_store \
  --input-batch-size=1024 --prediction-write-batch-size=1024
