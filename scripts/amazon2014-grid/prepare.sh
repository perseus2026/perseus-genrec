#!/usr/bin/env bash
set -euo pipefail

data_dir=${1:?Usage: scripts/amazon2014-grid/prepare.sh DATA_DIR WORKDIR CONFIG [SID_MAPPING [SID_IMPORT_REPORT]]}
workdir=${2:?WORKDIR is required}
config=${3:?CONFIG is required}
mapping=${4:-$data_dir/sids/item_to_sid.pq}
report=${5-$data_dir/sids/sid_import_report.json}
export PYTHONPATH=src
export POLARS_MAX_THREADS=4
export OMP_NUM_THREADS=1

test -d "$data_dir/basis/eventhub"
test -s "$mapping"

if [ "$#" -lt 3 ] || [ "$#" -gt 5 ]; then
  echo "Preparation is shared; pass model seed to run.sh, not prepare.sh." >&2
  exit 2
fi

# DATA_DIR/basis is the newly built NoAug/h20 bundle.
# The default mapping is ES1; ES2 supplies its selected mapping explicitly.
set -- --basis="$data_dir/basis" --sid-mapping="$mapping" \
  --workdir="$workdir" --config="$config" --user-bins=2000
if [ -n "$report" ]; then set -- "$@" --sid-import-report="$report"; fi
.venv/bin/python -m model_datasets.amazon2014_grid materialize-basis "$@"
INTERNAL_STORAGE_AMAZON2014GRID="$data_dir/basis/eventhub" \
  .venv/bin/python -m genrecsys train prepare-dataset --workdir="$workdir"

# Warm the existing atomic IPC cache once before independent training jobs.
.venv/bin/python - "$workdir" <<'PY_READY'
import sys
from pathlib import Path
from genrecsys.core.sample_table import MappedSamples

root = Path(sys.argv[1])
for split in ("train", "validation", "test"):
    MappedSamples.from_parquet(root / "dataset" / split / "samples.pq")
PY_READY
