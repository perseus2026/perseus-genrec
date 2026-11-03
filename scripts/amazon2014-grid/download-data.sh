#!/usr/bin/env bash
set -euo pipefail
# RAW_ROOT [--archive LOCAL_ZIP]; all three categories share one download.
test -x .venv/bin/python
export PYTHONPATH=src
exec .venv/bin/python scripts/amazon2014-grid/download-data.py "$@"
