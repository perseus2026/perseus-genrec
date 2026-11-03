#!/usr/bin/env bash
set -euo pipefail
# MODEL_DIR [--revision HF_COMMIT]; prints the usable snapshot path.
test -x .venv/bin/python
export PYTHONPATH=src
exec .venv/bin/python scripts/amazon2014-grid/download-embedder.py "$@"
