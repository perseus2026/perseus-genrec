#!/usr/bin/env bash
set -euo pipefail
# Restore GRID and prepare its separate environment; never replace the framework .venv.
grid=${1:?GRID_CHECKOUT}
grid_python=${2:?GRID_PYTHON, e.g. /absolute/path/grid-venv/bin/python}
test -x .venv/bin/python
export PYTHONPATH=src
.venv/bin/python - "$grid" "$grid_python" <<'PY_GRID'
import runpy
import subprocess
import sys
from pathlib import Path

from filelock import FileLock

grid = Path(sys.argv[1]).absolute()
python = Path(sys.argv[2]).absolute()  # Keep venv interpreter symlinks intact.
env = python.parent.parent
if python.parent.name != "bin" or python.name != "python":
    raise ValueError("GRID_PYTHON must end with /bin/python")
if env.resolve() == Path(".venv").resolve():
    raise ValueError("GRID requires its own environment, separate from .venv")
grid.parent.mkdir(parents=True, exist_ok=True)
env.parent.mkdir(parents=True, exist_ok=True)
with FileLock(str(grid) + ".lock"):
    if not grid.exists():
        subprocess.run([sys.executable, "scripts/amazon2014-grid/restore-grid.py", str(grid)], check=True)
    restore = runpy.run_path("scripts/amazon2014-grid/restore-grid.py")
    expected = restore["COMMIT"]
    marker = grid / ".grid-commit"
    if not marker.is_file() or marker.read_text().strip() != expected:
        raise ValueError("GRID revision differs; use a new GRID_DIR for the pinned source")
    if not (grid / "configs/experiment/reproduce_embeddings.yaml").is_file():
        raise ValueError("GRID checkout is incomplete; restore into a new directory")
    restore["patch_tiger_prefix_mask"](grid)
    restore["install_overlays"](grid)
with FileLock(str(env) + ".lock"):
    pending = env / ".amazon-grid-setup-pending"
    if not env.exists():
        env.mkdir()
        pending.touch()
    if pending.exists():
        subprocess.run([sys.executable, "-m", "venv", str(env)], check=True)
        subprocess.run([str(python), "-m", "pip", "install",
                        "--index-url=https://pypi.org/simple",
                        "--extra-index-url=https://download.pytorch.org/whl/cu124",
                        "-r", str(grid / "requirements.txt")], check=True)
        subprocess.run([str(python), "-m", "pip", "check"], check=True)
        pending.unlink()
    if not python.is_file():
        raise ValueError("Existing GRID environment has no bin/python; use a new directory")
PY_GRID
