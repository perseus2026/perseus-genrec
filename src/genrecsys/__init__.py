"""Public API for the GenRecSys recommendation lifecycle."""

from importlib import import_module
from typing import Any

from genrecsys import core
from genrecsys.core import Config, Dataset, checkpoint

__all__ = [
    "Config",
    "Dataset",
    "api",
    "checkpoint",
    "core",
    "fit_baseline",
    "fit_model",
    "make_predictions",
    "prepare_dataset",
]


def __getattr__(name: str) -> Any:
    """Load API modules lazily so external dataset packages can use core."""

    if name == "api":
        return import_module("genrecsys.api")
    if name in {"fit_baseline", "fit_model", "make_predictions", "prepare_dataset"}:
        return getattr(import_module("genrecsys.api"), name)
    raise AttributeError(name)
