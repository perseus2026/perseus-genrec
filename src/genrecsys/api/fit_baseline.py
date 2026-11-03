"""Configured baseline lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from genrecsys.core.baselines.global_count import fit_and_evaluate
from genrecsys.core.config import Config
from genrecsys.core.dataset import Dataset


def fit_baseline(
    config: Config,
    dataset: Dataset,
    /,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Fit global_count, evaluate holdout splits, and save baseline artifacts.

    Return the metrics report without changing the trained model checkpoint.
    """

    config.validate_data()
    baseline = config.task.baseline
    if baseline is None:
        raise ValueError("task.baseline is not configured")
    if baseline.type != "global_count":
        raise ValueError(f"unsupported baseline {baseline.type!r}")
    return fit_and_evaluate(config, dataset, output_dir=output_dir)


__all__ = ["fit_baseline"]
