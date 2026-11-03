"""Evaluate predictions against an authoritative test basis using Polars."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import polars as pl

from genrecsys.core.config import Config
from genrecsys.core.evaluation import (
    metric_specs,
    relevance_by_item,
    retrieval_metric,
)

logger = logging.getLogger(__name__)


def evaluate_predictions(
    config: Config,
    paths: Iterable[str | Path],
    /,
    *,
    expected_samples: pl.DataFrame | Path,
    candidate_items: pl.DataFrame | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate retrieval predictions against the complete expected sample set.

    Read targets and labels only from expected_samples, ignoring prediction payload
    labels. Return a JSON-compatible report and optionally write it to output_path.
    """
    schema = (
        expected_samples.schema
        if isinstance(expected_samples, pl.DataFrame)
        else pl.read_parquet_schema(expected_samples)
    )
    identity = ("sample_id",) if "sample_id" in schema else ("client_id", "timestamp")
    if missing := {*identity, "target"} - set(schema):
        raise ValueError(f"evaluation basis misses columns: {sorted(missing)}")
    truth_columns = [
        *identity,
        "target",
        *(["target_labels"] if "target_labels" in schema else []),
    ]
    truth = (
        expected_samples.select(truth_columns)
        if isinstance(expected_samples, pl.DataFrame)
        else pl.read_parquet(expected_samples, columns=truth_columns)
    )
    if truth.select(pl.any_horizontal(pl.col(identity).is_null()).any()).item():
        raise ValueError("evaluation identity contains null values")
    if truth.select(identity).is_duplicated().any():
        raise ValueError("inference basis has duplicate evaluation identities")

    paths = sorted(Path(value) for value in paths)
    if not paths:
        raise ValueError("prediction directory contains no samples")
    specs = metric_specs(config)
    sample_specs = tuple(spec for spec in specs if spec.metric_type != "coverage_at_k")
    coverage_specs = tuple(
        spec for spec in specs if spec.metric_type == "coverage_at_k"
    )
    if coverage_specs and candidate_items is None:
        raise ValueError("coverage_at_k requires the authoritative candidate catalog")
    totals = {spec.name: 0.0 for spec in sample_specs}
    covered_items = {spec.k: set() for spec in coverage_specs}
    prediction_identities = []
    unknown_samples = 0
    evaluated_samples = 0
    for index, path in enumerate(paths, start=1):
        prediction_schema = pl.read_parquet_schema(path)
        if missing := {*identity, "prediction"} - set(prediction_schema):
            raise ValueError(
                f"prediction partition {path} misses columns: {sorted(missing)}"
            )
        dtype = prediction_schema["prediction"]
        if (
            not isinstance(dtype, pl.List)
            or not isinstance(dtype.inner, pl.Struct)
            or {field.name for field in dtype.inner.fields} != {"item", "score"}
        ):
            raise ValueError(
                f"{path}: prediction must be List[Struct[item, score]], got {dtype}"
            )
        predictions = pl.read_parquet(path, columns=[*identity, "prediction"])
        if predictions.select(
            pl.any_horizontal(pl.col(identity).is_null()).any()
        ).item():
            raise ValueError("evaluation identity contains null values")
        if predictions.select(identity).is_duplicated().any():
            raise ValueError("prediction partitions contain duplicate samples")
        prediction_identities.append(predictions.select(identity))
        joined = predictions.join(
            truth, on=list(identity), how="inner", maintain_order="left"
        )
        unknown_samples += len(predictions) - len(joined)
        evaluated_samples += len(joined)
        for row in joined.iter_rows(named=True):
            relevance = relevance_by_item(row["target"], row.get("target_labels"))
            ranking = [value["item"] for value in (row["prediction"] or [])]
            for spec in sample_specs:
                totals[spec.name] += retrieval_metric(spec, ranking, relevance)
            for k, items in covered_items.items():
                items.update(ranking[:k])
        if index % 10 == 0 or index == len(paths):
            logger.info(
                "evaluated prediction partitions %d/%d: samples=%d",
                index,
                len(paths),
                evaluated_samples,
            )
    identities = pl.concat(prediction_identities)
    if identities.select(identity).is_duplicated().any():
        raise ValueError("prediction partitions contain duplicate samples")
    if unknown_samples:
        raise ValueError("predictions contain samples absent from the inference basis")
    if evaluated_samples != len(truth):
        raise ValueError(
            "predictions do not cover the full inference basis: "
            f"missing={len(truth) - evaluated_samples}/{len(truth)}"
        )
    if not evaluated_samples:
        raise ValueError("prediction directory contains no samples")
    metrics = {name: value / evaluated_samples for name, value in totals.items()}
    if coverage_specs:
        catalog = (
            candidate_items
            if isinstance(candidate_items, pl.DataFrame)
            else pl.read_parquet(candidate_items, columns=["item_id"])
        )
        if catalog["item_id"].null_count() or catalog["item_id"].n_unique() != len(
            catalog
        ):
            raise ValueError(
                "coverage candidate catalog must contain unique non-null item_id"
            )
        n_items = catalog["item_id"].n_unique()
        metrics.update(
            {
                spec.name: len(covered_items[spec.k]) / n_items if n_items else 0.0
                for spec in coverage_specs
            }
        )
    report = {
        "samples": evaluated_samples,
        "partitions": len(paths),
        "population_verified": True,
        "identity_columns": list(identity),
        "metrics": metrics,
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return report


__all__ = ["evaluate_predictions"]
