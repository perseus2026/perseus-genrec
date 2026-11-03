"""Global target-frequency retrieval baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import polars as pl

from genrecsys.core.config import Config
from genrecsys.core.dataset import Dataset
from genrecsys.core.evaluation import (
    coverage_at_k,
    metric_specs,
    relevance_by_item,
    retrieval_metric,
)
from genrecsys.core.tasks import retrieval


@dataclass
class GlobalCount:
    ranking: pl.DataFrame

    @classmethod
    def fit(cls, dataset: Dataset, /) -> Self:
        """Rank items by train target log frequency, breaking ties by item ID."""

        preprocessor = dataset.task_preprocessor
        if not isinstance(preprocessor, retrieval.Preprocessor):
            raise TypeError("global_count requires retrieval.Preprocessor")
        return cls(
            preprocessor.items.select("item_id", "logq").sort(
                "logq", "item_id", descending=[True, False]
            )
        )

    def predict(self, *, k: int) -> list[Any]:
        """Return the same top-k external item IDs for every user."""

        return self.ranking["item_id"].head(k).to_list()

    def save(self, path: Path, /) -> None:
        """Save the global ranking to ranking.pq."""

        path.mkdir(parents=True, exist_ok=True)
        self.ranking.write_parquet(path / "ranking.pq")

    @classmethod
    def load(cls, path: Path, /) -> Self:
        """Load the global ranking from ranking.pq."""

        return cls(pl.read_parquet(path / "ranking.pq"))


def fit_and_evaluate(
    config: Config,
    dataset: Dataset,
    /,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Fit on train targets and evaluate immutable validation/test splits."""

    model = GlobalCount.fit(dataset)
    model.save(output_dir)
    config.save(output_dir / "config.yaml")
    validation = _evaluate_split(
        config,
        model,
        dataset.validation_samples,
        candidate_items=dataset.validation_artifacts.items,
        name="validation",
        output_dir=output_dir,
    )
    report: dict[str, Any] = {
        "baseline": "global_count",
        "validation": validation,
    }
    if dataset.test_samples is not None:
        report["test"] = _evaluate_split(
            config,
            model,
            dataset.test_samples,
            candidate_items=dataset.test_artifacts.items,
            name="test",
            output_dir=output_dir,
        )
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def _evaluate_split(
    config: Config,
    model: GlobalCount,
    samples: pl.DataFrame,
    *,
    candidate_items: pl.DataFrame,
    name: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Evaluate a shared ranking against split targets without writing predictions."""

    if not len(samples):
        raise ValueError(f"cannot evaluate an empty {name} split")
    specs = metric_specs(config)
    ranking = model.predict(k=max(spec.k for spec in specs))
    sample_specs = tuple(spec for spec in specs if spec.metric_type != "coverage_at_k")
    coverage_specs = tuple(
        spec for spec in specs if spec.metric_type == "coverage_at_k"
    )
    totals = {spec.name: 0.0 for spec in sample_specs}
    columns = ["target", *(["target_labels"] if "target_labels" in samples else [])]
    for row in samples.select(columns).iter_rows(named=True):
        relevance = relevance_by_item(row["target"], row.get("target_labels"))
        for spec in sample_specs:
            totals[spec.name] += retrieval_metric(spec, ranking, relevance)
    metrics = {key: value / len(samples) for key, value in totals.items()}
    metrics.update(
        {
            spec.name: coverage_at_k(
                [ranking], spec.k, candidate_items["item_id"].n_unique()
            )
            for spec in coverage_specs
        }
    )
    report = {
        "samples": len(samples),
        "metrics": metrics,
    }
    (output_dir / f"{name}_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report
