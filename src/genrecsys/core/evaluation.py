"""Shared retrieval metric declarations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from genrecsys.core.config import Config


@dataclass(frozen=True)
class MetricSpec:
    name: str
    metric_type: str
    k: int


def metric_specs(config: Config) -> tuple[MetricSpec, ...]:
    """Return the metric contract consumed by model evaluation."""

    return tuple(
        MetricSpec(
            name=name,
            metric_type=metric.type,
            k=int(metric.params["k"]),
        )
        for name, metric in config.task.metrics.items()
        if metric is not None and metric.type is not None
    )


def relevance_by_item(
    targets: list[Any] | None,
    labels: list[float] | None = None,
) -> dict[Any, float]:
    """Build the canonical item-to-gain view of one prepared target row."""

    items = targets or []
    if items and isinstance(items[0], dict):
        if labels is not None:
            raise ValueError(
                "structured target already contains relevance; omit target_labels"
            )
        labels = [value["relevance"] for value in items]
        items = [value["item"] for value in items]
    values = labels if labels is not None else [1.0] * len(items)
    if len(values) != len(items):
        raise ValueError(
            f"target/target_labels length mismatch: {len(items)} != {len(values)}"
        )
    relevance: dict[Any, float] = {}
    for item, gain in zip(items, values):
        relevance[item] = max(relevance.get(item, 0.0), float(gain))
    return relevance


def retrieval_metric(
    spec: MetricSpec,
    ranking: list[Any],
    relevance: dict[Any, float],
) -> float:
    """Evaluate one prepared retrieval row for one configured metric."""

    if len(ranking) != len(set(ranking)):
        raise ValueError("prediction ranking contains duplicate items")
    hits = [
        (rank, relevance[item])
        for rank, item in enumerate(ranking[: spec.k])
        if item in relevance
    ]
    if spec.metric_type == "hit_rate_at_k":
        return float(bool(hits))
    if spec.metric_type == "recall_at_k":
        return len(hits) / len(relevance) if relevance else 0.0
    if spec.metric_type == "precision_at_k":
        size = len(ranking[: spec.k])
        return len(hits) / size if size else 0.0
    if spec.metric_type == "map_at_k":
        denominator = min(len(relevance), spec.k)
        if not denominator:
            return 0.0
        return (
            sum(
                hit_number / (rank + 1)
                for hit_number, (rank, _) in enumerate(hits, start=1)
            )
            / denominator
        )
    if spec.metric_type == "ndcg_at_k":
        dcg = sum(gain / math.log2(rank + 2) for rank, gain in hits)
        ideal = sorted(relevance.values(), reverse=True)[: spec.k]
        idcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(ideal))
        return dcg / idcg if idcg else 0.0
    if spec.metric_type == "coverage_at_k":
        raise ValueError("coverage_at_k is a global metric, not a per-sample metric")
    raise ValueError(f"unsupported retrieval metric {spec.metric_type!r}")


def coverage_at_k(rankings: list[list[Any]], k: int, n_items: int) -> float:
    """Return public-Perseus catalog coverage over complete prediction rankings."""

    if n_items <= 0:
        return 0.0
    return len({item for ranking in rankings for item in ranking[:k]}) / n_items


__all__ = [
    "MetricSpec",
    "coverage_at_k",
    "metric_specs",
    "relevance_by_item",
    "retrieval_metric",
]
