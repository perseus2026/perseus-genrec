"""Retrieval metrics shared by validation and standalone evaluation."""

import pytest

from genrecsys.core.evaluation import MetricSpec, retrieval_metric


def test_precision_at_k_matches_tecd_definition() -> None:
    spec = MetricSpec(name="Precision@3", metric_type="precision_at_k", k=3)

    value = retrieval_metric(spec, ["a", "x", "b"], {"a": 1.0, "b": 1.0})

    assert value == pytest.approx(2 / 3)


def test_map_at_k_matches_tecd_definition() -> None:
    spec = MetricSpec(name="MAP@3", metric_type="map_at_k", k=3)

    value = retrieval_metric(spec, ["a", "x", "b"], {"a": 1.0, "b": 1.0})

    assert value == pytest.approx((1.0 + 2 / 3) / 2)


@pytest.mark.parametrize("metric_type", ["precision_at_k", "map_at_k"])
def test_new_metrics_are_zero_without_targets(metric_type: str) -> None:
    spec = MetricSpec(name=metric_type, metric_type=metric_type, k=3)

    assert retrieval_metric(spec, ["a", "b"], {}) == 0.0
