"""Predictions have a single external item field, with no legacy fallback."""

import math

import polars as pl
import pytest

from genrecsys.api.evaluate_predictions import evaluate_predictions
from genrecsys.core.config import Config, Metric


def _coverage_config() -> Config:
    config = Config.load_data("configs/tecd-small/onerec-retail-rq1.yaml")
    config.task.metrics["Coverage@100"] = Metric(
        type="coverage_at_k", params={"k": 100}
    )
    return config


@pytest.mark.parametrize("field", ["item", "item_id"])
def test_evaluator_requires_item_prediction_field(tmp_path, field):
    config = Config.load("tests/fixtures/framework/multi-history.yaml")
    samples = pl.DataFrame({"sample_id": ["a"], "target": [[20]]})
    predictions = samples.with_columns(
        pl.Series(
            "prediction", [[{field: 20, "score": -0.2}, {field: 10, "score": -0.3}]]
        )
    )
    path = tmp_path / "predictions.pq"
    predictions.write_parquet(path)
    if field == "item_id":
        with pytest.raises(
            ValueError, match=r"prediction must be List\[Struct\[item, score\]\]"
        ):
            evaluate_predictions(config, [path], expected_samples=samples)
        return
    report = evaluate_predictions(config, [path], expected_samples=samples)
    assert report["population_verified"] is True
    assert report["metrics"]["recall@20"] == 1.0


def test_evaluator_rejects_duplicate_recommendations(tmp_path):
    config = Config.load("tests/fixtures/framework/multi-history.yaml")
    samples = pl.DataFrame({"sample_id": ["a"], "target": [[20]]})
    predictions = samples.with_columns(
        pl.Series(
            "prediction", [[{"item": 20, "score": 1.0}, {"item": 20, "score": 0.5}]]
        )
    )
    path = tmp_path / "duplicate.pq"
    predictions.write_parquet(path)
    with pytest.raises(ValueError, match="duplicate items"):
        evaluate_predictions(config, [path], expected_samples=samples)


@pytest.mark.parametrize("payload", [False, True])
def test_evaluator_uses_only_canonical_targets(tmp_path, payload):
    config = Config.load("tests/fixtures/framework/sequence.yaml")
    basis = pl.DataFrame({"sample_id": [1, 2], "target": [["correct"], ["other"]]})
    basis_path = tmp_path / "basis.pq"
    basis.write_parquet(basis_path)
    predictions = pl.DataFrame(
        {
            "sample_id": [2, 1],
            "prediction": [[{"item": "forged", "score": 1.0}]] * 2,
        }
    )
    if payload:
        predictions = predictions.with_columns(
            target=pl.lit(["forged"]), target_labels=pl.lit([1.0])
        )
    path = tmp_path / "predictions.pq"
    predictions.write_parquet(path)
    result = evaluate_predictions(config, [path], expected_samples=basis_path)
    assert result["population_verified"]
    assert result["metrics"]["Recall@5"] == 0


def test_evaluator_rejects_duplicate_and_unknown_samples(tmp_path):
    config = Config.load("tests/fixtures/framework/sequence.yaml")
    basis = pl.DataFrame({"sample_id": [1, 2], "target": [["a"], ["b"]]})
    path = tmp_path / "predictions.pq"
    for identities, message in (([1, 1], "duplicate samples"), ([1, 3], "absent from")):
        pl.DataFrame(
            {"sample_id": identities, "prediction": [[{"item": "a", "score": 1.0}]] * 2}
        ).write_parquet(path)
        with pytest.raises(ValueError, match=message):
            evaluate_predictions(config, [path], expected_samples=basis)


def test_evaluator_rejects_duplicates_across_files_and_missing_samples(tmp_path):
    config = Config.load_data("tests/fixtures/framework/sequence.yaml")
    basis = pl.DataFrame({"sample_id": [1, 2], "target": [["a"], ["b"]]})
    first, second = tmp_path / "first.pq", tmp_path / "second.pq"
    frame = pl.DataFrame(
        {"sample_id": [1], "prediction": [[{"item": "a", "score": 1.0}]]}
    )
    frame.write_parquet(first)
    frame.write_parquet(second)
    with pytest.raises(ValueError, match="duplicate samples"):
        evaluate_predictions(config, [first, second], expected_samples=basis)
    with pytest.raises(ValueError, match="missing=1/2"):
        evaluate_predictions(config, [first], expected_samples=basis)


@pytest.mark.parametrize(
    "invalid", ["duplicate_basis", "null_basis", "null_prediction"]
)
def test_evaluator_rejects_invalid_identities(tmp_path, invalid):
    config = Config.load_data("tests/fixtures/framework/sequence.yaml")
    basis_ids = [1, 1] if invalid == "duplicate_basis" else [1, 2]
    prediction_ids = [1, 2]
    if invalid == "null_basis":
        basis_ids[1] = None
    if invalid == "null_prediction":
        prediction_ids[1] = None
    basis = pl.DataFrame({"sample_id": basis_ids, "target": [["a"], ["b"]]})
    path = tmp_path / "predictions.pq"
    pl.DataFrame(
        {"sample_id": prediction_ids, "prediction": [[{"item": "a", "score": 1.0}]] * 2}
    ).write_parquet(path)
    message = (
        "duplicate evaluation identities" if invalid == "duplicate_basis" else "null"
    )
    with pytest.raises(ValueError, match=message):
        evaluate_predictions(config, [path], expected_samples=basis)


def test_evaluator_preserves_weighted_metrics_across_reordered_partitions(tmp_path):
    config = Config.load_data("tests/fixtures/framework/multi-history.yaml")
    basis = pl.DataFrame(
        {
            "sample_id": [1, 2],
            "target": [[10, 20], []],
            "target_labels": [[3.0, 1.0], []],
        }
    )
    paths = [tmp_path / "second.pq", tmp_path / "first.pq"]
    for path, identity in zip(paths, (2, 1)):
        pl.DataFrame(
            {
                "sample_id": [identity],
                "target": [[999]],
                "target_labels": [[100.0]],
                "prediction": [[{"item": 10, "score": 1.0}]],
            }
        ).write_parquet(path)
    report = evaluate_predictions(config, paths, expected_samples=basis)
    assert report["samples"] == 2
    assert report["partitions"] == 2
    assert "catalog_size" not in report
    assert report["metrics"]["recall@20"] == 0.25
    assert report["metrics"]["ndcg@20"] == pytest.approx(3 / (3 + 1 / math.log2(3)) / 2)


def test_evaluator_computes_global_catalog_coverage(tmp_path):
    config = _coverage_config()
    basis = pl.DataFrame(
        {"sample_id": [1, 2], "target": [["a"], ["d"]], "target_labels": [[1.0], [1.0]]}
    )
    predictions = pl.DataFrame(
        {
            "sample_id": [1, 2],
            "prediction": [
                [{"item": "a", "score": 1.0}, {"item": "b", "score": 0.5}],
                [{"item": "b", "score": 1.0}, {"item": "c", "score": 0.5}],
            ],
        }
    )
    path = tmp_path / "predictions.pq"
    predictions.write_parquet(path)
    candidates = pl.DataFrame({"item_id": ["a", "b", "c", "d"]})

    report = evaluate_predictions(
        config,
        [path],
        expected_samples=basis,
        candidate_items=candidates,
    )

    assert report["metrics"] == {
        "Recall@100": 0.5,
        "NDCG@100": 0.5,
        "Coverage@100": 0.75,
    }


def test_coverage_requires_a_unique_authoritative_catalog(tmp_path):
    config = _coverage_config()
    basis = pl.DataFrame({"sample_id": [1], "target": [["a"]]})
    path = tmp_path / "predictions.pq"
    pl.DataFrame(
        {"sample_id": [1], "prediction": [[{"item": "a", "score": 1.0}]]}
    ).write_parquet(path)

    with pytest.raises(ValueError, match="authoritative candidate catalog"):
        evaluate_predictions(config, [path], expected_samples=basis)
    with pytest.raises(ValueError, match="unique non-null"):
        evaluate_predictions(
            config,
            [path],
            expected_samples=basis,
            candidate_items=pl.DataFrame({"item_id": ["a", "a"]}),
        )
