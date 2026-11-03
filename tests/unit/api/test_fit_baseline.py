"""Global-count evaluation does not require a model plugin or prediction files."""

import json
from types import SimpleNamespace

import polars as pl
import pytest

from genrecsys.api.evaluate_predictions import evaluate_predictions
from genrecsys.api.fit_baseline import fit_baseline
from genrecsys.core.config import Config
from genrecsys.core.tasks.retrieval import Preprocessor


def test_baseline_matches_prediction_metrics_without_model_validation(tmp_path):
    config = Config.load_data("tests/fixtures/framework/multi-history.yaml")
    config.model.type = "not_installed"
    samples = pl.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "target": [[10, 20], [20], []],
            "target_labels": [[3.0, 1.0], [2.0], []],
        }
    )
    dataset = SimpleNamespace(
        task_preprocessor=Preprocessor(
            pl.DataFrame({"item_id": [10, 20], "logq": [-0.1, -0.5]})
        ),
        validation_samples=samples,
        test_samples=samples,
        validation_artifacts=SimpleNamespace(items=pl.DataFrame({"item_id": [10, 20]})),
        test_artifacts=SimpleNamespace(items=pl.DataFrame({"item_id": [10, 20]})),
    )
    output = tmp_path / "baseline"
    report = fit_baseline(config, dataset, output_dir=output)
    assert not (output / "predictions").exists()
    assert (output / "ranking.pq").is_file()
    assert json.loads((output / "metrics.json").read_text()) == report
    path = tmp_path / "reference.pq"
    samples.with_columns(
        pl.Series(
            "prediction",
            [[{"item": 10, "score": -0.1}, {"item": 20, "score": -0.5}]] * 3,
        )
    ).write_parquet(path)
    expected = evaluate_predictions(config, [path], expected_samples=samples)
    for split in ("validation", "test"):
        assert report[split]["samples"] == len(samples)
        assert report[split]["metrics"] == pytest.approx(expected["metrics"])
