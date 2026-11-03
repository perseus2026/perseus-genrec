"""CLI process-level behavior."""

import json
import logging

import polars as pl

from genrecsys import core
from genrecsys.cli import _logging_level, inference, train


def test_only_global_rank_zero_emits_application_logs(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    assert _logging_level() == logging.INFO
    monkeypatch.setenv("RANK", "0")
    assert _logging_level() == logging.INFO
    monkeypatch.setenv("RANK", "3")
    assert _logging_level() > logging.CRITICAL


def test_plan_reports_prepared_dataset_reuse(tmp_path, capsys):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config_path = "tests/fixtures/framework/sequence.yaml"
    config = core.Config.load(config_path)
    core.DataContract.from_config(config).save(checkpoint / "data_contract.json")

    train.plan(config=config_path, workdir=tmp_path)

    result = json.loads(capsys.readouterr().out)
    assert result["read_only"] is True
    assert result["reuses_prepared_dataset"] is True
    assert result["stages"] == [10, 12, 13]
    assert set(result["model_requirements"]) == {
        "artifact_columns",
        "checkpoint_files",
    }


def test_evaluate_needs_no_model_plugin_manifest_or_catalog(tmp_path):
    config = core.Config.load_data("tests/fixtures/framework/sequence.yaml")
    config.model.type = "not_installed"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config.save(checkpoint / "config.yaml")
    samples_dir = tmp_path / "dataset/test"
    samples_dir.mkdir(parents=True)
    basis = pl.DataFrame({"sample_id": [1], "target": [["a"]]})
    basis.write_parquet(samples_dir / "samples.pq")
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    basis.with_columns(
        pl.Series("prediction", [[{"item": "a", "score": 1.0}]])
    ).write_parquet(predictions / "part.pq")

    inference.evaluate(workdir=tmp_path)

    report = json.loads((tmp_path / "metrics.json").read_text())
    assert report["metrics"]["Recall@100"] == 1.0
    assert "catalog_size" not in report
