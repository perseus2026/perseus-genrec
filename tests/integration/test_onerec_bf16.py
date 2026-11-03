"""Exercise configured BF16 in real fitting, validation, save/load and inference."""

import math

import polars as pl
import pytest
import torch

from genrecsys import api, core
from genrecsys.core.models.onerec import inference
from tests.integration.test_lifecycle import (
    _synthetic_basis,
    _synthetic_config,
    _write_synthetic_eventhub,
)


@pytest.mark.parametrize("precision", ["no", "bf16"])
def test_configured_precision_covers_validation_and_inference(
    tmp_path, monkeypatch, fp32_runtime, precision
):
    torch.set_num_threads(1)
    events = tmp_path / "events"
    _write_synthetic_eventhub(events)
    monkeypatch.setenv("INTERNAL_STORAGE_EVENT_HUB", str(events))
    config = _synthetic_config()
    config.training.mixed_precision = config.inference.mixed_precision = precision
    train, validation, train_artifacts, validation_artifacts, tables, mappings = (
        _synthetic_basis()
    )
    prepared, dataset = api.prepare_dataset(
        config,
        train,
        validation,
        train_artifacts=train_artifacts,
        validation_artifacts=validation_artifacts,
        resources=core.Resources(tables),
        event_feature_mappings=mappings,
    )
    original = inference._predict_batch
    calls = []

    def predict(*args, **kwargs):
        calls.append(torch.is_autocast_enabled("cpu"))
        return original(*args, **kwargs)

    monkeypatch.setattr(inference, "_predict_batch", predict)
    trained = api.fit_model(prepared, dataset)
    assert calls and set(calls) == {precision == "bf16"}
    assert all(
        torch.isfinite(value).all()
        for value in trained.model.state_dict.values()
        if value.is_floating_point()
    )
    assert all(
        value.dtype == torch.float32
        for value in trained.model.state_dict.values()
        if value.is_floating_point()
    )
    assert math.isfinite(trained.model.training["history"][0]["train_loss"])
    trained.save(tmp_path / "checkpoint")
    restored = core.checkpoint.Trained.load(tmp_path / "checkpoint")
    calls.clear()
    session = api.create_inference_session(restored, artifacts=validation_artifacts)
    predictions = pl.concat(
        session.iter_predictions(dataset.validation_samples, dataset.event_store)
    )
    assert len(predictions) == len(validation)
    assert calls and set(calls) == {precision == "bf16"}
    assert not torch.is_autocast_enabled("cpu")


@pytest.mark.parametrize("stage", ["training", "inference"])
def test_unsupported_precision_fails_before_execution(stage):
    config = _synthetic_config()
    getattr(config, stage).mixed_precision = "fp16"
    with pytest.raises(ValueError, match=f"{stage}.mixed_precision"):
        config.validate()
