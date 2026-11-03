"""End-to-end smoke coverage of the shared GenRecSys lifecycle."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from data_sources import event_hub
from genrecsys import api, core
from genrecsys.core.config import Config, Metric


def _dt(day: int) -> datetime:
    return datetime(2026, 1, day, tzinfo=UTC)


def _synthetic_config() -> Config:
    config = Config.load("tests/fixtures/framework/multi-history.yaml")
    for name in ("short", "positive"):
        config.inputs[name].max_events_per_sequence = 2
    for name in ("lifelong", "transactions"):
        model_input = config.inputs[name]
        model_input.max_events_per_sequence = 3
        assert model_input.compression is not None
        model_input.compression.params.update(
            num_queries=2,
            num_layers=1,
            num_heads=2,
            dim_feedforward=32,
        )
    config.encoders["merchant_id"].preprocessor["vocab_size"] = 2
    config.model.params.update(
        dim=16,
        num_heads=2,
        dim_feedforward=32,
        dropout=0.0,
        target_chunk_size=1,
    )
    config.model.params["encoder"]["num_layers"] = 1
    config.model.params["decoder"]["num_layers"] = 1
    config.task.metrics = {
        "hit rate@2": Metric("hit_rate_at_k", {"k": 2}),
        "recall@2": Metric("recall_at_k", {"k": 2}),
        "ndcg@2": Metric("ndcg_at_k", {"k": 2}),
    }
    config.training.tracker.type = "logging"
    config.training.tracker.params = {}
    config.training.num_epochs = 1
    config.training.dataloader.batch_size = 2
    config.training.dataloader.num_workers = 0
    config.training.dataloader.pin_memory = False
    config.training.validation_dataloader.batch_size = 2
    config.training.validation_dataloader.num_workers = 0
    config.training.validation_dataloader.pin_memory = False
    assert config.training.early_stopping is not None
    config.training.early_stopping.metric = "ndcg@2"
    config.inference.dataloader.batch_size = 2
    config.inference.dataloader.num_workers = 0
    config.inference.dataloader.pin_memory = False
    config.inference.predict_kwargs = {"k": 2, "beam_size": 2}
    config.validate()
    return config


def _write_synthetic_eventhub(root: Path) -> None:
    users = {
        "u1": ("Phone", "Case", 10),
        "u2": ("Case", "Phone", 20),
    }
    for client_id, (seen, tapped, brand) in users.items():
        partition = int(
            event_hub.distribute_to_partitions(
                pl.DataFrame({"client_id": [client_id]})
            )["partition_id"][0]
        )
        filename = event_hub.partition_filename(partition)
        item_rows = {
            "item-view": (_dt(1), seen),
            "item-click": (_dt(2), tapped),
        }
        for event, (timestamp, item_label) in item_rows.items():
            path = root / event / "2026-01-01" / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            row = pl.DataFrame(
                {
                    "client_id": [client_id],
                    "timestamp": [timestamp],
                    "item_label": [item_label],
                    "source_id": ["Merchant"],
                }
            )
            if path.exists():
                row = pl.concat([pl.read_parquet(path), row])
            row.write_parquet(path)
        path = root / "merchant-visit" / "2026-01-01" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        row = pl.DataFrame(
            {
                "client_id": [client_id],
                "timestamp": [_dt(2)],
                "merchant_id": [brand],
            }
        )
        if path.exists():
            row = pl.concat([pl.read_parquet(path), row])
        row.write_parquet(path)


def test_synthetic_eventhub_caps_item_and_transaction_inputs_independently(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _synthetic_config()
    client_id = "bounded-user"
    partition = int(
        event_hub.distribute_to_partitions(pl.DataFrame({"client_id": [client_id]}))[
            "partition_id"
        ][0]
    )
    filename = event_hub.partition_filename(partition)
    eventhub = tmp_path / "eventhub"
    item_path = eventhub / "item-view" / "2026-01-01" / filename
    item_path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "client_id": [client_id] * 4,
            "timestamp": [_dt(day) for day in range(1, 5)],
            "item_label": [f"item-{day}" for day in range(1, 5)],
            "source_id": ["merchant"] * 4,
        }
    ).write_parquet(item_path)
    transaction_path = eventhub / "merchant-visit" / "2026-01-01" / filename
    transaction_path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "client_id": [client_id] * 4,
            "timestamp": [_dt(day) for day in range(1, 5)],
            "merchant_id": list(range(1, 5)),
        }
    ).write_parquet(transaction_path)
    monkeypatch.setenv("INTERNAL_STORAGE_EVENT_HUB", str(eventhub))

    events = event_hub.Reader(config).read(
        partition,
        pl.DataFrame(
            {
                "client_id": [client_id],
                "_sample_timestamp": [_dt(5)],
            }
        ),
    )

    assert events is not None
    counts = dict(events["event"].value_counts().iter_rows())
    assert counts == {"item-view": 3, "merchant-visit": 3}


def _synthetic_basis() -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    core.tasks.retrieval.Artifacts,
    core.tasks.retrieval.Artifacts,
    dict[str, pl.DataFrame],
    tuple[core.event_feature_mapping.Spec, ...],
]:
    context = {
        name: f"{name}-value"
        for name in (
            "user_category_a",
            "user_category_b",
            "user_category_c",
            "user_category_d",
            "user_category_e",
            "user_category_f",
            "user_category_g",
        )
    }
    train = pl.DataFrame(
        {
            "client_id": ["u1", "u2", "u3"],
            "timestamp": [_dt(3), _dt(3), _dt(3)],
            # Retrieval targets stay in the shared external item_id namespace.
            "target": [[101], [102], [101]],
            "context": [context, context, context],
        }
    )
    test = pl.DataFrame(
        {
            "client_id": ["u1", "u2", "u3"],
            "timestamp": [_dt(4), _dt(4), _dt(4)],
            "target": [
                [{"item": 101, "relevance": 1.0}],
                [{"item": 102, "relevance": 1.0}],
                [{"item": 101, "relevance": 1.0}],
            ],
            "context": [context, context, context],
        }
    )
    train_items = pl.DataFrame(
        {
            "item_id": [101, 102],
            "item_key": [1, 2],
        }
    )
    test_items = pl.DataFrame(
        {
            # Both retrieval catalogs use the external item_id identity.
            "item_id": [101, 102],
            "item_key": [1, 2],
        }
    )
    event_item_mapping = pl.DataFrame(
        {
            "item_label": ["Phone", "Case"],
            "source_id": ["Merchant", "Merchant"],
            "item_key": [1, 2],
        }
    )
    resources = {
        "event_item_mapping": event_item_mapping,
        "item_to_sid": pl.DataFrame(
            {
                "item_key": [1, 2],
                "sid": [[2, 6], [3, 7]],
            }
        ),
        "context_vocab": pl.DataFrame(
            {
                "feature": list(context),
                "value": list(context.values()),
                "index": range(1, len(context) + 1),
            }
        ),
        "time_vocab": pl.DataFrame(
            {
                "feature": ["interaction_weekday"] * 7
                + ["interaction_time_of_day"] * 4,
                "value": list(range(7)) + list(range(4)),
                "index": range(1, 12),
            }
        ),
        "merchant_vocab": pl.DataFrame({"value": ["10"], "index": [1]}),
    }
    mappings = (
        core.event_feature_mapping.Spec(
            feature="item",
            resource="event_item_mapping",
            attributes=("item_label", "source_id"),
            value_column="item_key",
            events=(
                "item-view",
                "item-click",
                "item-favorite",
                "item-open",
                "item-rate",
                "item-confirm",
                "item-select",
            ),
        ),
    )
    return (
        train,
        test,
        core.tasks.retrieval.Artifacts(train_items),
        core.tasks.retrieval.Artifacts(test_items),
        resources,
        mappings,
    )


def test_native_synthetic_lifecycle_uses_frozen_artifacts(
    tmp_path: Path,
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
    fp32_runtime,
) -> None:
    caplog.set_level("INFO", logger="genrecsys")
    eventhub = tmp_path / "eventhub"
    _write_synthetic_eventhub(eventhub)
    monkeypatch.setenv("INTERNAL_STORAGE_EVENT_HUB", str(eventhub))
    config = _synthetic_config()
    (
        train,
        test,
        train_artifacts,
        validation_artifacts,
        resources,
        mappings,
    ) = _synthetic_basis()

    prepared, dataset = api.prepare_dataset(
        config,
        train,
        test,
        train_artifacts=train_artifacts,
        validation_artifacts=validation_artifacts,
        resources=resources,
        event_feature_mappings=mappings,
    )
    item = prepared.encoder_to_preprocessor["item"]
    assert isinstance(item, core.encoders.semantic_id.Preprocessor)
    assert set(prepared.event_feature_mappers) == {"item"}
    assert set(dataset.event_store.open_reader(6).read("u1").columns) >= {
        "_event",
        "item",
        "merchant_id",
        "interaction_weekday",
        "interaction_time_of_day",
    }
    assert {"item_label", "source_id"}.isdisjoint(
        dataset.event_store.open_reader(6).read("u1").columns
    )
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        message.startswith("processing EventHub partition") for message in messages
    )
    assert any(
        message.startswith("processed EventHub partition") for message in messages
    )
    assert any(
        message.startswith("encoding EventStore partition") for message in messages
    )
    assert any(
        message.startswith("encoded EventStore partition") for message in messages
    )

    dataset_path = tmp_path / "dataset"
    checkpoint = tmp_path / "checkpoint"
    dataset.save(dataset_path)
    prepared.save(checkpoint)
    loaded_dataset = core.Dataset.load(dataset_path, config)
    loaded_prepared = core.checkpoint.Prepared.load(checkpoint)
    assert not hasattr(loaded_dataset.train_artifacts, "resources")
    assert set(loaded_prepared.event_feature_mappers) == {"item"}
    assert len(item.mapping) == 2

    trained = api.fit_model(loaded_prepared, loaded_dataset)
    model_inputs = {
        value["name"]: value for value in trained.model.model_config["inputs"]
    }
    assert (
        next(
            feature
            for feature in model_inputs["short"]["features"]
            if feature["name"] == "event"
        )["vocab_size"]
        == loaded_prepared.encoder_to_preprocessor["event"].num_embeddings
    )
    assert model_inputs["lifelong"]["compression"]["num_queries"] == 2
    assert model_inputs["transactions"]["compression"]["num_queries"] == 2
    assert "qformer_recency" in trained.model.model_config["position_embeddings"]
    if not trained.model.is_main_process:
        return
    trained.save(checkpoint)
    restored = core.checkpoint.Trained.load(checkpoint)

    samples = loaded_dataset.validation_samples.select(
        "client_id", "timestamp", "context"
    )
    predictions = pl.concat(
        prediction
        for partition, partition_samples in api.distribute_samples(samples)
        if (
            prediction := api.make_predictions(
                partition,
                partition_samples,
                restored,
                event_store=loaded_dataset.event_store,
                samples_prepared=True,
            )
        )
        is not None
    )
    assert predictions.columns == ["client_id", "timestamp", "context", "prediction"]
    assert predictions.height == 3
    assert set(
        predictions["prediction"].explode(empty_as_null=True).struct.field("item")
    ) <= {101, 102}

    live_predictions = pl.concat(
        prediction
        for partition, partition_samples in api.distribute_samples(
            test.select("client_id", "timestamp", "context")
        )
        if (
            prediction := api.make_predictions(
                partition,
                partition_samples,
                restored,
            )
        )
        is not None
    )
    assert live_predictions.height == predictions.height
    assert live_predictions.columns == predictions.columns

    evaluation_input = loaded_dataset.validation_samples.join(
        predictions.select("client_id", "prediction"),
        on="client_id",
        how="inner",
        validate="1:1",
    )
    prediction_path = tmp_path / "predictions.pq"
    evaluation_input.write_parquet(prediction_path)
    report = api.evaluate_predictions(
        config,
        [prediction_path],
        expected_samples=loaded_dataset.validation_samples,
    )
    assert report["samples"] == 3
    assert report["population_verified"] is True
    assert report["identity_columns"] == ["client_id", "timestamp"]
    assert set(report["metrics"]) == {"hit rate@2", "recall@2", "ndcg@2"}

    incomplete_path = tmp_path / "incomplete-predictions.pq"
    evaluation_input.head(2).write_parquet(incomplete_path)
    with pytest.raises(ValueError, match="do not cover the full inference basis"):
        api.evaluate_predictions(
            config,
            [incomplete_path],
            expected_samples=loaded_dataset.validation_samples,
        )

    identified_path = tmp_path / "identified-predictions.pq"
    evaluation_input.with_row_index("sample_id").write_parquet(identified_path)
    identified_report = api.evaluate_predictions(
        config,
        [identified_path],
        expected_samples=loaded_dataset.validation_samples.with_row_index("sample_id"),
    )
    assert identified_report["identity_columns"] == ["sample_id"]
