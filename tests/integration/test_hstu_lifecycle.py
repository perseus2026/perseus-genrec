"""Real preparation, native HSTU fitting, checkpoint reload and both inference paths."""

from copy import deepcopy
from datetime import UTC, datetime

import polars as pl
import pytest
import torch

from data_sources import event_hub
from genrecsys import api, core
from genrecsys.core.config import (
    Attribute,
    Encoder,
    Event,
    Feature,
    Input,
    LocatedIn,
    Metric,
    Model,
)
from genrecsys.core.models.hstu.data import Samples, catalog, sid_lookup, vocabulary


def prepared_fixture(tmp_path, monkeypatch, *, with_context=False):
    config = core.Config.load("tests/fixtures/tecd-small/retail-features.yaml")
    config.events = {
        "main": Event(
            attributes={
                "item": Attribute("item_id"),
                "brand_id": Attribute("brand_id"),
                "price": Attribute("price"),
            }
        ),
        "aux": Event(attributes={"brand_id": Attribute("brand_id")}),
    }
    config.inputs = {
        "lifelong": Input(
            type="sequence",
            events=("main", "aux"),
            features=("item", "event", "brand_id", "price"),
            optional_features=("item",),
            max_events_per_sequence=5,
            event_encoder="history",
        )
    }
    config.features = {
        name: config.features[name] for name in ("item", "event", "brand_id", "price")
    }
    config.encoders = {
        name: config.encoders[name]
        for name in {feature.encoder for feature in config.features.values()}
    }
    config.encoders["event"].preprocessor["categories"] = ["main", "aux"]
    config.features["brand_id"].located_in.artifacts = False
    config.encoders["price"] = Encoder("ple", {"max_bins": 4})
    config.features["price"] = Feature(LocatedIn(event=True), "price")
    config.validate_data()
    hub = tmp_path / "hub"
    for user, item in [("u1", "a"), ("u2", "b")]:
        partition = event_hub.distribute_to_partitions(
            pl.DataFrame({"client_id": [user]})
        )["partition_id"][0]
        for name, day in [("main", 1), ("aux", 2)]:
            path = (
                hub / name / f"2026-01-0{day}" / event_hub.partition_filename(partition)
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "client_id": [user],
                "timestamp": [datetime(2026, 1, day, tzinfo=UTC)],
                "brand_id": ["merchant"],
            }
            if name == "main":
                row["item_id"] = [item]
                row["price"] = [10.0 if user == "u1" else 20.0]
            values = pl.DataFrame(row)
            if path.exists():
                values = pl.concat([pl.read_parquet(path), values])
            values.write_parquet(path)
    monkeypatch.setenv("INTERNAL_STORAGE_EVENT_HUB", str(hub))
    train = pl.DataFrame(
        {
            "client_id": ["u1", "u2", "cold"],
            "timestamp": [datetime(2026, 1, 3, tzinfo=UTC)] * 3,
            "target": [["a", "b"], ["b"], ["a", "c"]],
        }
    )
    validation = pl.DataFrame(
        {
            "client_id": ["u1", "u2", "cold"],
            "timestamp": [datetime(2026, 1, 4, tzinfo=UTC)] * 3,
            "target": [["b"], ["c"], ["b"]],
        }
    )
    if with_context:
        config.inputs["user"] = Input(type="context", features=("region",))
        config.encoders["region"] = Encoder("id", {})
        config.features["region"] = Feature(LocatedIn(context=True), "region")
        train = train.with_columns(context=pl.struct(pl.lit("north").alias("region")))
        validation = validation.with_columns(
            context=pl.struct(pl.lit("north").alias("region"))
        )
    prepared, dataset = api.prepare_dataset(
        config,
        train,
        validation,
        train_artifacts=core.tasks.retrieval.Artifacts(
            pl.DataFrame({"item_id": ["a", "b", "c", "z"]})
        ),
        validation_artifacts=core.tasks.retrieval.Artifacts(
            pl.DataFrame({"item_id": ["b", "c"]})
        ),
        resources={
            "item_to_sid": pl.DataFrame(
                {
                    "item_id": ["z", "c", "a", "b"],
                    "sid": [[2, 4, 9], [2, 4, 8], [2, 4, 6], [2, 4, 7]],
                }
            )
        },
    )
    return prepared, dataset


def hstu_config(
    config, reduction, checkpointing, loss_type="cross_entropy", fusion="sum"
):
    config = deepcopy(config)
    config.model = Model(
        "hstu",
        {
            "dim": 8,
            "embedding_dim": 4,
            "num_heads": 2,
            "num_layers": 1,
            "dropout": 0.0,
            "catalog_chunk_size": 2,
            "max_len": 6,
            "num_time_buckets": 64,
            "weight_initialization": "original",
            "feature_fusion": fusion,
        },
    )
    config.task.objective.type = "item_cross_entropy"
    config.task.objective.loss = {"type": loss_type, "reduction": reduction}
    if loss_type == "sampled_cross_entropy":
        config.task.objective.loss["num_negatives"] = 2
    config.task.metrics = {
        "Recall@2": Metric("recall_at_k", {"k": 2}),
        "NDCG@2": Metric("ndcg_at_k", {"k": 2}),
        "Coverage@2": Metric("coverage_at_k", {"k": 2}),
    }
    config.training.num_epochs = 1
    config.training.mixed_precision = config.inference.mixed_precision = "no"
    config.training.gradient_accumulation_steps = 2
    config.training.gradient_checkpointing = checkpointing
    config.training.early_stopping.metric = "Recall@2"
    config.training.tracker.type = "logging"
    config.inference.predict_kwargs = {"k": 2}
    for settings in [
        config.training.dataloader,
        config.training.validation_dataloader,
        config.inference.dataloader,
    ]:
        settings.batch_size, settings.num_workers, settings.pin_memory = 2, 0, False
    config.validate()
    return config


@pytest.mark.parametrize(
    "reduction,checkpointing,loss_type,with_context",
    [
        ("mean_targets_per_sample", False, "cross_entropy", False),
        ("mean_targets_global", True, "cross_entropy", False),
        ("mean_targets_per_sample", False, "sampled_cross_entropy", False),
        ("mean_targets_global", True, "sampled_cross_entropy", False),
        ("mean_targets_per_sample", True, "sampled_cross_entropy", True),
    ],
)
@pytest.mark.parametrize("fusion", ["sum", "concat_projection"])
def test_hstu_reuses_prepared_data_and_roundtrips_predictions(
    tmp_path,
    monkeypatch,
    fp32_runtime,
    reduction,
    checkpointing,
    loss_type,
    with_context,
    fusion,
):
    torch.set_num_threads(1)
    prepared, dataset = prepared_fixture(
        tmp_path, monkeypatch, with_context=with_context
    )
    dataset.save(tmp_path / "dataset")
    prepared.save(tmp_path / "checkpoint")
    original = (tmp_path / "dataset/events/data.arrow").read_bytes()
    contract = prepared.data_contract
    config = hstu_config(prepared.config, reduction, checkpointing, loss_type, fusion)
    prepared = core.Prepared.load(tmp_path / "checkpoint", config=config)
    dataset = core.Dataset.load(tmp_path / "dataset", config, memory_map_samples=True)
    vocab = vocabulary(prepared)
    assert vocab["item_id"].to_list() == ["a", "b", "c", "z"]
    samples = Samples(
        dataset.validation_samples,
        config,
        sid_lookup(vocab),
        event_store=dataset.event_store,
    )
    user_index = next(
        index
        for index in range(len(samples))
        if dataset.validation_samples.row(index, named=True)["client_id"] == "u1"
    )
    assert samples[user_index]["history"]["features"]["item"]["values"] == [[1], [0]]
    trained = api.fit_model(prepared, dataset)
    assert trained.prepared.data_contract == contract
    assert (tmp_path / "dataset/events/data.arrow").read_bytes() == original
    assert torch.isfinite(
        torch.tensor(trained.model.training["history"][0]["train_loss"])
    )
    trained.save(tmp_path / "checkpoint")
    restored = core.Trained.load(tmp_path / "checkpoint")
    assert restored.model.training["input"] == "lifelong"
    # Inference consumes bounded DataFrames, not training mmap readers.
    validation = pl.read_parquet(tmp_path / "dataset/validation/samples.pq")
    frames = list(
        api.iter_predictions(
            1,
            validation,
            restored,
            event_store=dataset.event_store,
            samples_prepared=True,
        )
    )
    predictions = pl.concat(frames)
    assert len(predictions) == 3
    assert set(predictions["prediction"].explode().struct.field("item")) == {"b", "c"}
    assert catalog(dataset.train_artifacts.items, vocab)["item_id"].to_list() == [
        "a",
        "b",
        "c",
        "z",
    ]
    path = tmp_path / "predictions.pq"
    predictions.write_parquet(path)
    report = api.evaluate_predictions(
        config,
        [path],
        expected_samples=validation,
        candidate_items=dataset.validation_artifacts.items,
    )
    assert report["population_verified"] is True
    assert report["metrics"]["Recall@2"] == 1
    assert report["metrics"]["Coverage@2"] == 1
    # Explicit inference catalogs can be narrower than the checkpoint catalog.
    session = api.create_inference_session(
        restored,
        artifacts=core.tasks.retrieval.Artifacts(pl.DataFrame({"item_id": ["b"]})),
    )
    restricted = pl.concat(session.iter_predictions(validation, dataset.event_store))
    assert restricted["prediction"].list.len().to_list() == [1, 1, 1]
    assert set(restricted["prediction"].explode().struct.field("item")) == {"b"}
    # Live EventHub inference uses the same saved preprocessors and item identities.
    live_samples = (
        validation.with_columns(context=pl.struct(pl.lit("north").alias("region")))
        if with_context
        else validation
    )
    live = pl.concat(
        prediction
        for partition, values in api.distribute_samples(live_samples)
        for prediction in api.iter_predictions(partition, values, restored)
    )
    actual = live.sort("client_id")["prediction"].to_list()
    expected = predictions.sort("client_id")["prediction"].to_list()
    for left, right in zip(actual, expected):
        assert [value["item"] for value in left] == [value["item"] for value in right]
        assert [value["score"] for value in left] == pytest.approx(
            [value["score"] for value in right], abs=1e-6
        )
