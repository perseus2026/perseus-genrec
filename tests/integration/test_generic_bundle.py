"""A synthetic bundle exercises the shared lifecycle with optional features."""

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
import yaml

from data_sources import event_hub
from genrecsys import core
from genrecsys.cli import inference, train


def _config(path: Path, *, auxiliary: bool, backbone: str) -> core.Config:
    payload = {
        "task": {
            "type": "retrieval",
            "baseline": {"type": "global_count"},
            "metrics": {
                name: {"type": kind, "params": {"k": 2}}
                for name, kind in (
                    ("hit@2", "hit_rate_at_k"),
                    ("recall@2", "recall_at_k"),
                    ("ndcg@2", "ndcg_at_k"),
                )
            },
            "objective": {
                "target": "article_id",
                "loss": {
                    "type": "cross_entropy",
                    "reduction": "mean_tokens_then_targets_per_sample",
                },
            },
        },
        "max_duration_per_sequence": "180d",
        "events": {
            name: {"attributes": {"article_id": None}} for name in ("view", "like")
        },
        "encoders": {
            "article_codes": {
                "type": "semantic_id",
                "preprocessor": {
                    "artifact": "article_sid",
                    "item_id_column": "item_id",
                },
                "embedder": {"prefix_length": 1},
            },
        },
        "features": {
            "article_id": {"located_in": {"event": True}, "encoder": "article_codes"}
        },
        "event_encoders": {
            "activity": {
                "type": "concat_projection",
                "params": {"additive_features": ["event"]},
            }
        },
        "inputs": {},
        "model": {
            "type": "onerec",
            "params": {
                "backbone": backbone,
                "dim": 16,
                "encoder": {
                    "num_layers": 1,
                    "position_embeddings": ["absolute", "qformer_recency"],
                },
                "decoder": {"num_layers": 1},
                "num_heads": 2,
                "dim_feedforward": 32,
                "dropout": 0.0,
                "weight_initialization": "xavier_uniform",
                "target_chunk_size": 1,
            },
        },
        "training": {
            "num_epochs": 1,
            "dataloader": {"batch_size": 2},
            "validation_dataloader": {"batch_size": 2},
            "optimizer": {"type": "adam", "params": {"lr": 0.001}},
        },
        "inference": {
            "dataloader": {"batch_size": 2},
            "predict_kwargs": {"k": 2, "beam_size": 2},
        },
    }
    compression = {
        "type": "qformer",
        "params": {
            "num_queries": 2,
            "num_layers": 1,
            "num_heads": 2,
            "dim_feedforward": 32,
        },
    }
    for name in ("recent", "liked", "archive"):
        payload["inputs"][name] = {
            "type": "sequence",
            "events": ["like"] if name == "liked" else ["view", "like"],
            "features": ["article_id", "event"],
            "max_events_per_sequence": 3,
            "event_encoder": "activity",
        }
    payload["inputs"]["archive"]["compression"] = compression
    if auxiliary:
        # Non-Synthetic names for every optional feature, encoder and input.
        payload["events"]["purchase"] = {"attributes": {"merchant": None}}
        payload["events"]["view"]["attributes"]["quality"] = None
        payload["events"]["like"]["attributes"]["quality"] = None
        for feature, encoder, location in (
            ("merchant", "merchant_vocab", "event"),
            ("quality", "quality_vocab", "event"),
            ("country", "audience_vocab", "context"),
        ):
            payload["encoders"][encoder] = {"type": "id"}
            payload["features"][feature] = {
                "encoder": encoder,
                "located_in": {location: True},
            }
        for name in ("recent", "liked", "archive"):
            payload["inputs"][name]["features"].append("quality")
        payload["event_encoders"]["purchase_projection"] = {"type": "identity"}
        payload["inputs"]["payments"] = {
            "type": "sequence",
            "events": ["purchase"],
            "features": ["merchant"],
            "max_events_per_sequence": 3,
            "event_encoder": "purchase_projection",
            "compression": compression,
        }
        payload["inputs"]["audience"] = {
            "type": "context",
            "features": ["country"],
            "aggregation": {"type": "concat_projection"},
        }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return core.Config.load(path)


@pytest.mark.parametrize("auxiliary", [False, True])
@pytest.mark.parametrize("backbone", ["t5", "switch_dense"])
def test_generic_bundle_lifecycle(
    tmp_path: Path, monkeypatch, auxiliary: bool, backbone: str
) -> None:
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    config = _config(workdir / "config.yaml", auxiliary=auxiliary, backbone=backbone)
    catalog = core.tasks.retrieval.Artifacts(pl.DataFrame({"item_id": [10, 20]}))
    for day, split in enumerate(("train", "validation", "test"), start=3):
        path = workdir / "basis" / split
        path.mkdir(parents=True)
        samples = pl.DataFrame(
            {
                "sample_id": [f"{split}-u1", f"{split}-u2"],
                "client_id": ["u1", "u2"],
                "timestamp": [datetime(2026, 1, day, tzinfo=UTC)] * 2,
                "target": [[10], [20]],
            }
        )
        if auxiliary:
            samples = samples.with_columns(
                pl.Series("context", [{"country": "a"}, {"country": "b"}])
            )
        samples.write_parquet(path / "samples.pq")
        catalog.save(path / "artifacts")
    resources = workdir / "resources"
    resources.mkdir()
    pl.DataFrame(
        {
            "item_id": [10, 20],
            "sid": [[2, 4, 6], [3, 5, 7]],
        }
    ).write_parquet(resources / "article_sid.pq")
    (resources / "manifest.json").write_text(
        json.dumps({"article_sid": "article_sid.pq"})
    )
    manifest = core.DatasetManifest.build(workdir)
    manifest.validate(workdir, config)
    manifest.save(workdir / "manifest.json")
    hub = tmp_path / "hub"
    for user, item in (("u1", 10), ("u2", 20)):
        partition = event_hub.distribute_to_partitions(
            pl.DataFrame({"client_id": [user]})
        )["partition_id"][0]
        for event in config.events:
            path = hub / event / "2026-01-01" / event_hub.partition_filename(partition)
            path.parent.mkdir(parents=True, exist_ok=True)
            row = {"client_id": [user], "timestamp": [datetime(2026, 1, 2, tzinfo=UTC)]}
            if event == "purchase":
                row["merchant"] = ["merchant-a"]
            else:
                row["article_id"] = [item]
                if auxiliary:
                    row["quality"] = ["high"]
            frame = pl.DataFrame(row)
            if path.exists():
                frame = pl.concat([pl.read_parquet(path), frame])
            frame.write_parquet(path)
    monkeypatch.setenv("INTERNAL_STORAGE_EVENT_HUB", str(hub))
    train.prepare_dataset(workdir=workdir)
    train.fit_baseline(workdir=workdir)
    train.fit_model(workdir=workdir)
    state = core.checkpoint.Trained.load(workdir / "checkpoint")
    assert state.prepared.dataset_identity == manifest.identity(workdir)
    assert state.model.model_config["backbone"] == backbone
    assert [value["name"] for value in state.model.model_config["inputs"]] == (
        ["archive", "audience", "liked", "payments", "recent"]
        if auxiliary
        else ["archive", "liked", "recent"]
    )
    for source in ("event_store", "event_hub"):
        inference.distribute_samples(workdir=workdir, history_source=source)
        inference.make_predictions(workdir=workdir, history_source=source)
        inference.evaluate(workdir=workdir)
        report = json.loads((workdir / "metrics.json").read_text())
        assert report["samples"] == 2 and report["population_verified"]
        predictions = pl.concat(
            [pl.read_parquet(path) for path in (workdir / "predictions").glob("*.pq")]
        )
        fields = predictions["prediction"].explode().struct.fields
        assert fields == ["item", "score"]
        assert set(predictions["prediction"].explode().struct.field("item")) <= {10, 20}
