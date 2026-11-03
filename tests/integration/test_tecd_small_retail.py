"""T-ECD Small Retail keeps Perseus targets while using the native lifecycle."""

# ruff: noqa: DTZ001

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from data_sources import event_hub
from genrecsys import core
from genrecsys.cli import train
from genrecsys.core.models.onerec import config as onerec_config
from item_representations.semantic_ids.mapping import fit_sid_mapper
from model_datasets.tecd.materialize import materialize_basis
from model_datasets.tecd.retail import (
    archive_eventhub,
    build_small_retail,
    extract_eventhub,
)


@pytest.fixture
def raw_small_retail(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    events = root / "dataset/small/retail/events"
    events.mkdir(parents=True)
    start = datetime(2025, 1, 1, 12)
    rows = []
    for day in range(12):
        for user in range(10):
            target = start + timedelta(days=day, minutes=user, hours=3)
            for offset, action in (
                (-3, "view"),
                (-2, "click"),
                (-1, "added_to_cart"),
                (0, "order"),
            ):
                timestamp = target + timedelta(hours=offset)
                rows.append(
                    {
                        "timestamp": timedelta(
                            microseconds=int(timestamp.timestamp() * 1_000_000)
                        ),
                        "user_id": user,
                        "item_id": (day + user) % 5,
                        "action_type": action,
                        "subdomain": "retail",
                        "os": "ios" if user % 2 else "android",
                    }
                )
    pl.DataFrame(rows).write_parquet(events / "events.pq")
    # Item 4 is intentionally absent: targets and histories must survive a source
    # catalog gap and receive a transparent fallback SID representation.
    vectors = [np.full(300, item + 1, dtype=np.float32).tolist() for item in range(4)]
    pl.DataFrame(
        {
            "item_id": list(range(4)),
            "brand_id": [0, 1, 2, 0],
            "category": [0, 0, 1, 1],
            "subcategory": list(range(4)),
            "price": [10.0, 20.0, 30.0, 40.0],
            "embedding": vectors,
        },
        schema_overrides={"embedding": pl.Array(pl.Float32, 300)},
    ).write_parquet(root / "dataset/small/retail/items.pq")
    pl.DataFrame(
        {
            # User 9 is intentionally absent: public Perseus keeps such target rows
            # after its left join, and OneRec must map missing context to unknown 0.
            "user_id": list(range(9)),
            "socdem_cluster": [user % 3 for user in range(9)],
            "region": [user % 2 for user in range(9)],
        }
    ).write_parquet(root / "dataset/small/users.pq")
    return root


def _perseus_test_targets(source: Path) -> pl.DataFrame:
    """Independent transcription of the published Perseus 80/20 target policy."""

    targets = (
        pl.read_parquet(source / "dataset/small/retail/events")
        .filter(pl.col("action_type") == "order")
        .select(
            pl.col("timestamp")
            .dt.total_microseconds()
            .cast(pl.Datetime("us"))
            .dt.truncate("1s")
            .cast(pl.Datetime("ns")),
            pl.col("user_id").cast(pl.String).alias("client_id"),
            pl.col("item_id").cast(pl.String),
        )
    )
    counts = targets.group_by("client_id").len(name="count")
    threshold = counts["count"].quantile(0.99)
    targets = (
        targets.join(
            counts.filter(pl.col("count") <= threshold).select("client_id"),
            on="client_id",
            how="semi",
        )
        .sort("timestamp", "client_id", maintain_order=True)
        .with_columns(
            pl.col("timestamp")
            + pl.duration(
                microseconds=pl.int_range(pl.len()).over("timestamp", "client_id")
            )
        )
    )
    basis = (
        targets.group_by("timestamp", "client_id")
        .agg(pl.col("item_id").unique().alias("target"))
        .sort("timestamp")
        .with_columns(item_id=pl.col("target").list.first())
    )
    boundary = basis["timestamp"][int(len(basis) * 0.8)]
    development = basis.filter(pl.col("timestamp") < boundary)
    test = basis.filter(pl.col("timestamp") >= boundary)
    items = development["item_id"].unique()
    while True:
        development = development.filter(pl.len().over("client_id") > 1)
        positives = development.filter(pl.int_range(pl.len()).over("client_id") > 0)[
            "item_id"
        ].unique()
        if len(positives) == len(items):
            break
        items = positives
        development = development.filter(pl.col("item_id").is_in(items))
        test = test.filter(pl.col("item_id").is_in(items))
    return test.select(
        "client_id",
        pl.col("timestamp").alias("target_ts"),
        pl.col("item_id").alias("target"),
    ).sort("target_ts", "client_id", "target")


def test_basis_preserves_test_targets_and_adds_validation_and_gap(
    raw_small_retail: Path, tmp_path: Path
) -> None:
    output = tmp_path / "basis"
    build_small_retail(source_dir=raw_small_retail, output_dir=output)

    frames = {
        split: pl.read_parquet(output / f"basis/{split}/samples.pq")
        for split in ("train", "validation", "test")
    }
    assert all(len(frame) for frame in frames.values())
    for frame in frames.values():
        assert frame["target_ts"].list.first().equals(frame["timestamp"])
        gaps = frame["target_ts"].list.first() - frame["history_timestamp"]
        assert gaps.eq(timedelta(hours=12)).all()
        assert frame["target_labels"].to_list() == [[1.0]] * len(frame)
        assert frame["context"].struct.fields == ["socdem_cluster", "region"]

    actual_test = (
        frames["test"]
        .select("client_id", "target_ts", "target")
        .explode("target_ts", "target")
        .sort("target_ts", "client_id", "target")
    )
    assert actual_test.equals(_perseus_test_targets(raw_small_retail))
    report = json.loads((output / "report.json").read_text())
    assert (
        report["test_target_digest"] == report["perseus_reference_test_target_digest"]
    )
    assert report["split"] == {
        "train_ratio": 0.7,
        "validation_ratio": 0.1,
        "test_ratio": 0.2,
        "test_created_before_validation_split": True,
    }
    assert report["item_catalog"] == {
        "source_items": 4,
        "event_items": 5,
        "imputed_items": 1,
        "imputation": "null-features-zero-300d-embedding",
    }

    candidates = {
        split: pl.read_parquet(output / f"basis/{split}/artifacts/items.pq")
        for split in frames
    }
    assert candidates["train"].equals(candidates["validation"])
    assert candidates["train"].equals(candidates["test"])
    assert candidates["train"].columns == [
        "item_id",
        "brand_id",
        "category",
        "subcategory",
        "price",
    ]
    missing_item = candidates["train"].filter(pl.col("item_id") == "4")
    assert len(missing_item) == 1
    assert missing_item.select(
        pl.col("brand_id", "category", "subcategory", "price").is_null().all()
    ).row(0) == (
        True,
        True,
        True,
        True,
    )
    train_targets = set(frames["train"]["target"].explode())
    train_embedding_items = set(
        pl.read_parquet(output / "representations/train/items.pq")["item_id"]
    )
    assert train_embedding_items == train_targets
    assert np.load(output / "representations/train/embeddings.npy").shape == (
        len(train_targets),
        300,
    )
    all_items = pl.read_parquet(output / "representations/all/items.pq")
    all_embeddings = np.load(output / "representations/all/embeddings.npy")
    assert all_embeddings.shape == (5, 300)
    missing_index = all_items["item_id"].to_list().index("4")
    assert not all_embeddings[missing_index].any()


def test_comparison_basis_has_no_gap_or_views_and_keeps_test_targets(
    raw_small_retail: Path, tmp_path: Path
) -> None:
    output = tmp_path / "comparison-basis"
    build_small_retail(
        source_dir=raw_small_retail,
        output_dir=output,
        source_revision="pinned-test-revision",
        history_gap_hours=0,
        include_views=False,
    )

    test = pl.read_parquet(output / "basis/test/samples.pq")
    assert "history_timestamp" not in test.columns
    assert test["target_ts"].list.first().equals(test["timestamp"])
    actual = (
        test.select("client_id", "target_ts", "target")
        .explode("target_ts", "target")
        .sort("target_ts", "client_id", "target")
    )
    assert actual.equals(_perseus_test_targets(raw_small_retail))
    event_names = {path.parts[-3] for path in (output / "eventhub").rglob("*.pq")}
    assert event_names == {
        "t-ecd-small-retail-click",
        "t-ecd-small-retail-added-to-cart",
        "t-ecd-small-retail-like",
    }
    report = json.loads((output / "report.json").read_text())
    assert report["history_gap_hours"] == 0
    assert report["includes_views"] is False
    assert report["source_revision"] == "pinned-test-revision"

    mapper = tmp_path / "comparison-mapper"
    fit_sid_mapper(
        output / "representations/train/items.pq",
        output / "representations/train/embeddings.npy",
        mapper,
        method="rq_kmeans_full",
        codebook_size=(2,),
        device="cpu",
        fit_batch_size=8,
        eval_batch_size=8,
        max_iter=2,
        n_init=1,
        mapper_only=True,
    )
    for setup, config_path in (
        ("rq1", Path("configs/tecd-small/onerec-retail-rq1.yaml")),
        ("features", Path("tests/fixtures/tecd-small/retail-features.yaml")),
    ):
        workdir = tmp_path / f"comparison-{setup}"
        materialize_basis(
            config=config_path,
            basis=output,
            sid_checkpoint=mapper / "checkpoint.pkl",
            embedding_items=output / "representations/all/items.pq",
            embeddings=output / "representations/all/embeddings.npy",
            workdir=workdir,
        )
        manifest = core.DatasetManifest.load(workdir / "manifest.json")
        assert {mapping.feature for mapping in manifest.event_feature_mappings} == (
            set() if setup == "rq1" else {"category", "subcategory", "price"}
        )
        assert manifest.metadata["protocol"]["source_revision"] == (
            "pinned-test-revision"
        )
        assert not any(
            "mutable main" in limitation
            for limitation in manifest.metadata["protocol"]["limitations"]
        )


def test_eventhub_and_materialization_keep_features_and_frozen_sid_mapper(
    raw_small_retail: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    basis = tmp_path / "basis"
    build_small_retail(source_dir=raw_small_retail, output_dir=basis)
    eventhub_files = {
        path.relative_to(basis / "eventhub"): path.read_bytes()
        for path in (basis / "eventhub").rglob("*.pq")
    }
    archive_eventhub(basis=basis)
    assert not (basis / "eventhub").exists()
    assert (basis / "eventhub.tar").is_file()
    extract_eventhub(basis=basis)
    assert not (basis / "eventhub.tar").exists()
    assert eventhub_files == {
        path.relative_to(basis / "eventhub"): path.read_bytes()
        for path in (basis / "eventhub").rglob("*.pq")
    }
    config_path = Path("tests/fixtures/tecd-small/retail-features.yaml")
    config = core.Config.load(config_path)
    assert set(config.events) == {
        "t-ecd-small-retail-click",
        "t-ecd-small-retail-added-to-cart",
        "t-ecd-small-retail-like",
    }

    test_sample = pl.read_parquet(basis / "basis/test/samples.pq").row(0, named=True)
    reader = event_hub.Reader(config, source_to_path={"tecd": basis / "eventhub"})
    history = reader.read(
        event_hub.client_partition_id(test_sample["client_id"]),
        pl.DataFrame(
            {
                "client_id": [test_sample["client_id"]],
                "_sample_timestamp": [test_sample["history_timestamp"]],
            }
        ),
    )
    assert history is not None and len(history)
    assert history["timestamp"].max() < test_sample["history_timestamp"]
    assert history["timestamp"].dt.date().max() < test_sample["target_ts"][0].date()
    assert {"item", "event", "subdomain", "brand_id"} <= set(history.columns)

    mapper = tmp_path / "mapper"
    fit_sid_mapper(
        basis / "representations/train/items.pq",
        basis / "representations/train/embeddings.npy",
        mapper,
        method="rq_kmeans_full",
        codebook_size=(2, 2, 2),
        device="cpu",
        fit_batch_size=8,
        eval_batch_size=8,
        max_iter=2,
        n_init=1,
        mapper_only=True,
    )
    workdir = tmp_path / "workdir"
    materialize_basis(
        config=config_path,
        basis=basis,
        sid_checkpoint=mapper / "checkpoint.pkl",
        embedding_items=basis / "representations/all/items.pq",
        embeddings=basis / "representations/all/embeddings.npy",
        workdir=workdir,
    )
    manifest = core.DatasetManifest.load(workdir / "manifest.json")
    assert {mapping.feature for mapping in manifest.event_feature_mappings} == {
        "category",
        "subcategory",
        "price",
    }
    assert manifest.metadata["protocol"]["holdout_targets_filtered"] is True
    assert "warm-item" in manifest.metadata["protocol"]["limitations"][0]
    provenance = json.loads((workdir / "provenance.json").read_text())
    assert provenance["sid_fit_scope"] == "70% train target items only"

    monkeypatch.setenv("INTERNAL_STORAGE_TECD", str(basis / "eventhub"))
    train.prepare_dataset(workdir=workdir)
    prepared = core.checkpoint.Prepared.load(workdir / "checkpoint")
    item_mapping = prepared.encoder_to_preprocessor["item"].mapping
    assert set(item_mapping["item_id"]) == set(
        pl.read_parquet(basis / "representations/all/items.pq")["item_id"]
    )
    model_config = onerec_config._model_config(prepared)
    assert {value["name"] for value in model_config["inputs"]} == {
        "short",
        "like",
        "lifelong",
        "user",
    }
    dataset = core.Dataset.load(workdir / "dataset", config, include_test=True)
    assert dataset.test_samples is not None
    assert len(dataset.validation_samples) and len(dataset.test_samples)
    encoded_context = pl.concat(
        [
            frame.select(pl.col("context").struct.unnest())
            for frame in (
                dataset.train_samples,
                dataset.validation_samples,
                dataset.test_samples,
            )
        ]
    )
    assert encoded_context.select(
        ((pl.col("socdem_cluster") == 0) & (pl.col("region") == 0)).any()
    ).item()


def test_cross_materialization_uses_native_brand_attributes(
    raw_small_retail: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    basis = tmp_path / "cross-basis"
    build_small_retail(
        source_dir=raw_small_retail,
        output_dir=basis,
        source_revision="pinned-test-revision",
        history_gap_hours=0,
        include_views=False,
    )
    mapper = tmp_path / "cross-mapper"
    fit_sid_mapper(
        basis / "representations/train/items.pq",
        basis / "representations/train/embeddings.npy",
        mapper,
        method="rq_kmeans_full",
        codebook_size=(2,),
        device="cpu",
        fit_batch_size=8,
        eval_batch_size=8,
        max_iter=2,
        n_init=1,
        mapper_only=True,
    )
    workdir = tmp_path / "cross-workdir"
    materialize_basis(
        config=Path("configs/tecd-small/onerec-retail-cross-global.yaml"),
        basis=basis,
        sid_checkpoint=mapper / "checkpoint.pkl",
        embedding_items=basis / "representations/all/items.pq",
        embeddings=basis / "representations/all/embeddings.npy",
        workdir=workdir,
    )

    manifest = core.DatasetManifest.load(workdir / "manifest.json")
    assert {mapping.feature for mapping in manifest.event_feature_mappings} == {
        "category",
        "subcategory",
        "price",
    }
    assert (
        pl.read_parquet_schema(workdir / "resources/item_features.pq")["brand_id"]
        == pl.String
    )

    cross_eventhub = tmp_path / "cross-eventhub"
    auxiliary_events = {
        "t-ecd-small-marketplace-click",
        "t-ecd-small-marketplace-like",
        "t-ecd-small-marketplace-clickout",
        "t-ecd-small-offers-click",
        "t-ecd-small-offers-like",
        "t-ecd-small-offers-clickout",
        "t-ecd-payments",
    }
    target_eventhub = basis / "eventhub"
    for source in (target_eventhub / "t-ecd-small-retail-click").rglob("*.pq"):
        relative = source.relative_to(target_eventhub / "t-ecd-small-retail-click")
        values = pl.read_parquet(source).select(
            "client_id", "timestamp", "item_id", "subdomain"
        )
        for event_name in auxiliary_events:
            destination = cross_eventhub / event_name / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            values.with_columns(
                pl.col("client_id").alias("brand_id"), price=pl.lit(10.0)
            ).write_parquet(destination)

    monkeypatch.setenv("INTERNAL_STORAGE_TECD", str(target_eventhub))
    monkeypatch.setenv("INTERNAL_STORAGE_TECD_CROSS", str(cross_eventhub))
    train.prepare_dataset(workdir=workdir)
    prepared = core.checkpoint.Prepared.load(workdir / "checkpoint")
    brand_values = prepared.encoder_to_preprocessor["brand_id"].values["value"]
    assert brand_values.dtype == pl.String
    assert len(brand_values)


def test_cross_materialization_combines_target_and_auxiliary_item_features(
    raw_small_retail: Path,
    tmp_path: Path,
) -> None:
    basis = tmp_path / "feature-basis"
    build_small_retail(
        source_dir=raw_small_retail,
        output_dir=basis,
        source_revision="pinned-test-revision",
        history_gap_hours=0,
        include_views=False,
    )
    mapper = tmp_path / "feature-mapper"
    fit_sid_mapper(
        basis / "representations/train/items.pq",
        basis / "representations/train/embeddings.npy",
        mapper,
        method="rq_kmeans_full",
        codebook_size=(2,),
        device="cpu",
        fit_batch_size=8,
        eval_batch_size=8,
        max_iter=2,
        n_init=1,
        mapper_only=True,
    )
    target_features = pl.read_parquet(basis / "item_features.pq")
    auxiliary_features = tmp_path / "auxiliary-item-features.pq"
    target_features.with_columns(
        (pl.lit("aux-") + pl.col("item_id")).alias("item_id")
    ).write_parquet(auxiliary_features)
    workdir = tmp_path / "feature-workdir"

    materialize_basis(
        config=Path("configs/tecd-small/onerec-retail-cross-global.yaml"),
        basis=basis,
        sid_checkpoint=mapper / "checkpoint.pkl",
        embedding_items=basis / "representations/all/items.pq",
        embeddings=basis / "representations/all/embeddings.npy",
        auxiliary_item_features=auxiliary_features,
        workdir=workdir,
    )

    manifest = core.DatasetManifest.load(workdir / "manifest.json")
    assert {mapping.feature for mapping in manifest.event_feature_mappings} == {
        "category",
        "subcategory",
        "price",
    }
    assert all(
        "t-ecd-small-offers-click" not in mapping.events
        and "t-ecd-payments" not in mapping.events
        for mapping in manifest.event_feature_mappings
    )
    assert len(pl.read_parquet(workdir / "resources/item_features.pq")) == 2 * len(
        target_features
    )
