"""T-ECD Small Marketplace preserves public targets and adds native features."""

# ruff: noqa: DTZ001

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from model_datasets.tecd.marketplace import build_small_marketplace


@pytest.fixture
def raw_small_marketplace(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    events = root / "dataset/small/marketplace/events"
    events.mkdir(parents=True)
    start = datetime(2025, 1, 1, 12)
    rows = []
    for day in range(12):
        for user in range(10):
            target = start + timedelta(days=day, minutes=user)
            for offset, action in (
                (-15, "view"),
                (-3, "like"),
                (-2, "clickout"),
                (0, "click"),
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
                        "subdomain": "u2i",
                        "os": "ios" if user % 2 else "android",
                    }
                )
    pl.DataFrame(rows).write_parquet(events / "events.pq")
    vectors = pl.Series(
        "embedding",
        [
            np.full(300, item + 1, dtype=np.float32).tolist() if item != 3 else None
            for item in range(4)
        ],
        dtype=pl.List(pl.Float32),
    )
    pl.DataFrame(
        {
            "item_id": list(range(4)),
            "brand_id": [0, 1, 2, 0],
            "category": [0, 0, 1, 1],
            "subcategory": list(range(4)),
            "price": [10.0, 20.0, 30.0, 40.0],
            "embedding": vectors,
        }
    ).write_parquet(root / "dataset/small/marketplace/items.pq")
    pl.DataFrame(
        {
            "user_id": list(range(9)),
            "socdem_cluster": [user % 3 for user in range(9)],
            "region": [user % 2 for user in range(9)],
        }
    ).write_parquet(root / "dataset/small/users.pq")
    return root


def _public_test_targets(source: Path) -> pl.DataFrame:
    targets = (
        pl.read_parquet(source / "dataset/small/marketplace/events")
        .filter(pl.col("action_type") == "click")
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


def test_marketplace_basis_preserves_public_test_targets(
    raw_small_marketplace: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "basis"
    build_small_marketplace(source_dir=raw_small_marketplace, output_dir=output)

    frames = {
        split: pl.read_parquet(output / f"basis/{split}/samples.pq")
        for split in ("train", "validation", "test")
    }
    actual = (
        frames["test"]
        .select("client_id", "target_ts", "target")
        .explode("target_ts", "target")
        .sort("target_ts", "client_id", "target")
    )
    assert actual.equals(_public_test_targets(raw_small_marketplace))
    for frame in frames.values():
        assert (
            (frame["target_ts"].list.first() - frame["history_timestamp"])
            .eq(timedelta(hours=12))
            .all()
        )
        assert frame["context"].struct.fields == ["socdem_cluster", "region"]

    candidates = pl.read_parquet(output / "basis/test/artifacts/items.pq")
    assert candidates.columns == [
        "item_id",
        "brand_id",
        "category",
        "subcategory",
        "price",
    ]
    assert len(candidates.filter(pl.col("item_id") == "4")) == 1
    report = json.loads((output / "report.json").read_text())
    assert report["target_action"] == "click"
    assert (
        report["test_target_digest"] == report["perseus_reference_test_target_digest"]
    )
    assert report["item_catalog"] == {
        "source_items": 4,
        "source_null_embeddings": 1,
        "event_items": 5,
        "imputed_items": 1,
        "imputation": "null-features-null-300d-embedding",
    }
    assert report["embeddings"] == {
        "dimension": 300,
        "sid_fit_items": 3,
        "sid_fit_missing_excluded": 2,
        "frozen_apply_items": 5,
        "frozen_apply_zero_filled": 2,
        "missing_policy": "exclude-from-fit-zero-fill-for-apply",
    }
    train_embedding_items = pl.read_parquet(output / "representations/train/items.pq")[
        "item_id"
    ].to_list()
    assert train_embedding_items == ["0", "1", "2"]
    assert np.load(output / "representations/train/embeddings.npy").shape == (3, 300)
    all_items = pl.read_parquet(output / "representations/all/items.pq")[
        "item_id"
    ].to_list()
    all_embeddings = np.load(output / "representations/all/embeddings.npy")
    assert all_embeddings.shape == (5, 300)
    assert not all_embeddings[all_items.index("3")].any()
    assert not all_embeddings[all_items.index("4")].any()
    event_names = {path.parts[-3] for path in (output / "eventhub").rglob("*.pq")}
    assert event_names == {
        "t-ecd-small-marketplace-view",
        "t-ecd-small-marketplace-click",
        "t-ecd-small-marketplace-like",
        "t-ecd-small-marketplace-clickout",
    }


def test_marketplace_comparison_basis_has_no_gap_or_views(
    raw_small_marketplace: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "comparison-basis"
    build_small_marketplace(
        source_dir=raw_small_marketplace,
        output_dir=output,
        source_revision="pinned-test-revision",
        history_gap_hours=0,
        include_views=False,
    )

    test = pl.read_parquet(output / "basis/test/samples.pq")
    assert "history_timestamp" not in test.columns
    actual = (
        test.select("client_id", "target_ts", "target")
        .explode("target_ts", "target")
        .sort("target_ts", "client_id", "target")
    )
    assert actual.equals(_public_test_targets(raw_small_marketplace))
    event_names = {path.parts[-3] for path in (output / "eventhub").rglob("*.pq")}
    assert event_names == {
        "t-ecd-small-marketplace-click",
        "t-ecd-small-marketplace-like",
        "t-ecd-small-marketplace-clickout",
    }
    report = json.loads((output / "report.json").read_text())
    assert report["history_gap_hours"] == 0
    assert report["includes_views"] is False
    assert report["source_revision"] == "pinned-test-revision"
