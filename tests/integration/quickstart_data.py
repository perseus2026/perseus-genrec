"""Synthetic inputs used only by preparation and CLI integration tests."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from data_sources import event_hub
from genrecsys.core import Config
from genrecsys.core.tasks.retrieval import Artifacts


def make_data(workdir: Path) -> None:
    # Require a new directory: rerunning the example must not replace a trained model.
    workdir.mkdir(parents=True)
    Config.load(
        Path(__file__).resolve().parents[2] / "examples/quickstart/config.yaml"
    ).save(workdir / "config.yaml")
    representations = workdir / "embeddings/inference"
    representations.mkdir(parents=True)
    items = pl.DataFrame({"item_id": list(range(16))})
    items.write_parquet(representations / "items.pq")
    # Eight content features: four categories and four variants within a category.
    # These synthetic vectors replace a pretrained text embedder in this small demo.
    eye = np.eye(4, dtype=np.float32)
    vectors = np.stack([np.concatenate([eye[i // 4], eye[i % 4]]) for i in range(16)])
    np.save(representations / "embeddings.npy", vectors)
    training = workdir / "embeddings/train"
    training.mkdir()
    items.write_parquet(training / "items.pq")
    np.save(training / "embeddings.npy", vectors)

    start = datetime(2026, 1, 1, tzinfo=UTC)
    events = pl.DataFrame(
        [
            {
                "client_id": f"user-{user}",
                "timestamp": start + timedelta(days=step),
                "item_id": (user % 4) * 4 + ((step + user // 4) % 4),
            }
            for user in range(32)
            for step in range(8)
        ]
    )
    events = events.with_columns(
        pl.col("timestamp").dt.replace_time_zone(None).dt.cast_time_unit("ns")
    )
    events.write_parquet(workdir / "events.pq")
    partitioned = event_hub.distribute_to_partitions(events).with_columns(
        date=pl.col("timestamp").dt.strftime("%Y-%m-%d")
    )
    for (date, partition), frame in partitioned.partition_by(
        ["date", "partition_id"], as_dict=True
    ).items():
        path = workdir / "eventhub" / "view" / date
        path.mkdir(parents=True, exist_ok=True)
        frame.drop("date", "partition_id").write_parquet(
            path / event_hub.partition_filename(partition)
        )

    # One target per user and split; EventHub includes later events so the shared
    # reader applies the strict timestamp cutoff for each prediction request.
    for split, step in zip(("train", "validation", "test"), (5, 6, 7), strict=True):
        root = workdir / "basis" / split
        root.mkdir(parents=True)
        samples = events.filter(
            pl.col("timestamp") == (start + timedelta(days=step)).replace(tzinfo=None)
        )
        samples.select(
            (pl.lit(split + "-") + pl.col("client_id")).alias("sample_id"),
            "client_id",
            "timestamp",
            pl.concat_list("item_id").alias("target"),
        ).write_parquet(root / "samples.pq")
        Artifacts(items).save(root / "artifacts")
