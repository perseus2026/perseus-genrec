"""Port Perseus Yambda preprocessing into the OneRec data contract."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from data_sources.event_hub import distribute_to_partitions, partition_filename
from genrecsys.utils.files import publish_directory
from genrecsys.utils.progress import log_phase, logged_stage

logger = logging.getLogger(__name__)
SOURCE_REPOSITORY = "yandex/yambda"
SOURCE_REVISION = "dd6f3a19eef5866e346c3270e098baa641a44948"
PREPROCESSING_REVISION = "a3a248cc493a03fe3602952f4e61af1732d8a452"
EVENT_NAMES = {name: f"yambda-50m-{name}" for name in ("like", "dislike", "listen")}
ITEM_FEATURES = ("album_id", "artist_id", "item_track_length_seconds")
EMBEDDING_DIM = 128
DOWNLOAD_ATTEMPTS = 5
DOWNLOAD_RETRY_DELAY_SECONDS = 5


def download_source(
    *,
    output_dir: Path,
    revision: str = SOURCE_REVISION,
    dataset_size: str = "50m",
    like_only: bool = False,
) -> None:
    """Download the selected interactions and shared metadata/audio files."""
    from huggingface_hub import hf_hub_download
    from requests.exceptions import ChunkedEncodingError

    if dataset_size not in ("50m", "500m"):
        raise ValueError("Yambda dataset_size must be 50m or 500m")
    filenames = [f"flat/{dataset_size}/likes.parquet", "embeddings.parquet"]
    if not like_only:
        filenames += [
            f"flat/{dataset_size}/dislikes.parquet",
            f"flat/{dataset_size}/listens.parquet",
            "album_item_mapping.parquet",
            "artist_item_mapping.parquet",
        ]
    for filename in filenames:
        with log_phase(logger, f"download {filename}", output_path=output_dir):
            for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
                try:
                    hf_hub_download(
                        repo_id=SOURCE_REPOSITORY,
                        repo_type="dataset",
                        revision=revision,
                        filename=filename,
                        local_dir=output_dir,
                    )
                    break
                except ChunkedEncodingError as error:
                    if attempt == DOWNLOAD_ATTEMPTS:
                        raise
                    # Keep the partial file so Hugging Face can resume it.
                    logger.warning(
                        "Interrupted download %s (attempt %d/%d); retrying in %ds: %s",
                        filename,
                        attempt,
                        DOWNLOAD_ATTEMPTS,
                        DOWNLOAD_RETRY_DELAY_SECONDS,
                        error,
                    )
                    time.sleep(DOWNLOAD_RETRY_DELAY_SECONDS)
        logger.info(
            "Downloaded %s bytes=%s", filename, (output_dir / filename).stat().st_size
        )


def _events(
    raw: pl.DataFrame, name: str, *, include_organic: bool = True
) -> pl.DataFrame:
    # Preserve the benchmark's second-based interpretation of the binned clock.
    # Sorting makes tie redistribution reproducible despite Polars unique ordering.
    columns = ["is_organic"] if include_organic else []
    if name == "listen":
        columns += ["played_ratio_pct", "track_length_seconds"]
    return (
        raw.select(
            pl.from_epoch("timestamp", time_unit="s")
            .cast(pl.Datetime("ns"))
            .alias("timestamp"),
            pl.col("uid").cast(pl.String).alias("client_id"),
            pl.col("item_id").cast(pl.String),
            *columns,
        )
        .unique()
        .sort("timestamp", "client_id", "item_id", *columns)
    )


def _split(likes: pl.DataFrame) -> tuple[dict[str, pl.DataFrame], pl.DataFrame, dict]:
    """Split at the 75%/80% clocks and derive warm items from training only."""
    counts = likes.group_by("client_id").len(name="count")
    threshold = counts["count"].quantile(0.99)
    likes = likes.join(
        counts.filter(pl.col("count") <= threshold).select("client_id"),
        on="client_id",
        how="semi",
    )
    likes = likes.sort("timestamp", "client_id", maintain_order=True).with_columns(
        (
            pl.col("timestamp")
            + pl.duration(
                microseconds=pl.int_range(pl.len()).over("timestamp", "client_id")
            )
        ).cast(pl.Datetime("ns"))
    )
    basis = (
        likes.group_by("timestamp", "client_id")
        .agg(pl.col("item_id").unique())
        .sort("timestamp", "client_id")
    )
    if basis.is_empty() or basis["item_id"].list.len().max() != 1:
        raise ValueError("Yambda timestamp redistribution must yield singleton targets")
    basis = basis.with_columns(pl.col("item_id").list.first())
    n = len(basis)
    if not 0 < int(n * 0.75) < int(n * 0.8) < n:
        raise ValueError("Not enough Yambda likes for a 75/5/20 split")
    # Index as integers: Python datetime would silently lose nanoseconds.
    validation_cutoff = int(basis["timestamp"].cast(pl.Int64)[int(n * 0.75)])
    test_cutoff = int(basis["timestamp"].cast(pl.Int64)[int(n * 0.8)])
    train = basis.filter(pl.col("timestamp").cast(pl.Int64) < validation_cutoff)
    validation = basis.filter(
        (pl.col("timestamp").cast(pl.Int64) >= validation_cutoff)
        & (pl.col("timestamp").cast(pl.Int64) < test_cutoff)
    )
    test = basis.filter(pl.col("timestamp").cast(pl.Int64) >= test_cutoff)
    items = train["item_id"].unique()
    iterations = 0
    while True:
        iterations += 1
        train = train.filter(pl.len().over("client_id") > 1)
        positive = train.filter(pl.int_range(pl.len()).over("client_id") > 0)[
            "item_id"
        ].unique()
        if len(positive) == len(items):
            break
        if positive.is_empty():
            raise ValueError("Yambda warm-item filtering removed every item")
        items = positive
        train = train.filter(pl.col("item_id").is_in(items.implode()))
    splits = {
        "train": train,
        "validation": validation.filter(pl.col("item_id").is_in(items.implode())),
        "test": test.filter(pl.col("item_id").is_in(items.implode())),
    }
    if any(frame.is_empty() for frame in splits.values()):
        raise ValueError("Yambda filtering produced an empty split")
    retained = pl.concat(list(splits.values()))
    history = likes.join(retained, on=["timestamp", "client_id", "item_id"], how="semi")
    return (
        splits,
        history,
        {
            "raw_like_samples": n,
            "active_like_threshold": threshold,
            "warm_filter_iterations": iterations,
            "validation_cold_target_samples": len(validation)
            - len(splits["validation"]),
            "test_cold_target_samples": len(test) - len(splits["test"]),
            "validation_cutoff_ns": validation_cutoff,
            "test_cutoff_ns": test_cutoff,
            "timestamp_units": "benchmark uses source ticks as seconds; likes ties add microseconds",
        },
    )


def _samples(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.struct("client_id", "timestamp", "item_id").hash(seed=0).alias("sample_id"),
        target=pl.concat_list("item_id"),
        target_labels=pl.concat_list(pl.lit(1.0, dtype=pl.Float32)),
        target_ts=pl.concat_list("timestamp"),
    ).select(
        "sample_id", "client_id", "timestamp", "target", "target_labels", "target_ts"
    )


def _digest(frame: pl.DataFrame, columns: list[str]) -> str:
    # Arrow IPC preserves nanosecond identities, unlike iter_rows datetime values.
    data = frame.select(columns).sort(columns).write_ipc(None)
    return hashlib.sha256(data.getvalue()).hexdigest()


def _write_events(
    root: Path, name: str, events: pl.DataFrame, *, event_names: dict = EVENT_NAMES
) -> None:
    values = distribute_to_partitions(
        events.with_columns(pl.col("timestamp").dt.date().alias("date"))
    )
    started = time.perf_counter()
    groups = values.partition_by(
        ["date", "partition_id"], as_dict=True, maintain_order=True
    )
    for index, ((day, partition), frame) in enumerate(groups.items(), 1):
        destination = root / event_names[name] / day.isoformat()
        destination.mkdir(parents=True, exist_ok=True)
        frame.drop("date", "partition_id").sort(
            "client_id", "timestamp", "item_id"
        ).write_parquet(destination / partition_filename(int(partition)))
        if index % 100 == 0 or index == len(groups):
            elapsed = time.perf_counter() - started
            logger.info(
                "EventHub %s partitions=%s/%s rows=%s elapsed=%.1fs ETA=%.1fs",
                name,
                index,
                len(groups),
                len(events),
                elapsed,
                elapsed / index * (len(groups) - index),
            )


def _representations(
    source: Path, root: Path, all_items: pl.DataFrame, train_items: pl.DataFrame
) -> dict:
    """Stream the 7.72M-track source, preserving every requested item with zero fallback."""
    ids = all_items.select("item_id").unique().sort("item_id").with_row_index("row")
    all_dir = root / "all"
    all_dir.mkdir(parents=True)
    ids.select("item_id").write_parquet(all_dir / "items.pq")
    matrix = np.lib.format.open_memmap(
        all_dir / "embeddings.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(ids), EMBEDDING_DIM),
    )
    matrix[:] = 0
    found = np.zeros(len(ids), dtype=bool)
    # Build the numeric lookup once; joining every 8192-row batch to millions
    # of requested IDs would repeatedly rebuild the same large hash table.
    numeric_ids = ids["item_id"].cast(pl.UInt32).to_numpy()
    numeric_order = np.argsort(numeric_ids)
    sorted_ids = numeric_ids[numeric_order]
    parquet = pq.ParquetFile(source)
    read = 0
    started = last_log = time.perf_counter()
    for batch in parquet.iter_batches(
        batch_size=8192, columns=["item_id", "normalized_embed"]
    ):
        read += len(batch)
        values = pl.from_arrow(batch)
        positions = np.searchsorted(sorted_ids, values["item_id"].to_numpy())
        valid = positions < len(sorted_ids)
        valid[valid] &= (
            sorted_ids[positions[valid]] == values["item_id"].to_numpy()[valid]
        )
        selected = values.filter(pl.Series(valid))
        if len(selected):
            vectors = (
                selected["normalized_embed"]
                .list.to_array(EMBEDDING_DIM)
                .to_numpy()
                .astype(np.float32)
            )
            if (
                not np.isfinite(vectors).all()
                or selected["normalized_embed"].null_count()
            ):
                raise ValueError("Yambda audio embeddings must be finite 128d vectors")
            rows = numeric_order[positions[valid]]
            if found[rows].any() or len(np.unique(rows)) != len(rows):
                raise ValueError("Yambda source has duplicate embedding item IDs")
            matrix[rows] = vectors
            found[rows] = True
        now = time.perf_counter()
        if now - last_log >= 30 or read == parquet.metadata.num_rows:
            elapsed = now - started
            logger.info(
                "Audio scan rows=%s/%s found=%s/%s elapsed=%.1fs rows/s=%.1f ETA=%.1fs",
                read,
                parquet.metadata.num_rows,
                found.sum(),
                len(ids),
                elapsed,
                read / max(elapsed, 0.001),
                elapsed / read * (parquet.metadata.num_rows - read),
            )
            last_log = now
    matrix.flush()
    train = (
        train_items.unique()
        .sort("item_id")
        .join(ids, on="item_id", how="left", validate="1:1")
    )
    if train["row"].null_count():
        raise ValueError("SID fit items are absent from the complete history universe")
    # Match T-ECD: train target scope, with the same explicit zero-vector policy.
    missing_train = int((~found[train["row"].to_numpy()]).sum())
    train_dir = root / "train"
    train_dir.mkdir()
    train.select("item_id").write_parquet(train_dir / "items.pq")
    np.save(
        train_dir / "embeddings.npy",
        np.asarray(matrix[train["row"].to_numpy()]),
        allow_pickle=False,
    )
    missing = int((~found).sum())
    logger.info(
        "Audio coverage=%s/%s; zero-imputed=%s; SID fit train items=%s",
        found.sum(),
        len(ids),
        missing,
        len(train),
    )
    return {
        "column": "normalized_embed",
        "dimension": EMBEDDING_DIM,
        "items": len(ids),
        "missing_items": missing,
        "train_fit_items": len(train),
        "train_target_items_without_embedding": missing_train,
        "missing_policy": "zero vectors, as in T-ECD; no target/history item is dropped",
    }


@logged_stage
def build_basis(
    *,
    source_dir: Path,
    output_dir: Path,
    source_revision: str = SOURCE_REVISION,
    validation_size: int = 5000,
    seed: int = 42,
    dataset_size: str = "50m",
    like_only: bool = False,
) -> None:
    """Publish a train-warm 75/5/20 basis, metadata and train-fit vectors."""
    if validation_size < 1:
        raise ValueError("validation_size must be positive")
    if dataset_size not in ("50m", "500m"):
        raise ValueError("Yambda dataset_size must be 50m or 500m")
    source_events = source_dir / "flat" / dataset_size
    with log_phase(logger, "read Yambda interactions"):
        likes = pl.read_parquet(source_events / "likes.parquet")
        # The historical 500M likes-only protocol drops organic before dedup.
        # Four-way comparisons preserve it, as in the 50M feature recipes.
        events = {
            "like": _events(
                likes, "like", include_organic=dataset_size == "50m" or not like_only
            )
        }
        features = events["like"].select("item_id").unique()
        if not like_only:
            features, auxiliary = _read_auxiliary(source_dir, source_events, likes)
            events.update(auxiliary)
        del likes
    with log_phase(logger, "split Yambda likes and filter warm items"):
        raw_splits, events["like"], split_report = _split(events["like"])
        samples = {name: _samples(frame) for name, frame in raw_splits.items()}
        full_validation = samples["validation"]
        # Seeded hash selection also survives an unrelated input row-order change.
        samples["validation"] = (
            full_validation.with_columns(
                pl.col("sample_id").hash(seed=seed).alias("selection")
            )
            .sort("selection", "sample_id")
            .head(validation_size)
            .drop("selection")
            .sort("timestamp", "client_id")
        )
        train_items = raw_splits["train"].select("item_id").unique()
        catalog = train_items.sort("item_id")
        clients = pl.concat(list(raw_splits.values())).select("client_id").unique()
        events = {
            name: frame.join(clients, on="client_id", how="semi")
            for name, frame in events.items()
        }
        universe = (
            pl.concat(
                [frame.select("item_id") for frame in events.values()] + [catalog]
            )
            .unique()
            .sort("item_id")
        )
        features = universe.join(features, on="item_id", how="left", validate="1:1")
    with publish_directory(output_dir, overwrite=False) as output:
        for name, frame in {**samples, "validation-full": full_validation}.items():
            destination = output / "basis" / name
            (destination / "artifacts").mkdir(parents=True)
            frame.write_parquet(destination / "samples.pq")
            catalog.write_parquet(destination / "artifacts" / "items.pq")
        features.write_parquet(output / "item_features.pq")
        with log_phase(
            logger, "write Yambda audio representations", output_path=output
        ):
            embeddings = _representations(
                source_dir / "embeddings.parquet",
                output / "representations",
                universe,
                train_items,
            )
        event_names = {name: f"yambda-{dataset_size}-{name}" for name in events}
        for name, frame in events.items():
            with log_phase(logger, f"write EventHub {name}", output_path=output):
                _write_events(output / "eventhub", name, frame, event_names=event_names)
        report = {
            "dataset": f"yambda-{dataset_size}",
            "protocol": "perseus-train-warm-75-5-20-next-like-v1",
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": source_revision,
            "perseus_preprocessing_revision": PREPROCESSING_REVISION,
            "split": split_report,
            "split_ratios": [0.75, 0.05, 0.2],
            "rows": {name: len(frame) for name, frame in samples.items()}
            | {"validation_full": len(full_validation)},
            "validation_seed": seed,
            "validation_requested_size": validation_size,
            "validation_subset_digest": _digest(samples["validation"], ["sample_id"]),
            "test_target_digest": _digest(
                raw_splits["test"].with_columns(pl.col("timestamp").cast(pl.Int64)),
                ["client_id", "timestamp", "item_id"],
            ),
            "candidate_digest": _digest(catalog, ["item_id"]),
            "candidate_items": len(catalog),
            "candidate_policy": "unique filtered train like target items",
            "validation_only_candidate_items": len(
                catalog.join(train_items, on="item_id", how="anti")
            ),
            "history_items": len(universe),
            "event_rows": {name: len(frame) for name, frame in events.items()},
            "history_gap_hours": 0,
            "history_cutoff": "strictly before sample timestamp; rolling test history",
            "sid_fit_scope": "train target items only; missing audio uses the recorded zero-vector policy",
            "embeddings": embeddings,
            "seen_filtering": False,
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        logger.info("Yambda basis complete: %s", json.dumps(report))


def _read_auxiliary(
    source_dir: Path, source_events: Path, likes: pl.DataFrame
) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    """Read the 50M heterogeneous recipe only when it is requested."""
    with log_phase(logger, "read Yambda auxiliary events and metadata"):
        dislikes = pl.read_parquet(source_events / "dislikes.parquet")
        listens = (
            pl.scan_parquet(source_events / "listens.parquet")
            .filter(pl.col("played_ratio_pct") > 50)
            .collect(engine="streaming")
        )
        durations = listens.group_by("item_id").agg(
            pl.col("track_length_seconds")
            .mean()
            .cast(pl.Int64)
            .alias("item_track_length_seconds")
        )
        item_ids = pl.concat(
            [frame.select("item_id") for frame in (likes, dislikes, listens)]
        ).unique()
        features = item_ids.join(durations, on="item_id", how="left")
        for name in ("album", "artist"):
            mapping = (
                pl.read_parquet(source_dir / f"{name}_item_mapping.parquet")
                .group_by("item_id")
                .agg(pl.col(f"{name}_id").first())
            )
            features = features.join(mapping, on="item_id", how="left")
        features = features.with_columns(
            pl.col("item_id", "album_id", "artist_id").cast(pl.String)
        )
        return features, {
            "dislike": _events(dislikes, "dislike"),
            "listen": _events(listens, "listen"),
        }
