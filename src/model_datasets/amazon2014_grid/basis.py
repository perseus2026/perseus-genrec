"""Prepared GRID Amazon → deterministic windows, basis and synthetic EventHub."""

from __future__ import annotations

import json
import logging
import shutil
import tarfile
from pathlib import Path
from typing import Literal

import polars as pl

from data_sources.event_hub import distribute_to_partitions, partition_filename
from genrecsys.utils.files import file_sha256, publish_directory
from model_datasets.amazon2014_grid.grid_embeddings import export_grid_embeddings
from model_datasets.amazon2014_grid.prepare import (
    Category,
    event_name,
    import_amazon,
    source_protocol,
)

logger = logging.getLogger(__name__)
Augmentation = Literal["none", "full", "prefix"]
EVENT = "amazon2014grid-beauty-review"
SOURCE = "amazon2014grid"
# A positional clock only: the archive contains no original timestamps.
EPOCH_NS = 946_684_800_000_000_000  # 2000-01-01T00:00:00Z
STEP_NS = 1_000_000_000


def _sample_frame(rows: list[tuple]) -> pl.DataFrame:
    return (
        pl.DataFrame(
            rows,
            schema={
                "sample_id": pl.String,
                "client_id": pl.String,
                "grid_user_id": pl.Int64,
                "window_start_index": pl.UInt32,
                "history_start_index": pl.UInt32,
                "target_index": pl.UInt32,
                "_target": pl.String,
                "timestamp": pl.Int64,
            },
            orient="row",
        )
        .with_columns(pl.col("timestamp").cast(pl.Datetime("ns")))
        .with_columns(
            history_timestamp=pl.col("timestamp"),
            target=pl.concat_list("_target"),
            target_ts=pl.concat_list("timestamp"),
            target_labels=pl.lit([1.0], dtype=pl.List(pl.Float32)),
            history_length=pl.col("target_index") - pl.col("history_start_index"),
        )
        .drop("_target")
    )


def _samples(
    source: Path, augmentation: Augmentation, max_history: int
) -> tuple[dict[str, pl.DataFrame], pl.DataFrame]:
    full = pl.read_parquet(source / "sequences/full.pq")
    holdouts = {
        split: {
            row["grid_user_id"]: row["item_ids"]
            for row in pl.read_parquet(source / f"sequences/{split}.pq").iter_rows(
                named=True
            )
        }
        for split in ("validation", "test")
    }
    rows: dict[str, list[tuple]] = {s: [] for s in ("train", "validation", "test")}
    segments: dict[tuple[int, int], str] = {}

    def append(split: str, user: int, start: int, target: int, sequence: list[str]):
        history_start = max(start, target - max_history)
        client = segments.setdefault(
            (user, history_start), f"grid:{user}:start:{history_start}"
        )
        rows[split].append(
            (
                f"{split}:{user}:{start}:{target}",
                client,
                user,
                start,
                history_start,
                target,
                sequence[target],
                EPOCH_NS + target * STEP_NS,
            )
        )

    for row in full.iter_rows(named=True):
        user, sequence = row["grid_user_id"], row["item_ids"]
        train_length = len(sequence) - 2
        targets = (
            [train_length - 1] if augmentation == "none" else range(1, train_length)
        )
        for target in targets:
            starts = range(target) if augmentation == "full" else [0]
            for start in starts:
                append("train", user, start, target, sequence)
        for split, target in (("validation", train_length), ("test", train_length + 1)):
            # Preserve the author-provided, already truncated holdout history.
            start = target - (len(holdouts[split][user]) - 1)
            append(split, user, start, target, sequence)

    segment_frame = pl.DataFrame(
        [(client, user, start) for (user, start), client in segments.items()],
        schema={
            "client_id": pl.String,
            "grid_user_id": pl.Int64,
            "history_start_index": pl.UInt32,
        },
        orient="row",
    ).sort("grid_user_id", "history_start_index")
    return {
        split: _sample_frame(values) for split, values in rows.items()
    }, segment_frame


def _sequences(source: Path) -> pl.DataFrame:
    return (
        pl.read_parquet(source / "sequences/full.pq")
        .rename({"client_id": "source_client_id", "item_ids": "item_id"})
        .with_columns(
            train_sequence_length=pl.col("item_id").list.len() - 2,
            sequence_index=pl.int_ranges(
                0, pl.col("item_id").list.len(), dtype=pl.UInt32
            ),
        )
        .explode("item_id", "sequence_index")
        .with_columns(
            timestamp=(
                EPOCH_NS + pl.col("sequence_index").cast(pl.Int64) * STEP_NS
            ).cast(pl.Datetime("ns")),
            split=pl.when(pl.col("sequence_index") < pl.col("train_sequence_length"))
            .then(pl.lit("train"))
            .when(pl.col("sequence_index") == pl.col("train_sequence_length"))
            .then(pl.lit("validation"))
            .otherwise(pl.lit("test")),
        )
    )


def _write_eventhub(
    root: Path, sequences: pl.DataFrame, segments: pl.DataFrame, event: str = EVENT
) -> dict:
    # Each key denotes a suffix of one original user's sequence. The shared reader
    # can now return the exact permitted prefix without a per-sample lower bound.
    events = (
        segments.join(sequences, on="grid_user_id", how="inner")
        .filter(pl.col("sequence_index") >= pl.col("history_start_index"))
        .select("client_id", "timestamp", "item_id", "grid_user_id", "sequence_index")
    )
    partitioned = distribute_to_partitions(events).with_columns(
        date=pl.col("timestamp").dt.date()
    )
    files = 0
    for (day, partition), frame in partitioned.group_by("date", "partition_id"):
        destination = root / event / day.isoformat()
        destination.mkdir(parents=True, exist_ok=True)
        frame.drop("date", "partition_id").sort("client_id", "timestamp").write_parquet(
            destination / partition_filename(int(partition))
        )
        files += 1
    return {"files": files, "rows": len(events), "history_segments": len(segments)}


def build_basis(
    *,
    output_dir: Path,
    augmentation: Augmentation,
    source_dir: Path | None = None,
    prepared_source_dir: Path | None = None,
    stage01_dir: Path | None = None,
    max_history: int = 20,
    pack_eventhub: bool = False,
    category: Category = "beauty",
) -> None:
    """Build category Stage 02 from raw TFRecords or a validated prepared import.

    Augmentation must be selected explicitly; use none for NoAug/h20.
    none: one final next-item example per training sequence; prefix: every
    nonempty training prefix; full: every contiguous training window, including
    multiplicities after history truncation. No random window sampling is used.
    max_history applies to every split and excludes the target.
    This is an upstream basis, not a SID-equipped materialized framework dataset.
    stage01_dir optionally supplies author GRID vectors for validated CPU export.
    """
    protocol, event = source_protocol(category), event_name(category)
    if augmentation not in ("none", "full", "prefix"):
        raise ValueError("augmentation must be none, full or prefix")
    if max_history < 1:
        raise ValueError("max_history must be positive")
    if (source_dir is None) == (prepared_source_dir is None):
        raise ValueError("Pass exactly one of source_dir or prepared_source_dir")
    with publish_directory(output_dir, overwrite=False) as output:
        source = output / "source"
        if prepared_source_dir is None:
            # Reuse the validated importer, including hashes, 5-core and split checks.
            import_amazon(source_dir=source_dir, output_dir=source, category=category)
        else:
            prepared_report = json.loads(
                (prepared_source_dir / "report.json").read_text()
            )
            if (
                prepared_report.get("protocol") != protocol
                or prepared_report.get("category") != category
                or not prepared_report.get("artifact_files")
            ):
                raise ValueError(
                    "Expected a validated GRID prepared source with hashes"
                )
            for relative, digest in prepared_report["artifact_files"].items():
                if file_sha256(prepared_source_dir / relative) != digest:
                    raise ValueError(f"Prepared source SHA-256 mismatch: {relative}")
            shutil.copytree(prepared_source_dir, source)
        source_report = json.loads((source / "report.json").read_text())
        samples, segments = _samples(source, augmentation, max_history)
        catalog = pl.read_parquet(source / "catalog/items.pq")
        for split, frame in samples.items():
            destination = output / "basis" / split
            (destination / "artifacts").mkdir(parents=True)
            frame.write_parquet(destination / "samples.pq")
            catalog.write_parquet(destination / "artifacts/items.pq")
        # GRID quantizer fitting uses the full catalog, including holdout-only items.
        for scope in ("train", "inference"):
            destination = output / "catalog" / scope
            destination.mkdir(parents=True)
            catalog.write_parquet(destination / "items.pq")
        sequences = _sequences(source)
        sequences.write_parquet(output / "sequences.pq")
        segments.write_parquet(output / "history_segments.pq")
        eventhub = _write_eventhub(output / "eventhub", sequences, segments, event)
        train_sequences = pl.read_parquet(source / "sequences/train.pq")[
            "item_ids"
        ].to_list()
        effective_items = {
            item
            for sequence in train_sequences
            for item in (
                sequence[-(max_history + 1) :] if augmentation == "none" else sequence
            )
        }
        diagnostics = {
            **source_report.get("sid_fit_diagnostics", {}),
            "effective_train_items": len(effective_items),
            "outside_effective_train_items": len(catalog) - len(effective_items),
            "outside_effective_train_fraction": (len(catalog) - len(effective_items))
            / len(catalog),
            "effective_train_scope": f"{augmentation}/h{max_history}: histories and targets of materialized examples",
        }
        if pack_eventhub:
            with tarfile.open(output / "eventhub.tar", "w") as archive:
                archive.add(output / "eventhub", arcname="eventhub")
            shutil.rmtree(output / "eventhub")
        if stage01_dir is not None:
            export_grid_embeddings(
                source_dir=source,
                stage01_dir=stage01_dir,
                output_dir=output / "representations",
            )
        report = {
            "protocol": f"amazon2014grid-{category}-{augmentation}-h{max_history}-v1",
            "stage": "02_basis_catalog_eventhub",
            "framework_dataset_ready": False,
            "category": category,
            "source_report_sha256": file_sha256(source / "report.json"),
            "sid_fit_diagnostics": diagnostics,
            "source_protocol": source_report["protocol"],
            "source_url": source_report["source_url"],
            "source_files": source_report["source_files"],
            "source_integrity": source_report["source_integrity"],
            "users": source_report["users"],
            "items": source_report["items"],
            "source_interactions": source_report["full_interactions"],
            "holdout_only_item_ids": source_report["holdout_only_item_ids"],
            "augmentation": {
                "mode": augmentation,
                "random_sampling": False,
                "enumeration": "user-id, target-position, window-start-position",
                "truncate_after_window_enumeration": True,
                "deduplicate_after_truncation": False,
                "full_windows_before_truncation": source_report[
                    "train_contiguous_subsequences_before_sampling"
                ],
            },
            "history": {
                "max_train_items": max_history,
                "max_holdout_items": max_history,
                "source_holdout_max_items": 20,
                "target_excluded": True,
                "client_id": "grid:<original-user-id>:start:<effective-history-start>",
                "original_user_column": "grid_user_id",
            },
            "timestamps": {
                "synthetic": True,
                "origin_utc": "2000-01-01T00:00:00Z",
                "step_seconds": 1,
                "rule": "origin + original per-user sequence position",
                "query_time": "target event time; histories are strictly earlier",
            },
            "rows": {split: len(frame) for split, frame in samples.items()},
            "eventhub": {
                "event": event,
                "source": SOURCE,
                "packed": pack_eventhub,
                **eventhub,
            },
            "candidate_policy": "full author catalog in every split",
            "quantizer_fit_scope": "full author catalog, as GRID config",
            "text_serialization": "verbatim author TFRecord text",
            "representations": {
                "dense_embeddings": stage01_dir is not None,
                "semantic_ids": False,
            },
            "limitations": [
                "Deterministic materialization replaces GRID online batch-window sampling; batches and optimizer updates differ",
                "Synthetic times preserve positional order only; no real calendar or user identity features may be inferred",
                "client_id is a history-segment key; original user identity must use grid_user_id",
                "Full mode retains repeated effective examples produced by truncating different raw windows",
                "Prefix mode uses the author split without the old adapter's tail-20-before-split protocol",
            ],
            "pending": [
                "frozen SID and representation coverage"
                if stage01_dir is not None
                else "frozen embeddings/SID and representation coverage",
                "framework config, materialization and resource manifests",
                "small framework preparation/training/evaluation end-to-end test",
            ],
        }
        (output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        logger.info(
            "GRID %s basis complete: mode=%s history=%d rows=%s eventhub=%s",
            category,
            augmentation,
            max_history,
            report["rows"],
            eventhub,
        )
