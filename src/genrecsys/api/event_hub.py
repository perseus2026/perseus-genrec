"""Manage a local EventHub using the Perseus Parquet storage layout."""

from __future__ import annotations

import datetime as dt
import logging
import shutil
from pathlib import Path

import polars as pl

from data_sources.event_hub import distribute_to_partitions, partition_filename
from genrecsys.core.storages import parse_from_env

logger = logging.getLogger(__name__)


def _source_path(source: str) -> Path:
    try:
        return parse_from_env()[source]
    except KeyError as error:
        raise ValueError(f"unknown EventHub source: {source!r}") from error


def _event_path(source: str, name: str) -> Path:
    root = _source_path(source)
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("event name must be a single directory name")
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("event directory must remain inside its source")
    return path


def add_events(events: pl.DataFrame, name: str, source: str = "event_hub") -> None:
    """Write events by date and client partition, replacing touched files.

    Require String client_id and UTC-naive Datetime(ns) timestamp, without nulls.
    As in Perseus, this is replacement per date/partition, not append or dedup.
    The date and partition_id columns are reserved for storage partitioning.
    """
    path = _event_path(source, name)
    if (
        events.schema.get("client_id") != pl.String
        or events["client_id"].is_null().any()
    ):
        raise ValueError("client_id must have dtype String and contain no nulls")
    if (
        events.schema.get("timestamp") != pl.Datetime("ns")
        or events["timestamp"].is_null().any()
    ):
        raise ValueError(
            "timestamp must have dtype Datetime(ns), UTC-naive, and contain no nulls"
        )
    if reserved := {"date", "partition_id"} & set(events.columns):
        raise ValueError(f"reserved EventHub columns: {sorted(reserved)}")
    partitioned = (
        distribute_to_partitions(events.lazy())
        .sort("timestamp", "client_id", maintain_order=True)
        .with_columns(date=pl.col("timestamp").dt.date())
        .collect()
    )
    for (date, partition), frame in partitioned.partition_by(
        ["date", "partition_id"], as_dict=True, maintain_order=True
    ).items():
        destination = path / date.isoformat()
        destination.mkdir(parents=True, exist_ok=True)
        frame.drop("partition_id").write_parquet(
            destination / partition_filename(partition)
        )
    logger.info(
        "saved %d %s events for %d clients",
        len(events),
        name,
        events["client_id"].n_unique(),
    )


def list_events(source: str = "event_hub") -> None:
    """Log event names, date coverage and attributes without reading event rows."""
    root = _source_path(source)
    if not root.exists():
        logger.info("no events in source %s", source)
        return
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        files = sorted(path.glob("*/*.pq"))
        if not files:
            continue
        dates = sorted({file.parent.name for file in files})
        schema = pl.scan_parquet(files).collect_schema()
        attributes = ", ".join(
            name for name in schema if name not in {"client_id", "timestamp", "date"}
        )
        logger.info(
            "%s events available for %d days from %s to %s with attributes: %s",
            path.name,
            len(dates),
            dates[0],
            dates[-1],
            attributes,
        )


def delete_events(
    name: str,
    source: str = "event_hub",
    date_from: dt.date | None = None,
    date_to: dt.date | None = None,
) -> None:
    """Delete an event type or its date directories in an inclusive date range."""
    path = _event_path(source, name)
    if date_from is not None and date_to is not None and date_from > date_to:
        raise ValueError("date_from must not be later than date_to")
    if not path.exists():
        logger.warning("%s events do not exist", name)
        return
    if date_from is None and date_to is None:
        shutil.rmtree(path)
        logger.info("deleted all %s events", name)
        return
    for directory in sorted(path.iterdir()):
        if not directory.is_dir():
            continue
        date = dt.date.fromisoformat(directory.name)
        if (date_from is None or date >= date_from) and (
            date_to is None or date <= date_to
        ):
            shutil.rmtree(directory)
            logger.info("deleted %s events for %s", name, date)
