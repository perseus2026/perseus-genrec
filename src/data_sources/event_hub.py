"""Shared EventHub reader and canonical client partitioning."""

from __future__ import annotations

import logging
import time
from functools import cached_property
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import polars as pl
from mmh3 import hash as murmur_hash

from genrecsys.core.config import Config, duration_days, materialization_histories
from genrecsys.core.storages import parse_from_env

PARTITIONS_NUM = 100
PARTITIONS_SEED = 0

Frame = TypeVar("Frame", pl.DataFrame, pl.LazyFrame)
logger = logging.getLogger(__name__)


def client_partition_id(
    client_id: Any,
    /,
    *,
    num_partitions: int = PARTITIONS_NUM,
    seed: int = PARTITIONS_SEED,
) -> int:
    """Assign a client to the one-based EventHub partition contract."""

    if num_partitions < 1:
        raise ValueError("num_partitions must be positive")
    if client_id is None:
        raise ValueError("cannot hash None")
    value = client_id if isinstance(client_id, bytes) else str(client_id)
    return murmur_hash(value, seed=seed & 0xFFFFFFFF, signed=False) % num_partitions + 1


def partition_filename(partition_id: int, /) -> str:
    """Return the canonical EventHub parquet filename for a partition."""

    return f"{partition_id:04d}.pq"


def partition_id_from_filename(value: str, /) -> int:
    """Parse a canonical EventHub partition filename."""

    path = Path(value)
    if path.suffix != ".pq" or not path.stem.isdigit():
        raise ValueError(f"invalid EventHub partition filename: {value!r}")
    return int(path.stem)


def distribute_to_partitions(samples: Frame, /) -> Frame:
    """Assign the canonical one-based MurmurHash EventHub partition."""

    return samples.join(
        samples.select("client_id")
        .unique()
        .with_columns(
            partition_id=pl.col("client_id").map_elements(
                lambda value: client_partition_id(value),
                return_dtype=pl.UInt32,
            )
        ),
        how="inner",
        on="client_id",
    )


def _effective_duration_days(config: Config, history: dict[str, Any]) -> int | None:
    """Return the shortest global/path history window, or no bound."""

    values = []
    for value, name in (
        (config.max_duration_per_sequence, "max_duration_per_sequence"),
        (history["max_duration"], "input.max_duration_per_sequence"),
    ):
        if value is not None:
            values.append(duration_days(value, field_name=name))
    return min(values) if values else None


def _minimal_materialization_histories(config: Config) -> list[dict[str, Any]]:
    """Drop a subset history guaranteed to be retained by a larger priority path."""

    histories = materialization_histories(config)
    result = []
    for candidate in histories:
        candidate_events = set(candidate["events"])
        candidate_priorities = [
            config.events[event].priority for event in candidate_events
        ]
        covered = False
        for cover in histories:
            cover_events = set(cover["events"])
            outside = cover_events - candidate_events
            if (
                candidate.get("max_priority_events") is not None
                or cover.get("max_priority_events") is not None
                or (
                    candidate.get("selection", "recent") == "recent"
                    and len(set(candidate_priorities)) > 1
                )
                or cover.get("selection", "recent") != "priority"
                or not outside
                or not candidate_events < cover_events
                or cover["max_events"] < candidate["max_events"]
                or _effective_duration_days(config, cover)
                != _effective_duration_days(config, candidate)
            ):
                continue
            outside_priorities = [config.events[event].priority for event in outside]
            if min(candidate_priorities) > max(outside_priorities):
                covered = True
                break
        if not covered:
            result.append(candidate)
    return result


class Reader:
    """Read config-declared events for one partition and sample frontier."""

    def __init__(
        self,
        config: Config,
        /,
        *,
        source_to_path: dict[str, Path] | None = None,
    ) -> None:
        """Resolve EventHub source paths from explicit mounts or environment settings.

        Raise RuntimeError when a declared source is not mounted.
        """

        self.source_to_path = source_to_path or parse_from_env()
        declared = {event.source for event in config.events.values()}
        if missing := declared - set(self.source_to_path):
            raise RuntimeError(f"need to connect sources: {', '.join(sorted(missing))}")
        self.config = config
        histories = materialization_histories(config)
        self.histories = _minimal_materialization_histories(config)
        if len(self.histories) < len(histories):
            logger.info(
                "removed %s covered EventHub materialization histories",
                len(histories) - len(self.histories),
            )

    def read(
        self,
        partition_id: int,
        samples: pl.DataFrame,
        /,
    ) -> pl.DataFrame | None:
        """Read bounded histories strictly before each sample timestamp.

        Return unique events for one partition, or None if no history is found.
        """

        started = time.monotonic()
        samples = samples.with_columns(
            _sample_timestamp=timestamp_expression("_sample_timestamp")
        )
        events = self._scan_events(partition_id, samples)
        result = self._select_histories(events, samples) if events is not None else None
        if result is not None:
            result = self._flatten_unique_events_by_client(result)
        logger.info(
            "EventHub partition completed partition_id=%s events=%s elapsed_sec=%.1f",
            partition_id,
            len(result) if result is not None else 0,
            time.monotonic() - started,
        )
        return result

    def _scan_events(
        self,
        partition_id: int,
        samples: pl.DataFrame,
    ) -> pl.LazyFrame | None:
        """Scan EventHub files within each client's combined sample time bounds."""

        expressions: list[pl.LazyFrame] = []
        partition_file = partition_filename(partition_id)
        if samples.is_empty():
            return None
        upper = samples["_sample_timestamp"].max()
        duration = self.config.max_duration_per_sequence
        lower = (
            samples.select(
                pl.col("_sample_timestamp").min().dt.offset_by(f"-{duration}")
            ).item()
            if duration
            else None
        )
        event_files = {}
        for name, event in self.config.events.items():
            event_root = self.source_to_path[event.source] / name
            if not event_root.exists():
                logger.warning("not found EventHub event directory %s", event_root)
                continue
            files = [
                candidate
                for date_dir in sorted(event_root.iterdir())
                if (lower is None or date_dir.name >= lower.date().isoformat())
                and date_dir.name <= upper.date().isoformat()
                if (candidate := date_dir / partition_file).exists()
            ]
            if not files:
                logger.warning(
                    "not found %s events for partition_id=%s",
                    name,
                    partition_id,
                )
                continue
            event_files[name] = files
        attribute_expressions = self._attribute_expressions(event_files)
        for name, files in event_files.items():
            attributes, list_attributes = attribute_expressions[name]
            expression = pl.scan_parquet(files, extra_columns="ignore").select(
                "client_id",
                timestamp_expression("timestamp"),
                **attributes,
            )
            if list_attributes:
                expression = expression.explode(*list_attributes)
            expressions.append(expression)
        if not expressions:
            return None
        events = pl.concat(expressions).filter(pl.col("timestamp") < upper)
        if lower is not None:
            events = events.filter(pl.col("timestamp") >= lower)
        client_bounds = (
            samples.lazy()
            .group_by("client_id")
            .agg(
                _history_upper=pl.col("_sample_timestamp").max(),
                _history_lower=pl.col("_sample_timestamp").min(),
            )
        )
        if duration:
            client_bounds = client_bounds.with_columns(
                pl.col("_history_lower").dt.offset_by(f"-{duration}")
            )
        events = events.join(
            client_bounds,
            on="client_id",
            how="inner",
            maintain_order="left",
        ).filter(pl.col("timestamp") < pl.col("_history_upper"))
        if duration:
            events = events.filter(pl.col("timestamp") >= pl.col("_history_lower"))
        return events.drop("_history_upper", "_history_lower")

    def _select_histories(
        self,
        scanned: pl.LazyFrame,
        samples: pl.DataFrame,
    ) -> pl.DataFrame | None:
        """Select the union of bounded histories without an events-by-samples join."""

        if not self.histories:
            raise ValueError("config contains no sequence inputs")

        event_codes = {name: index for index, name in enumerate(self.config.events)}
        priorities = {
            name: event.priority for name, event in self.config.events.items()
        }
        events = (
            scanned.with_columns(
                pl.col("event")
                .replace_strict(event_codes, return_dtype=pl.Int64)
                .alias("_event_code"),
                pl.col("event")
                .replace_strict(priorities, return_dtype=pl.Int64)
                .alias("_event_priority"),
            )
            .sort("client_id", "timestamp", maintain_order=True)
            .with_row_index("_event_row")
            .collect(engine="streaming")
        )
        if events.is_empty():
            return None

        unique_samples = (
            samples.select("client_id", "_sample_timestamp")
            .unique()
            .sort("client_id", "_sample_timestamp")
            .with_row_index("_sample_row")
        )
        sample_ranges = {
            client: (int(start), int(length))
            for client, start, length in unique_samples.group_by(
                "client_id", maintain_order=True
            )
            .agg(
                pl.col("_sample_row").first().alias("_start"),
                pl.len().alias("_length"),
            )
            .iter_rows()
        }
        event_ranges = events.group_by("client_id", maintain_order=True).agg(
            pl.col("_event_row").first().alias("_start"),
            pl.len().alias("_length"),
        )
        event_times = events["timestamp"].to_numpy()
        event_code_values = events["_event_code"].to_numpy()
        event_priority_values = events["_event_priority"].to_numpy()
        sample_times = unique_samples["_sample_timestamp"].to_numpy()
        selected = np.zeros(len(events), dtype=np.bool_)

        plans = []
        for history in self.histories:
            codes = np.asarray(
                [event_codes[event] for event in history["events"]],
                dtype=event_code_values.dtype,
            )
            history_priorities = sorted(
                {priorities[event] for event in history["events"]}, reverse=True
            )
            if history.get("selection", "recent") == "recent":
                priority_groups = [history_priorities]
            elif history.get("max_priority_events") is not None:
                priority_groups = [history_priorities[:1], history_priorities[1:]]
            else:
                priority_groups = [[priority] for priority in history_priorities]
            plans.append(
                (
                    codes,
                    priority_groups,
                    len(history_priorities) == 1,
                    int(history["max_events"]),
                    history.get("max_priority_events")
                    if len(history_priorities) > 1
                    else None,
                    _effective_duration_days(self.config, history),
                )
            )

        for client, event_start, event_length in event_ranges.iter_rows():
            sample_range = sample_ranges.get(client)
            if sample_range is None:
                continue
            event_start = int(event_start)
            event_stop = event_start + int(event_length)
            sample_start, sample_length = sample_range
            client_sample_times = sample_times[
                sample_start : sample_start + sample_length
            ]
            client_codes = event_code_values[event_start:event_stop]
            client_priorities = event_priority_values[event_start:event_stop]
            client_event_times = event_times[event_start:event_stop]

            for (
                codes,
                priority_groups,
                preserve_first_tie,
                limit,
                priority_limit,
                duration,
            ) in plans:
                history_mask = np.isin(client_codes, codes)
                remaining = np.full(len(client_sample_times), limit, dtype=np.int64)
                lower_times = (
                    client_sample_times - np.timedelta64(duration, "D")
                    if duration is not None
                    else None
                )
                for priority_index, priority_group in enumerate(priority_groups):
                    positions = np.flatnonzero(
                        history_mask & np.isin(client_priorities, priority_group)
                    )
                    if not len(positions):
                        continue
                    times = client_event_times[positions]
                    if preserve_first_tie and len(positions) > 1:
                        order = np.lexsort((-positions, times.astype(np.int64)))
                        positions = positions[order]
                        times = times[order]
                    stops = np.searchsorted(times, client_sample_times, side="left")
                    starts = (
                        np.searchsorted(times, lower_times, side="left")
                        if lower_times is not None
                        else np.zeros_like(stops)
                    )
                    take = np.minimum(remaining, stops - starts)
                    if priority_index == 0 and priority_limit is not None:
                        take = np.minimum(take, int(priority_limit))
                    interval_starts = stops - take
                    active = take > 0
                    if active.any():
                        # Retain the complete boundary timestamp group. The
                        # sample router decides the final cap after stable
                        # deduplication; pruning a tie here can change its input.
                        interval_starts[active] = np.searchsorted(
                            times, times[interval_starts[active]], side="left"
                        )
                        differences = np.zeros(len(positions) + 1, dtype=np.int64)
                        np.add.at(differences, interval_starts[active], 1)
                        np.add.at(differences, stops[active], -1)
                        used = np.cumsum(differences[:-1]) > 0
                        selected[event_start + positions[used]] = True
                    remaining -= take
                    if not remaining.any():
                        break

        if not selected.any():
            return None
        return events.filter(pl.Series(selected)).drop(
            "_event_row", "_event_code", "_event_priority"
        )

    def _attribute_expressions(
        self,
        event_files: dict[str, list[Path]],
    ) -> dict[str, tuple[dict[str, pl.Expr], tuple[str, ...]]]:
        """Build typed attribute expressions and identify list columns to explode.

        Fill missing stream attributes with nulls; reject conflicting or absent schemas.
        """

        event_to_schema = {}
        for name, files in event_files.items():
            event_to_schema[name] = pl.scan_parquet(files).collect_schema()
        attribute_to_dtype: dict[str, pl.DataType] = {}
        attribute_to_origin: dict[str, tuple[str, pl.DataType]] = {}
        for event_name, event in self.config.events.items():
            if event_name not in event_to_schema:
                continue
            for attribute_name, attribute in event.attributes.items():
                if attribute_name == "event":
                    continue
                assert attribute is not None and attribute.field is not None
                if attribute.field not in event_to_schema[event_name]:
                    continue
                source_dtype = event_to_schema[event_name][attribute.field]
                dtype = (
                    pl.String
                    if attribute_name == "brand_id"
                    else _scalar_dtype(source_dtype)
                )
                previous = attribute_to_dtype.setdefault(attribute_name, dtype)
                if dtype != previous:
                    previous_event, previous_dtype = attribute_to_origin[attribute_name]
                    raise RuntimeError(
                        f"mismatched scalar dtypes for attribute {attribute_name!r}: "
                        f"{previous_event}={previous_dtype}, "
                        f"{event_name}={source_dtype}"
                    )
                attribute_to_origin.setdefault(
                    attribute_name,
                    (event_name, source_dtype),
                )
        declared_attributes = {
            attribute_name
            for event_name, event in self.config.events.items()
            if event_name in event_to_schema
            for attribute_name in event.attributes
            if attribute_name != "event"
        }
        if missing := declared_attributes - set(attribute_to_dtype):
            raise RuntimeError(
                "EventHub attributes are absent from every available schema: "
                f"{sorted(missing)}"
            )
        # An optional stream may be absent in this partition. Preserve nullable
        # columns so generic encoders can still produce a consistent schema.
        for attribute in self._attribute_names:
            if attribute != "event":
                attribute_to_dtype.setdefault(attribute, pl.Null)
        result: dict[str, tuple[dict[str, pl.Expr], tuple[str, ...]]] = {}
        for event_name, event in self.config.events.items():
            if event_name not in event_to_schema:
                continue
            attributes = {"event": pl.lit(event_name)}
            list_attributes: list[str] = []
            for attribute_name, dtype in attribute_to_dtype.items():
                attribute = event.attributes.get(attribute_name)
                if (
                    attribute is None
                    or attribute.field is None
                    or attribute.field not in event_to_schema[event_name]
                ):
                    attributes[attribute_name] = pl.lit(None, dtype=dtype)
                else:
                    source_dtype = event_to_schema[event_name][attribute.field]
                    if isinstance(source_dtype, pl.List):
                        attributes[attribute_name] = pl.col(attribute.field).cast(
                            pl.List(dtype),
                            strict=False,
                        )
                        list_attributes.append(attribute_name)
                    else:
                        attributes[attribute_name] = pl.col(attribute.field).cast(
                            dtype,
                            strict=False,
                        )
            result[event_name] = attributes, tuple(list_attributes)
        return result

    @cached_property
    def _attribute_names(self) -> tuple[str, ...]:
        """Return the sorted union of attributes declared by all event streams."""

        result = {"event"}
        for event in self.config.events.values():
            result.update(event.attributes)
        return tuple(sorted(result))

    def _flatten_unique_events_by_client(
        self,
        values: pl.DataFrame,
    ) -> pl.DataFrame:
        """Deduplicate exact events and number each client's history by recency.

        Position 1 is the newest event; rows are ordered oldest to newest.
        Repeated items at different times or with different attributes are retained.
        """

        columns = ["client_id", "timestamp", *self._attribute_names]
        return (
            values.lazy()
            .select(columns)
            .unique(maintain_order=True)
            .sort("client_id", "timestamp", maintain_order=True)
            .with_columns(position=pl.int_range(pl.len(), 0, -1).over("client_id"))
            .collect()
        )


def timestamp_expression(column: str) -> pl.Expr:
    """Return UTC-naive Datetime(ns) for basis/EventHub point-in-time joins.

    Input wall-clock values must already be in UTC. This removes timezone
    metadata without converting the stored clock values.
    """

    return (
        pl.col(column).dt.replace_time_zone(None).cast(pl.Datetime("ns")).alias(column)
    )


def _scalar_dtype(dtype: pl.DataType) -> pl.DataType:
    """Return the occurrence dtype for scalar or list-valued EventHub fields."""

    return dtype.inner if isinstance(dtype, pl.List) else dtype


__all__ = [
    "PARTITIONS_NUM",
    "PARTITIONS_SEED",
    "Reader",
    "client_partition_id",
    "distribute_to_partitions",
    "partition_filename",
    "partition_id_from_filename",
    "timestamp_expression",
]
