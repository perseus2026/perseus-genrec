"""Concrete config-driven routing shared by multi-path model implementations."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import polars as pl

from genrecsys.core.config import Config, duration_days


def select_sequence(
    config: Config,
    name: str,
    events: pl.DataFrame,
    /,
    *,
    sample_timestamp: Any,
) -> pl.DataFrame:
    """Select a sequence branch and keep its latest events within configured windows.

    Apply global and branch-specific durations before the count limit. The caller
    must already have excluded events at or after the sample timestamp.
    """

    description = config.input(name)
    if description.type != "sequence":
        raise ValueError(f"input {name!r} is not a sequence")
    if events.is_empty():
        return events
    values = events.filter(pl.col("_event").is_in(description.events))
    durations = (
        (
            config.max_duration_per_sequence,
            "max_duration_per_sequence",
        ),
        (
            description.max_duration_per_sequence,
            f"inputs.{name}.max_duration_per_sequence",
        ),
    )
    for duration, field_name in durations:
        if duration is None:
            continue
        start = sample_timestamp - timedelta(
            days=duration_days(
                duration,
                field_name=field_name,
            )
        )
        values = values.filter(pl.col("timestamp") >= start)
    maximum = description.max_events_per_sequence
    if maximum is None:
        raise ValueError(f"sequence input {name!r} has no event limit")
    if description.selection == "recent":
        return values.sort("timestamp", maintain_order=True).tail(maximum)
    priorities = {event: config.events[event].priority for event in description.events}
    priority_maximum = description.max_priority_events_per_sequence
    if priority_maximum is not None and len(set(priorities.values())) > 1:
        values = values.with_columns(
            pl.col("_event")
            .replace_strict(priorities, return_dtype=pl.Int64)
            .alias("_event_priority")
        )
        highest = max(priorities.values())
        preferred = (
            values.filter(pl.col("_event_priority") == highest)
            .sort("timestamp")
            .tail(priority_maximum)
        )
        remaining = maximum - len(preferred)
        ordinary = (
            values.filter(pl.col("_event_priority") != highest)
            .sort("timestamp")
            .tail(remaining)
        )
        return (
            pl.concat([preferred, ordinary])
            .sort("timestamp", maintain_order=True)
            .drop("_event_priority")
        )
    if len(set(priorities.values())) == 1:
        return values.sort("timestamp").tail(maximum)
    return (
        values.with_columns(
            pl.col("_event")
            .replace_strict(priorities, return_dtype=pl.Int64)
            .alias("_event_priority")
        )
        .sort("_event_priority", "timestamp", maintain_order=True)
        .tail(maximum)
        .sort("timestamp", maintain_order=True)
        .drop("_event_priority")
    )


__all__ = ["select_sequence"]
