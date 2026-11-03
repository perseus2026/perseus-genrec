"""Backend-independent EventStore reader/writer contracts."""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Self

import polars as pl


class EventsWriter(AbstractContextManager["EventsWriter"], ABC):
    def __init__(self, path: Path, /) -> None:
        """Store the backend writer's destination path."""

        self.path = path

    def __enter__(self) -> Self:
        """Return the writer for use in a with block."""

        return self

    def write(self, events: pl.DataFrame, /) -> None:
        """Pass a nonempty event frame to the backend writer."""

        if events.is_empty():
            return
        self._write(events)

    @abstractmethod
    def _write(self, events: pl.DataFrame, /) -> None:
        """Write one nonempty event frame using the selected backend."""
        ...


class EventsReader(ABC):
    def __init__(
        self,
        path: Path,
        /,
        *,
        max_events_per_sequence: int | None,
    ) -> None:
        """Store the EventStore path and optional latest-event count limit."""

        self.path = path
        self.max_events_per_sequence = max_events_per_sequence

    def read(
        self,
        client_id: str,
        /,
        *,
        before: dt.date | dt.datetime | None = None,
    ) -> pl.DataFrame:
        """Read a client's events strictly before the cutoff and apply the count limit.

        Renumber positions so 1 is the newest event remaining after filtering.
        """

        events = self._read_events(client_id)
        if before is not None:
            events = (
                events.lazy()
                .filter(pl.col("timestamp") < before)
                .with_columns(
                    position=pl.col("position") - pl.col("position").min() + 1
                )
                .collect()
            )
        if self.max_events_per_sequence is None:
            return events
        return events.filter(pl.col("position") <= self.max_events_per_sequence)

    @abstractmethod
    def _read_events(self, client_id: str) -> pl.DataFrame:
        """Read all stored events for one client."""
        ...
