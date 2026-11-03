"""Parquet-backed EventStore optimized for repeated inference reads."""

from __future__ import annotations

import shutil
import types
from functools import cached_property
from pathlib import Path

import polars as pl

from genrecsys.core.event_store import base


class EventsWriter(base.EventsWriter):
    def __init__(self, path: Path, /) -> None:
        """Initialize an in-memory buffer of event frames."""

        super().__init__(path)
        self._parts: list[pl.DataFrame] = []

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> bool | None:
        """Write buffered events and their index; remove a partial store on failure."""

        if exc_type is not None:
            if self.path.exists():
                shutil.rmtree(self.path)
            return None
        if not self._parts:
            raise ValueError("cannot persist an empty EventStore")
        data = pl.concat(self._parts, rechunk=False)
        index = (
            data.lazy()
            .select("client_id")
            .with_row_index()
            .group_by("client_id", maintain_order=True)
            .agg(
                from_index=pl.col("index").min(),
                to_index=pl.col("index").max() + 1,
            )
            .collect()
        )
        self.path.mkdir(parents=True, exist_ok=True)
        data.write_parquet(self.path / "data.pq")
        index.write_parquet(self.path / "index.pq")
        return None

    def _write(self, events: pl.DataFrame, /) -> None:
        """Append an event frame to the in-memory write buffer."""

        self._parts.append(events)


class EventsReader(base.EventsReader):
    def _read_events(self, client_id: str) -> pl.DataFrame:
        """Return a client's event slice from the cached DataFrame."""

        index = self._index.get(str(client_id))
        if index is None:
            raise KeyError(f"unknown client_id={client_id!r}")
        start, stop = index
        return self._data[start:stop]

    @cached_property
    def _index(self) -> dict[str, tuple[int, int]]:
        """Load and cache client-to-row offsets from index.pq."""

        return {
            str(client_id): (int(start), int(stop))
            for client_id, start, stop in pl.read_parquet(
                self.path / "index.pq"
            ).iter_rows()
        }

    @cached_property
    def _data(self) -> pl.DataFrame:
        """Load and cache the complete data.pq in RAM."""

        return pl.read_parquet(self.path / "data.pq")
