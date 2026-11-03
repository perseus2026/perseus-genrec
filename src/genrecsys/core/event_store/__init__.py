"""Per-client storage for prepared point-in-time events."""

from __future__ import annotations

import shutil
import tempfile
import weakref
from pathlib import Path
from typing import Literal

from genrecsys.core.event_store import disk, ram
from genrecsys.core.event_store.base import EventsReader, EventsWriter


class EventStore:
    def __init__(
        self,
        path: Path | None = None,
        /,
        *,
        backend: Literal["disk", "ram"],
    ) -> None:
        """Open an existing EventStore or create a temporary one with automatic cleanup.

        The disk backend uses Arrow batches; the RAM backend uses Parquet.
        """

        self.backend = backend
        if path is None:
            self.path = Path(tempfile.mkdtemp(prefix="genrecsys-events-"))
            self._finalizer: weakref.finalize | None = weakref.finalize(
                self,
                shutil.rmtree,
                self.path,
                ignore_errors=True,
            )
        else:
            self.path = path
            self._finalizer = None

    def open_writer(self) -> EventsWriter:
        """Create a context-managed writer for encoded events."""

        return self._backend_module.EventsWriter(self.path)

    def open_reader(self, max_events_per_sequence: int | None, /) -> EventsReader:
        """Create a per-client reader with an optional history count limit."""

        return self._backend_module.EventsReader(
            self.path,
            max_events_per_sequence=max_events_per_sequence,
        )

    @property
    def _backend_module(self):
        """Return the disk or RAM backend module, rejecting unknown backends."""

        if self.backend == "disk":
            return disk
        if self.backend == "ram":
            return ram
        raise ValueError(f"unknown EventStore backend {self.backend!r}")

    def move(self, path: Path, /) -> None:
        """Move the store to its destination and disable temporary-directory cleanup.

        Replace matching destination files and remove the source store.
        """

        if self.path.resolve() == path.resolve():
            return
        path.mkdir(parents=True, exist_ok=True)
        for source in sorted(self.path.iterdir()):
            destination = path / source.name
            if destination.is_file():
                destination.unlink()
            elif destination.is_dir():
                shutil.rmtree(destination)
            shutil.move(source, destination)
        if self.path != path:
            shutil.rmtree(self.path)
        self.path = path
        if self._finalizer is not None:
            self._finalizer.detach()
            self._finalizer = None


__all__ = ["EventStore", "EventsReader", "EventsWriter"]
