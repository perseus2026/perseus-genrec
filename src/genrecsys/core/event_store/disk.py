"""Arrow record-batch EventStore backend used during training."""

from __future__ import annotations

import io
import shutil
import types
from functools import cached_property
from pathlib import Path

import polars as pl
import pyarrow as pa

from genrecsys.core.event_store import base


class EventsWriter(base.EventsWriter):
    COMPRESSION = "zstd"

    def __init__(self, path: Path, /) -> None:
        """Initialize the Arrow writer and client-to-batch index."""

        super().__init__(path)
        self._writer: pa.ipc.RecordBatchFileWriter | None = None
        self._index: list[str] = []

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> bool | None:
        """Close the Arrow writer and publish index.pq; remove a partial store on failure."""

        if self._writer is not None:
            self._writer.close()
        if exc_type is None:
            if not self._index:
                raise ValueError("cannot persist an empty EventStore")
            pl.Series("client_id", self._index).to_frame().write_parquet(
                self.path / "index.pq"
            )
        elif self.path.exists():
            shutil.rmtree(self.path)
        return None

    def _write(self, events: pl.DataFrame, /) -> None:
        """Write one Arrow batch per client and update the client index."""

        if self._writer is None:
            self.path.mkdir(parents=True, exist_ok=True)
            schema = events.head(1).to_arrow().schema
            self._writer = pa.ipc.new_file(
                self.path / "data.arrow",
                schema,
                options=pa.ipc.IpcWriteOptions(compression=pa.Codec(self.COMPRESSION)),
            )
        assert self._writer is not None
        for (client_id,), client_events in events.group_by(
            "client_id", maintain_order=True
        ):
            copied = pl.DataFrame.deserialize(io.BytesIO(client_events.serialize()))
            self._writer.write_table(copied.to_arrow().combine_chunks())
            self._index.append(str(client_id))


class EventsReader(base.EventsReader):
    def _read_events(self, client_id: str) -> pl.DataFrame:
        """Read a client's Arrow batch; raise KeyError for an unknown client."""

        index = self._index.get(str(client_id))
        if index is None:
            raise KeyError(f"unknown client_id={client_id!r}")
        if self._locations is None:
            return pl.from_arrow(self._data.get_record_batch(index))
        file, batch = self._locations.row(index)
        if file not in self._files:
            self._files[file] = pa.ipc.open_file(self.path / file)
        return pl.from_arrow(self._files[file].get_record_batch(batch))

    @cached_property
    def _index(self) -> dict[str, int]:
        """Load and cache client-to-batch offsets from index.pq."""

        return dict(
            pl.scan_parquet(self.path / "index.pq", row_index_name="index")
            .select("client_id", "index")
            .collect()
            .iter_rows()
        )

    @cached_property
    def _locations(self) -> pl.DataFrame | None:
        """Keep shard locations in compact columns, not Python tuples per client."""
        path = self.path / "index.pq"
        if "file" not in pl.read_parquet_schema(path):
            return None
        return pl.read_parquet(path, columns=["file", "batch_index"])

    @cached_property
    def _data(self) -> pa.ipc.RecordBatchFileReader:
        """Open the original single-file store used by sequential prepare."""
        return pa.ipc.open_file(self.path / "data.arrow")

    @cached_property
    def _files(self) -> dict[str, pa.ipc.RecordBatchFileReader]:
        """Lazily open only Arrow shards accessed by this reader."""

        return {}
