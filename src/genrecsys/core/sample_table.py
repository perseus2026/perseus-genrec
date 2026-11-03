"""Random-access sample rows backed by an uncompressed, memory-mapped IPC file."""

from __future__ import annotations

import logging
from functools import cached_property
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class MappedSamples:
    """Share OS-backed buffers across ranks and reopen them in spawned workers."""

    def __init__(self, path: Path) -> None:
        """Store the IPC cache path; open Arrow buffers on first access."""

        self.path = path

    @classmethod
    def from_parquet(cls, path: Path) -> MappedSamples:
        # The CLI serializes cache creation with local_main_process_first. A
        # unique temporary file also makes concurrent standalone readers safe.
        """Create or reuse an uncompressed Arrow cache for Parquet samples.

        Key the cache by source size/mtime and publish it atomically. The cache requires
        additional disk space and can be shared by concurrent readers.
        """

        stat = path.stat()
        cache = path.parent / ".samples-cache"
        cache.mkdir(exist_ok=True)
        destination = cache / f"{path.stem}-{stat.st_size}-{stat.st_mtime_ns}.arrow"
        if not destination.exists():
            logger.info("Building memory-mapped sample cache: %s", path)
            with NamedTemporaryFile(dir=cache, suffix=".arrow") as temporary:
                source = pq.ParquetFile(path)
                with (
                    pa.OSFile(temporary.name, "wb") as sink,
                    pa.ipc.new_file(sink, source.schema_arrow) as writer,
                ):
                    for batch in source.iter_batches(batch_size=8192):
                        writer.write_batch(batch)
                # Keep the temporary name until its context manager removes it.
                # A hard link publishes the completed file without copying it.
                try:
                    destination.hardlink_to(temporary.name)
                except FileExistsError:
                    pass
            logger.info("Sample cache ready: %s", destination)
        return cls(destination)

    @cached_property
    def _table(self) -> pa.Table:
        """Open the Arrow table through memory-mapped IPC buffers."""

        return pa.ipc.open_file(pa.memory_map(str(self.path), "r")).read_all()

    def __len__(self) -> int:
        """Return the number of mapped sample rows."""

        return self._table.num_rows

    def row(self, index: int, *, named: bool = True) -> dict[str, Any]:
        """Return one sample row as a dict, checking the row index."""

        if not named:
            raise ValueError("MappedSamples only supports named rows")
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self._table.slice(index, 1).to_pylist()[0]

    def column(self, name: str) -> list[Any]:
        """Read one logical column without materialising every sample as a dict."""

        if name not in self._table.column_names:
            raise KeyError(name)
        return self._table.column(name).to_pylist()

    def __getstate__(self) -> dict[str, Any]:
        # Never pickle the mapped table: spawn would duplicate the full payload.
        """Serialize only the cache path so spawned workers reopen shared buffers."""

        return {"path": self.path}
