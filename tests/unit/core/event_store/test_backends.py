"""EventStore backend parity required by the Perseus-style lifecycle."""

from datetime import datetime
from typing import Literal

import polars as pl
import pytest

from genrecsys.core.event_store import EventStore


@pytest.mark.parametrize("backend", ["disk", "ram"])
def test_event_store_backends_preserve_point_in_time_reads(
    tmp_path,
    backend: Literal["disk", "ram"],
) -> None:
    store = EventStore(tmp_path / backend, backend=backend)
    events = pl.DataFrame(
        {
            "client_id": ["u1", "u1", "u2"],
            "timestamp": [
                datetime(2026, 1, 1),
                datetime(2026, 1, 2),
                datetime(2026, 1, 1),
            ],
            "position": [2, 1, 1],
            "event": [1, 2, 1],
        }
    )
    with store.open_writer() as writer:
        writer.write(events)

    reader = EventStore(store.path, backend=backend).open_reader(10)
    result = reader.read("u1", before=datetime(2026, 1, 2))

    assert result["event"].to_list() == [1]
    assert result["position"].to_list() == [1]


def test_event_store_rejects_unknown_backend(tmp_path) -> None:
    store = EventStore(tmp_path / "events", backend="ram")
    store.backend = "unknown"

    with pytest.raises(ValueError, match="unknown EventStore backend"):
        store.open_reader(10)
