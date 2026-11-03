"""High-level EventHub operations preserve the existing Perseus file contract."""

# EventHub timestamps are UTC-naive.

import datetime as dt
import logging

import polars as pl
import pytest
from mmh3 import hash as murmur_hash

from data_sources.event_hub import Reader, client_partition_id
from genrecsys import api
from genrecsys.cli.event_hub import app
from genrecsys.core.config import Attribute, Config, Event, Input


@pytest.fixture
def storage(tmp_path, monkeypatch):
    root = tmp_path / "hub"
    monkeypatch.setenv("INTERNAL_STORAGE_EVENT_HUB", str(root))
    monkeypatch.setenv("internal_storage_custom", str(tmp_path / "custom"))
    return root


def events():
    return pl.DataFrame(
        {
            "client_id": ["u1", "u1", "u2", "u1"],
            "timestamp": pl.Series(
                [dt.datetime(2026, 1, day, 12) for day in (1, 2, 2, 3)],
                dtype=pl.Datetime("ns"),
            ),
            "item_id": [10, 20, 30, 40],
            "tags": [["a", "b"], [], None, ["c"]],
        }
    )


def test_import_layout_rows_and_attributes(storage):
    frame = events()
    api.add_events(frame, "view")
    actual = pl.read_parquet(storage / "view/*/*.pq").select(frame.columns)
    assert actual.sort("item_id").equals(frame.sort("item_id"))
    expected = {
        f"view/{row['timestamp'].date()}/{murmur_hash(row['client_id'], seed=0, signed=False) % 100 + 1:04d}.pq"
        for row in frame.iter_rows(named=True)
    }
    assert {str(p.relative_to(storage)) for p in storage.rglob("*.pq")} == expected
    assert "partition_id" not in pl.read_parquet_schema(next(storage.rglob("*.pq")))


def test_import_is_replacement_per_date_and_partition(storage):
    frame = events()
    api.add_events(frame, "view")
    paths = {p: p.read_bytes() for p in storage.rglob("*.pq")}
    replacement = frame.head(1).with_columns(item_id=pl.lit(100, dtype=pl.Int64))
    api.add_events(replacement, "view")
    actual = pl.read_parquet(storage / "view/*/*.pq")
    assert sorted(actual["item_id"]) == [20, 30, 40, 100]
    for path, before in paths.items():
        if path.parent.name != "2026-01-01":
            assert path.read_bytes() == before
    api.add_events(replacement, "view")
    assert len(pl.read_parquet(storage / "view/*/*.pq")) == 4
    api.add_events(pl.concat([replacement, replacement]), "view")
    assert (
        len(pl.read_parquet(storage / "view/*/*.pq")) == 5
    )  # No hidden deduplication.


def test_existing_reader_selects_strict_history(storage):
    api.add_events(events(), "view")
    config = Config(
        events={"view": Event(attributes={"item": Attribute(field="item_id")})},
        inputs={
            "history": Input(
                type="sequence", events=("view",), max_events_per_sequence=20
            )
        },
    )
    reader = Reader(config)
    samples = pl.DataFrame(
        {
            "client_id": ["u1"],
            "_sample_timestamp": pl.Series(
                [dt.datetime(2026, 1, 2, 12)], dtype=pl.Datetime("ns")
            ),
        }
    )
    history = reader.read(client_partition_id("u1"), samples)
    assert history["item"].to_list() == [10]


@pytest.mark.parametrize(
    "change",
    [
        "no-client",
        "numeric-client",
        "null-client",
        "no-time",
        "null-time",
        "microseconds",
        "timezone",
        "reserved",
    ],
)
def test_invalid_input_rejected_before_writing(storage, change):
    frame = events()
    match change:
        case "no-client":
            frame = frame.drop("client_id")
        case "numeric-client":
            frame = frame.with_columns(client_id=pl.lit(3))
        case "null-client":
            frame = frame.with_columns(client_id=pl.lit(None, dtype=pl.String))
        case "no-time":
            frame = frame.drop("timestamp")
        case "null-time":
            frame = frame.with_columns(timestamp=pl.lit(None, dtype=pl.Datetime("ns")))
        case "microseconds":
            frame = frame.with_columns(pl.col("timestamp").dt.cast_time_unit("us"))
        case "timezone":
            frame = frame.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))
        case "reserved":
            frame = frame.with_columns(partition_id=pl.lit(1))
    with pytest.raises(ValueError):
        api.add_events(frame, "view")
    assert not storage.exists()


def test_empty_import_and_listing(storage, caplog):
    caplog.set_level(logging.INFO)
    api.add_events(events().head(0), "empty")
    assert not storage.exists()
    api.list_events()
    api.add_events(events(), "view")
    api.list_events()
    assert (
        "view events available for 3 days from 2026-01-01 to 2026-01-03" in caplog.text
    )
    assert "item_id, tags" in caplog.text
    api.delete_events(
        "view", date_from=dt.date(2026, 1, 1), date_to=dt.date(2026, 1, 3)
    )
    api.list_events()  # Empty type directory is valid after date-range deletion.


@pytest.mark.parametrize(
    ("start", "stop", "remaining"),
    [
        (None, None, []),
        (dt.date(2026, 1, 2), None, ["2026-01-01"]),
        (None, dt.date(2026, 1, 2), ["2026-01-03"]),
        (dt.date(2026, 1, 2), dt.date(2026, 1, 2), ["2026-01-01", "2026-01-03"]),
    ],
)
def test_delete_scope(storage, start, stop, remaining):
    api.add_events(events(), "view")
    api.add_events(events(), "purchase")
    other = {p: p.read_bytes() for p in (storage / "purchase").rglob("*.pq")}
    api.delete_events("view", date_from=start, date_to=stop)
    assert sorted(p.name for p in (storage / "view").glob("*")) == remaining
    assert all(p.read_bytes() == data for p, data in other.items())
    api.delete_events("absent")


@pytest.mark.parametrize(
    "name", ["", ".", "..", "../outside", "/absolute", "a/b", "a\\b"]
)
def test_names_cannot_escape_source(storage, name):
    with pytest.raises(ValueError, match="directory name"):
        api.add_events(events(), name)
    with pytest.raises(ValueError, match="directory name"):
        api.delete_events(name)


def test_external_symlink_and_reversed_dates_are_rejected(storage, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    storage.mkdir()
    (storage / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="inside"):
        api.delete_events("escape")
    assert outside.exists()
    api.add_events(events(), "view")
    with pytest.raises(ValueError, match="date_from"):
        api.delete_events(
            "view", date_from=dt.date(2026, 2, 1), date_to=dt.date(2026, 1, 1)
        )
    assert len(list((storage / "view").glob("*/*.pq"))) > 0


def test_cli_custom_source_and_date_parsing(storage, tmp_path):
    path = tmp_path / "events.pq"
    events().write_parquet(path)
    for tokens in (
        ["add-events", str(path), "--name=view", "--source=custom"],
        ["list-events", "--source=custom"],
        [
            "delete-events",
            "--name=view",
            "--source=custom",
            "--date-from=2026-01-02",
            "--date-to=2026-01-02",
        ],
    ):
        function, bound, _ = app.parse_args(tokens, exit_on_error=False)
        function(*bound.args, **bound.kwargs)
    assert not storage.exists()
    assert sorted(p.name for p in (tmp_path / "custom/view").iterdir()) == [
        "2026-01-01",
        "2026-01-03",
    ]
    with pytest.raises(ValueError, match="unknown"):
        api.list_events("unknown-source")
