"""EventHub partition contract tests."""

# ruff: noqa: DTZ001 -- EventHub's timestamp contract is UTC-naive.

import datetime as dt
import logging

import polars as pl
import pytest

from data_sources.event_hub import (
    Reader,
    client_partition_id,
    distribute_to_partitions,
    partition_filename,
    timestamp_expression,
)
from genrecsys.core.config import Attribute, Config, Event, Input
from genrecsys.core.inputs import select_sequence


@pytest.mark.parametrize("unit", ["ms", "us", "ns"])
def test_timestamp_expression_preserves_utc_values_and_normalizes_dtype(unit):
    timestamp = dt.datetime(2026, 9, 28, 13, 45, tzinfo=dt.UTC)
    values = pl.DataFrame(
        {"timestamp": pl.Series([timestamp], dtype=pl.Datetime(unit, "UTC"))}
    ).select(timestamp_expression("timestamp"))
    assert values.schema["timestamp"] == pl.Datetime("ns")
    assert values["timestamp"].item() == timestamp.replace(tzinfo=None)


def test_reader_deduplicates_exact_events_not_repeated_items(tmp_path):
    first = dt.datetime(2026, 9, 28, 9)
    later = first + dt.timedelta(minutes=1)
    folder = tmp_path / "click" / "2026-09-28"
    folder.mkdir(parents=True)
    pl.DataFrame(
        {
            "client_id": ["u"] * 4,
            "timestamp": [first, first, later, later],
            "entity_id": ["entity"] * 4,
            "subtype": ["open", "open", "open", "view"],
        }
    ).write_parquet(folder / "0001.pq")
    config = Config(
        events={
            "click": Event(
                attributes={
                    "item": Attribute(field="entity_id"),
                    "interaction_subtype": Attribute(field="subtype"),
                }
            )
        },
        inputs={
            "history": Input(
                type="sequence", events=("click",), max_events_per_sequence=10
            )
        },
    )
    samples = pl.DataFrame(
        {
            "client_id": ["u", "u"],
            "_sample_timestamp": [later + dt.timedelta(hours=1)] * 2,
        }
    )
    events = Reader(config, source_to_path={"event_hub": tmp_path}).read(1, samples)
    assert len(events) == 3
    assert events["item"].to_list() == ["entity"] * 3
    assert set(events.select("timestamp", "interaction_subtype").rows()) == {
        (first, "open"),
        (later, "open"),
        (later, "view"),
    }


def test_flatten_keeps_chronological_order_without_resorting_positions(tmp_path):
    first = dt.datetime(2026, 9, 28, 9)
    later = first + dt.timedelta(minutes=1)
    values = pl.DataFrame(
        [
            ("b", later, "tap", "b-later"),
            ("a", first, "tap", "repeated"),
            ("a", later, "tap", "repeated"),
            ("b", first, "tap", "b-first"),
            ("a", first, "tap", "tie"),
            ("a", first, "tap", "repeated"),
        ],
        schema=["client_id", "timestamp", "event", "item"],
        orient="row",
    )
    reader = Reader(
        Config(events={"tap": Event(attributes={"item": Attribute(field="item")})}),
        source_to_path={"event_hub": tmp_path},
    )

    actual = reader._flatten_unique_events_by_client(values)

    assert len(actual) == 5
    assert actual["client_id"].to_list() == ["a", "a", "a", "b", "b"]
    assert len(actual.filter(pl.col("item") == "repeated")) == 2
    for _, history in actual.group_by("client_id", maintain_order=True):
        assert history["timestamp"].is_sorted()
        assert history["position"].to_list() == list(range(len(history), 0, -1))
    assert actual.equals(actual.sort("client_id", "position", descending=[False, True]))


@pytest.mark.parametrize("duration", [None, "10d"])
def test_scan_prunes_client_bounds_without_changing_sample_histories(
    tmp_path, duration
):
    cutoff = dt.datetime(2026, 8, 20)
    config = Config(
        max_duration_per_sequence=duration,
        events={
            name: Event(
                attributes={"item": Attribute(field="item_id")}, priority=priority
            )
            for name, priority in (("target", 1), ("aux", 0))
        },
        inputs={
            "short": Input(
                type="sequence", events=("target",), max_events_per_sequence=1
            ),
            "lifelong": Input(
                type="sequence",
                events=("target", "aux"),
                max_events_per_sequence=5,
                selection="priority",
                max_priority_events_per_sequence=2,
            ),
            "recent": Input(
                type="sequence",
                events=("target", "aux"),
                max_events_per_sequence=3,
                max_duration_per_sequence="2d",
            ),
        },
    )
    frames = []
    offsets = (-11, -10, -9, -1, 0, 0, 5, 6, 9, 10, 11, 12, 13)
    for name in config.events:
        frame = pl.DataFrame(
            [
                (client, cutoff + dt.timedelta(days=offset), f"{name}-{client}-{index}")
                for client in ("early", "late", "unused")
                for index, offset in enumerate(offsets)
            ],
            schema=["client_id", "timestamp", "item_id"],
            orient="row",
        ).with_columns(timestamp_expression("timestamp"))
        for (date,), part in frame.with_columns(
            date=pl.col("timestamp").dt.date()
        ).group_by("date", maintain_order=True):
            path = tmp_path / name / date.isoformat() / partition_filename(1)
            path.parent.mkdir(parents=True, exist_ok=True)
            part.drop("date").write_parquet(path)
        frames.append(
            frame.rename({"item_id": "item"}).with_columns(pl.lit(name).alias("event"))
        )
    samples = pl.DataFrame(
        {
            "client_id": ["early"] * 4 + ["late"] * 2,
            "_sample_timestamp": [
                cutoff + dt.timedelta(days=day) for day in (0, 0, 2, 6, 10, 12)
            ],
            "_fold": ["train", "train", "validation", "test", "train", "test"],
        }
    ).with_columns(timestamp_expression("_sample_timestamp"))
    reader = Reader(config, source_to_path={"event_hub": tmp_path})
    raw = pl.concat(frames).select("client_id", "timestamp", "event", "item")
    previous_scan = raw.lazy().filter(
        pl.col("timestamp") < samples["_sample_timestamp"].max()
    )
    if duration:
        previous_scan = previous_scan.filter(
            pl.col("timestamp") >= cutoff - dt.timedelta(days=10)
        )
    previous_scan = previous_scan.join(
        samples.lazy().select("client_id").unique(), on="client_id", how="semi"
    )

    scanned = reader._scan_events(1, samples).collect(engine="streaming")
    expected = raw.filter(
        (
            (pl.col("client_id") == "early")
            & (pl.col("timestamp") < cutoff + dt.timedelta(days=6))
        )
        | (
            (pl.col("client_id") == "late")
            & (pl.col("timestamp") < cutoff + dt.timedelta(days=12))
        )
    )
    if duration:
        expected = expected.filter(
            (
                (pl.col("client_id") == "early")
                & (pl.col("timestamp") >= cutoff - dt.timedelta(days=10))
            )
            | ((pl.col("client_id") == "late") & (pl.col("timestamp") >= cutoff))
        )
    assert scanned.sort("item").equals(expected.sort("item"))
    assert len(scanned) < previous_scan.select(pl.len()).collect().item()
    assert (
        scanned.filter(
            (pl.col("client_id") == "early")
            & (pl.col("timestamp") == cutoff - dt.timedelta(days=9))
        ).height
        == 2
    )
    order = ["client_id", "timestamp", "event", "item"]
    assert (
        reader._select_histories(scanned.lazy(), samples)
        .sort(order)
        .equals(reader._select_histories(previous_scan, samples).sort(order))
    )
    assert scanned.filter(
        (pl.col("client_id") == "early")
        & (pl.col("event") == "target")
        & (pl.col("timestamp") == cutoff)
    )["item"].to_list() == ["target-early-4", "target-early-5"]


def test_client_partition_id_matches_eventhub_hash_reference_values():
    num_partitions = 2**32

    assert client_partition_id("", num_partitions=num_partitions) == 1
    assert client_partition_id("foo", num_partitions=num_partitions) == 4_138_058_785
    assert client_partition_id("hello", num_partitions=num_partitions) == 613_153_352
    assert client_partition_id(b"foo", num_partitions=num_partitions) == 4_138_058_785


@pytest.mark.parametrize("seed", [0, 42, 0xFFFFFFFF])
def test_client_partition_id_preserves_previous_sklearn_hashes(seed):
    from sklearn.utils import murmurhash3_32

    clients = [
        "",
        "foo",
        "hello",
        "клиент-42",
        "用户-42",
        "reader-😀",
        b"foo",
        b"\x00\xff",
        123456789,
        *[f"client-{index:06d}" for index in range(1000)],
    ]
    for client_id in clients:
        value = client_id if isinstance(client_id, bytes) else str(client_id)
        previous_hash = murmurhash3_32(value, seed=seed, positive=True)
        assert (
            client_partition_id(client_id, num_partitions=2**32, seed=seed) - 1
            == previous_hash
        )
        assert client_partition_id(client_id, seed=seed) == previous_hash % 100 + 1


@pytest.mark.parametrize("lazy", [False, True])
def test_distribute_to_partitions_preserves_repeated_samples(lazy):
    samples = pl.DataFrame(
        {
            "client_id": ["client-a", "client-b", "client-a"],
            "sample_id": [0, 1, 2],
        }
    )
    partitioned = distribute_to_partitions(samples.lazy() if lazy else samples)
    if lazy:
        assert isinstance(partitioned, pl.LazyFrame)
        partitioned = partitioned.collect()
    assert partitioned.sort("sample_id").select(
        "client_id", "sample_id", "partition_id"
    ).rows() == [
        (client_id, sample_id, client_partition_id(client_id))
        for client_id, sample_id in samples.iter_rows()
    ]


def test_reader_normalises_scalar_and_list_attributes(tmp_path):
    event_hub_root = tmp_path / "event-hub"
    filename = partition_filename(1)
    timestamp = dt.datetime(2026, 1, 1, tzinfo=None)

    scalar_path = event_hub_root / "scalar-items" / "2026-01-01" / filename
    scalar_path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "client_id": ["client"],
            "timestamp": [timestamp],
            "item_label": ["scalar-item"],
        }
    ).write_parquet(scalar_path)

    list_path = event_hub_root / "list-items" / "2026-01-01" / filename
    list_path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "client_id": ["client"],
            "timestamp": [timestamp],
            "item_label": [["list-item-a", "list-item-b"]],
            "source_id": [[8, 9]],
        }
    ).write_parquet(list_path)

    attributes = {
        "item_label": Attribute(field="item_label"),
        "source_id": Attribute(field="source_id"),
    }
    config = Config(
        events={
            "scalar-items": Event(attributes=attributes),
            "list-items": Event(attributes=attributes),
        },
        inputs={
            "history": Input(
                type="sequence",
                events=("scalar-items", "list-items"),
                max_events_per_sequence=10,
            )
        },
    )
    events = Reader(
        config,
        source_to_path={"event_hub": event_hub_root},
    ).read(
        1,
        pl.DataFrame(
            {
                "client_id": ["client"],
                "_sample_timestamp": [timestamp + dt.timedelta(days=1)],
            }
        ),
    )

    assert events is not None
    assert events.schema["item_label"] == pl.String
    assert events.schema["source_id"] == pl.Int64
    assert set(events.select("item_label", "source_id").iter_rows()) == {
        ("scalar-item", None),
        ("list-item-a", 8),
        ("list-item-b", 9),
    }


def test_history_selection_matches_independent_windows(tmp_path, caplog):
    """Keep independent limits, repeated samples, cutoff exclusion and date pruning."""
    caplog.set_level(logging.INFO, logger="data_sources.event_hub")
    cutoff = dt.datetime(2026, 8, 10)
    rows = []
    config = Config(
        max_duration_per_sequence="10d",
        events={
            name: Event(attributes={"item": Attribute(field="entity_id")})
            for name in ("tap", "view")
        },
        inputs={
            "short": Input(
                type="sequence", events=("tap", "view"), max_events_per_sequence=2
            ),
            "positive": Input(
                type="sequence", events=("tap",), max_events_per_sequence=3
            ),
            "archive": Input(
                type="sequence",
                events=("tap", "view"),
                max_events_per_sequence=4,
                max_duration_per_sequence="2d",
            ),
        },
    )
    for event in config.events:
        for day in range(-13, 3):
            folder = (
                tmp_path / event / (cutoff + dt.timedelta(days=day)).date().isoformat()
            )
            folder.mkdir(parents=True)
            frame = pl.DataFrame(
                {
                    "client_id": ["a", "b", "unused"],
                    "timestamp": [
                        cutoff
                        + dt.timedelta(days=day, hours=1 if event == "view" else 0)
                    ]
                    * 3,
                    "entity_id": [f"{event}-{day}"] * 3,
                }
            )
            frame.write_parquet(folder / "0001.pq")
            rows.append(frame.with_columns(pl.lit(event).alias("event")))
    poison = tmp_path / "tap/2020-01-01/0001.pq"
    poison.parent.mkdir(parents=True)
    poison.write_text("Must not be scanned")
    samples = pl.DataFrame(
        {
            "client_id": ["a", "a", "a", "b", "empty"],
            "_sample_timestamp": [
                cutoff,
                cutoff + dt.timedelta(days=1),
                cutoff,
                cutoff,
                cutoff,
            ],
        }
    )
    raw = pl.concat(rows)
    expected = set()
    for client, end in samples.iter_rows():
        for model_input in config.inputs.values():
            days = 2 if model_input.max_duration_per_sequence else 10
            part = (
                raw.filter(
                    (pl.col("client_id") == client)
                    & pl.col("event").is_in(model_input.events)
                    & pl.col("timestamp").is_between(
                        end - dt.timedelta(days=days), end, closed="left"
                    )
                )
                .sort("timestamp")
                .tail(model_input.max_events_per_sequence)
            )
            expected.update(
                part.select("client_id", "timestamp", "event", "entity_id").iter_rows()
            )
    reader = Reader(
        config,
        source_to_path={"event_hub": tmp_path},
    )
    actual = reader.read(1, samples)
    assert (
        set(actual.select("client_id", "timestamp", "event", "item").iter_rows())
        == expected
    )
    assert (
        f"EventHub partition completed partition_id=1 events={len(actual)}"
        in caplog.text
    )
    assert "file listing" not in caplog.text
    assert "first batch" not in caplog.text


def test_history_materialization_prefers_event_priority_before_recency(tmp_path):
    cutoff = dt.datetime(2026, 9, 28, 12)
    for event, hours in (("target", -3), ("click", -2), ("view", -1)):
        folder = tmp_path / event / "2026-09-28"
        folder.mkdir(parents=True)
        pl.DataFrame(
            {
                "client_id": ["client"],
                "timestamp": [cutoff + dt.timedelta(hours=hours)],
                "item_id": [event],
            }
        ).write_parquet(folder / partition_filename(1))
    config = Config(
        events={
            "target": Event(
                attributes={"item": Attribute(field="item_id")}, priority=1
            ),
            "click": Event(attributes={"item": Attribute(field="item_id")}),
            "view": Event(attributes={"item": Attribute(field="item_id")}),
        },
        inputs={
            "history": Input(
                type="sequence",
                events=("target", "click", "view"),
                max_events_per_sequence=2,
                selection="priority",
            ),
            "targets": Input(
                type="sequence",
                events=("target",),
                max_events_per_sequence=1,
            ),
        },
    )

    reader = Reader(config, source_to_path={"event_hub": tmp_path})
    assert reader.histories == [
        {
            "events": ["click", "target", "view"],
            "max_events": 2,
            "max_duration": None,
            "selection": "priority",
        }
    ]
    values = reader.read(
        1,
        pl.DataFrame({"client_id": ["client"], "_sample_timestamp": [cutoff]}),
    )

    assert values.select("event", "item").rows() == [
        ("target", "target"),
        ("view", "view"),
    ]


def test_history_materialization_caps_priority_events(tmp_path):
    cutoff = dt.datetime(2026, 9, 28, 12)
    for event, hours in (
        ("aux", (-8, -6, -4, -2)),
        ("target", (-7, -5, -3, -1)),
    ):
        folder = tmp_path / event / "2026-09-28"
        folder.mkdir(parents=True)
        pl.DataFrame(
            {
                "client_id": ["client"] * len(hours),
                "timestamp": [cutoff + dt.timedelta(hours=value) for value in hours],
                "item_id": [f"{event}-{value}" for value in hours],
            }
        ).write_parquet(folder / partition_filename(1))
    config = Config(
        events={
            "target": Event(
                attributes={"item": Attribute(field="item_id")}, priority=1
            ),
            "aux": Event(attributes={"item": Attribute(field="item_id")}),
        },
        inputs={
            "lifelong": Input(
                type="sequence",
                events=("target", "aux"),
                max_events_per_sequence=5,
                selection="priority",
                max_priority_events_per_sequence=2,
            )
        },
    )

    reader = Reader(config, source_to_path={"event_hub": tmp_path})
    values = reader.read(
        1,
        pl.DataFrame({"client_id": ["client"], "_sample_timestamp": [cutoff]}),
    )

    assert values.select("event", "item").rows() == [
        ("aux", "aux--6"),
        ("aux", "aux--4"),
        ("target", "target--3"),
        ("aux", "aux--2"),
        ("target", "target--1"),
    ]


@pytest.mark.parametrize("extra_lower_priority_event", [False, True])
def test_materialization_keeps_recent_events_alongside_saturated_priority_history(
    tmp_path,
    extra_lower_priority_event,
):
    config = Config.load("tests/fixtures/tecd-small/retail-features.yaml")
    if extra_lower_priority_event:
        config.events["auxiliary"] = Event(priority=-1, source="tecd")
        config.inputs["lifelong"].events += ("auxiliary",)
    cutoff = dt.datetime(2026, 10, 20)
    limit = config.inputs["lifelong"].max_events_per_sequence
    positive = "t-ecd-small-retail-like"
    click = "t-ecd-small-retail-click"
    events = pl.DataFrame(
        {
            "client_id": ["u"] * (limit + 1),
            "timestamp": [
                cutoff - dt.timedelta(minutes=i) for i in range(limit + 1, 0, -1)
            ],
            "event": [positive] * limit + [click],
            "item": [str(i) for i in range(limit + 1)],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("ns")))
    samples = pl.DataFrame(
        {
            "client_id": ["u"],
            "_sample_timestamp": [cutoff],
        }
    ).with_columns(pl.col("_sample_timestamp").cast(pl.Datetime("ns")))
    reader = Reader(config, source_to_path={"tecd": tmp_path})
    retained = reader._select_histories(events.lazy(), samples)
    for name in ("short", "like", "lifelong"):
        expected = select_sequence(
            config,
            name,
            events.rename({"event": "_event"}),
            sample_timestamp=cutoff,
        )
        actual = select_sequence(
            config,
            name,
            retained.rename({"event": "_event"}),
            sample_timestamp=cutoff,
        )
        assert actual.equals(expected), name


@pytest.mark.parametrize(
    "priorities, expected",
    [((2, 1, 0), ["high", "low", "low"]), ((2, 2, 2), ["middle", "low", "low"])],
)
def test_capped_priority_materialization_fills_other_levels_by_recency(
    tmp_path,
    priorities,
    expected,
):
    cutoff = dt.datetime(2026, 10, 20)
    config = Config(
        events={
            name: Event(priority=priority)
            for name, priority in zip(("high", "middle", "low"), priorities)
        },
        inputs={
            "arbitrary_name": Input(
                type="sequence",
                events=("high", "middle", "low"),
                max_events_per_sequence=3,
                selection="priority",
                max_priority_events_per_sequence=1,
            )
        },
    )
    events = pl.DataFrame(
        {
            "client_id": ["u"] * 5,
            "timestamp": [cutoff - dt.timedelta(hours=i) for i in range(5, 0, -1)],
            "event": ["high", "middle", "middle", "low", "low"],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("ns")))
    samples = pl.DataFrame(
        {
            "client_id": ["u"],
            "_sample_timestamp": [cutoff],
        }
    ).with_columns(pl.col("_sample_timestamp").cast(pl.Datetime("ns")))
    reader = Reader(config, source_to_path={"event_hub": tmp_path})
    retained = reader._select_histories(events.lazy(), samples)
    assert retained["event"].to_list() == expected


@pytest.mark.parametrize(
    "error", [OSError("read failure"), pl.exceptions.ComputeError("bad schema")]
)
def test_reader_propagates_errors_without_retry(tmp_path, monkeypatch, caplog, error):
    caplog.set_level(logging.INFO, logger="data_sources.event_hub")
    reader = Reader(Config(), source_to_path={"event_hub": tmp_path})
    attempts = []

    def scan(partition, samples):
        attempts.append(partition)
        raise error

    monkeypatch.setattr(reader, "_scan_events", scan)
    samples = pl.DataFrame(
        {"client_id": ["a"], "_sample_timestamp": [dt.datetime(2026, 8, 10)]}
    )
    with pytest.raises(type(error), match=str(error)) as caught:
        reader.read(1, samples)
    assert caught.value is error
    assert len(attempts) == 1
    assert "EventHub partition completed" not in caplog.text


def test_absent_auxiliary_stream_keeps_nullable_attributes(tmp_path):
    timestamp = dt.datetime(2026, 8, 10)
    for name, field, partition in (("items", "item", 1), ("transactions", "brand", 2)):
        folder = tmp_path / name / "2026-08-10"
        folder.mkdir(parents=True)
        pl.DataFrame(
            {"client_id": ["u"], "timestamp": [timestamp], field: [7]}
        ).write_parquet(folder / partition_filename(partition))
    config = Config(
        events={
            name: Event(attributes={field: Attribute(field=field)})
            for name, field in (("items", "item"), ("transactions", "brand"))
        },
        inputs={
            "all": Input(
                type="sequence",
                events=("items", "transactions"),
                max_events_per_sequence=5,
            )
        },
    )
    values = Reader(config, source_to_path={"event_hub": tmp_path}).read(
        1,
        pl.DataFrame(
            {
                "client_id": ["u"],
                "_sample_timestamp": [timestamp + dt.timedelta(hours=1)],
            }
        ),
    )
    assert values["item"].to_list() == [7]
    assert values["brand"].to_list() == [None]
