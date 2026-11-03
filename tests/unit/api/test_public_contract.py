"""Public Python API contract tests for the Synthetic lifecycle."""

import logging
from importlib import import_module
from threading import Event
from types import SimpleNamespace

import polars as pl
import pytest

from genrecsys import api, core
from genrecsys.cli import inference


def test_event_partitions_are_read_sequentially_without_prefetch() -> None:
    preparation = import_module("genrecsys.core.preparation")
    samples = pl.DataFrame({"client_id": ["u1"]})
    calls = []

    def read(partition, partition_samples):
        assert partition_samples is samples
        calls.append(partition)
        return pl.DataFrame({"partition": [partition]})

    partitions = ((1, 2, 7, samples), (2, 2, 8, samples))
    results = preparation.read_event_partitions(
        SimpleNamespace(read=read),
        partitions,
    )
    assert calls == []
    partition, _, events = next(results)
    assert partition == events.item() == 7
    assert calls == [7]
    partition, _, events = next(results)
    assert partition == events.item() == 8
    assert calls == [7, 8]
    assert list(results) == []


def test_event_partition_wait_logs_progress_and_propagates_error(
    monkeypatch, caplog
) -> None:
    preparation = import_module("genrecsys.core.preparation")
    monkeypatch.setattr(preparation, "PARTITION_PROGRESS_INTERVAL_SECONDS", 0.01)
    observed = Event()

    class ProgressHandler(logging.Handler):
        def emit(self, record):
            if "waiting for EventHub partition" in record.getMessage():
                observed.set()

    def read(*_):
        assert observed.wait(timeout=2)
        raise TimeoutError("source read failed")

    handler = ProgressHandler()
    preparation.logger.addHandler(handler)
    try:
        with (
            caplog.at_level(logging.INFO),
            pytest.raises(TimeoutError, match="source read failed"),
        ):
            list(
                preparation.read_event_partitions(
                    SimpleNamespace(read=read),
                    [(1, 1, 7, pl.DataFrame({"client_id": ["u1"]}))],
                )
            )
    finally:
        preparation.logger.removeHandler(handler)
    assert "completed=0/1" in caplog.text
    assert "eta=unknown (no completed partitions)" in caplog.text


def test_retrieval_catalog_does_not_require_internal_item_id() -> None:
    artifacts = core.tasks.retrieval.Artifacts(pl.DataFrame({"item_id": [10, 20]}))

    assert artifacts.items.columns == ["item_id"]


@pytest.mark.parametrize("split_catalog", [False, True])
def test_inference_factory_passes_artifacts_to_session(monkeypatch, split_catalog):
    prediction_api = import_module("genrecsys.api.make_predictions")
    checkpoint = SimpleNamespace(
        prepared=SimpleNamespace(
            config=SimpleNamespace(
                model=SimpleNamespace(type="test"), validate=lambda: None
            )
        )
    )
    artifacts = (
        core.tasks.retrieval.Artifacts(pl.DataFrame({"item_id": [10]}))
        if split_catalog
        else None
    )
    session = object()
    calls = []

    def create_session(value, *, artifacts):
        calls.append((value, artifacts))
        return session

    def get_implementation(name):
        assert name == "test"
        return SimpleNamespace(create_inference_session=create_session)

    monkeypatch.setattr(prediction_api, "get", get_implementation)
    assert api.create_inference_session(checkpoint, artifacts=artifacts) is session
    assert len(calls) == 1
    assert calls[0][0] is checkpoint
    assert calls[0][1] is artifacts


@pytest.mark.parametrize("batch_count", [0, 2])
def test_public_make_predictions_collects_session_batches(monkeypatch, batch_count):
    prediction_api = import_module("genrecsys.api.make_predictions")
    samples = pl.DataFrame({"client_id": ["u1", "u2"]})
    checkpoint, event_store = object(), object()
    frames = [
        samples.slice(index, 1).with_columns(
            prediction=pl.lit([{"item": 10 + index, "score": -1.0}])
        )
        for index in range(batch_count)
    ]

    def iter_predictions(partition_id, values, state, **kwargs):
        assert partition_id == 1
        assert values is samples
        assert state is checkpoint
        assert kwargs == {"event_store": event_store, "samples_prepared": True}
        yield from frames

    monkeypatch.setattr(prediction_api, "iter_predictions", iter_predictions)
    result = api.make_predictions(
        1, samples, checkpoint, event_store=event_store, samples_prepared=True
    )
    if batch_count:
        assert result.equals(pl.concat(frames))
    else:
        assert result is None


@pytest.mark.parametrize(
    ("history_source", "relative_path", "client_id"),
    [
        ("event_hub", "basis/test/samples.pq", "raw"),
        ("event_store", "dataset/test/samples.pq", "prepared"),
    ],
)
def test_distribute_samples_uses_history_compatible_sample_state(
    tmp_path,
    history_source: str,
    relative_path: str,
    client_id: str,
) -> None:
    source = tmp_path / relative_path
    source.parent.mkdir(parents=True)
    pl.DataFrame({"client_id": [client_id]}).write_parquet(source)

    inference.distribute_samples(
        history_source=history_source,
        workdir=tmp_path,
    )

    [partition] = (tmp_path / "samples").glob("*.pq")
    assert pl.read_parquet(partition)["client_id"].to_list() == [client_id]


@pytest.mark.parametrize(
    ("rank", "world_size", "expected_partition_ids"),
    [(0, 1, [1, 2]), (0, 2, [1]), (1, 2, [2])],
)
def test_partitioned_inference_reuses_session_and_streams_outputs(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    rank,
    world_size,
    expected_partition_ids,
) -> None:
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    (samples_dir / "logs").mkdir()
    (samples_dir / "logs" / "stage.log").write_text("distribution completed\n")
    (samples_dir / "metadata.json").write_text("{}")
    (samples_dir / "shadow.pq").mkdir()
    (tmp_path / "dataset" / "events").mkdir(parents=True)
    test_artifacts_path = tmp_path / "dataset" / "test" / "artifacts"
    core.tasks.retrieval.Artifacts(
        pl.DataFrame({"item_id": [10], "item_key": [1]})
    ).save(test_artifacts_path)
    for partition_id in (1, 2):
        pl.DataFrame(
            {
                "sample_id": [partition_id],
                "client_id": [f"u{partition_id}"],
            }
        ).write_parquet(samples_dir / f"{partition_id:04d}.pq")

    checkpoint = SimpleNamespace(
        config=SimpleNamespace(task=SimpleNamespace(type="retrieval"))
    )
    session = object()
    created_sessions = []
    used_sessions = []

    monkeypatch.setattr(
        core.checkpoint.Trained,
        "load",
        classmethod(lambda cls, path: checkpoint),
    )

    def create_inference_session(value, *, artifacts):
        assert value is checkpoint
        assert artifacts.items["item_id"].to_list() == [10]
        created_sessions.append(value)
        return session

    def iter_predictions(
        partition_id,
        samples,
        value,
        *,
        event_store,
        samples_prepared,
        inference_session,
    ):
        assert value is checkpoint
        assert event_store is not None
        assert samples_prepared is True
        used_sessions.append(inference_session)
        yield samples.with_columns(
            prediction=pl.lit([{"item": partition_id, "score": -1.0}])
        )

    monkeypatch.setattr(api, "create_inference_session", create_inference_session)
    monkeypatch.setattr(api, "iter_predictions", iter_predictions)

    inference.make_predictions(
        history_source="event_store",
        prediction_write_batch_size=1,
        workdir=tmp_path,
    )

    assert created_sessions == [checkpoint]
    assert used_sessions == [session] * len(expected_partition_ids)
    outputs = sorted((tmp_path / "predictions").glob("*.pq"))
    assert [path.name for path in outputs] == [
        f"{partition_id:04d}.pq" for partition_id in expected_partition_ids
    ]
    assert [pl.read_parquet(path).height for path in outputs] == [1] * len(outputs)
