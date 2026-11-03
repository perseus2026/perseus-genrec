"""Full MD-to-SD preprocessing keeps only target users and both payment streams."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tarfile
from datetime import timedelta
from pathlib import Path
from threading import Barrier, Event, Lock

import polars as pl
import pytest

from model_datasets.tecd import payments as preprocessing


def test_full_payments_history_filters_users_and_keeps_receipts(tmp_path: Path) -> None:
    samples = tmp_path / "samples"
    for index, users in enumerate((["1"], ["3"])):
        destination = samples / str(index) / "basis/train"
        destination.mkdir(parents=True)
        pl.DataFrame(
            {
                "client_id": users,
                "timestamp": [timedelta(days=2)] * len(users),
                "target": [["item"]] * len(users),
            },
            schema_overrides={"timestamp": pl.Duration("us")},
        ).write_parquet(destination / "samples.pq")

    source = tmp_path / "source/dataset/full/payments"
    (source / "events").mkdir(parents=True)
    (source / "receipts").mkdir()
    pl.DataFrame(
        {
            "timestamp": [timedelta(days=1), timedelta(days=1, minutes=1)],
            "user_id": [1, 2],
            "brand_id": [10, 20],
        },
        schema_overrides={"timestamp": pl.Duration("us")},
    ).write_parquet(source / "events/00001.pq")
    pl.DataFrame(
        {
            "timestamp": [timedelta(days=1, minutes=2), timedelta(days=1, minutes=3)],
            "user_id": [3, 4],
            "brand_id": [30, 40],
            "approximate_item_id": ["market_1", "market_2"],
        },
        schema_overrides={"timestamp": pl.Duration("us")},
    ).write_parquet(source / "receipts/00001.pq")

    output = tmp_path / "output"
    preprocessing.build_full_payments_history(
        source_dir=tmp_path / "source",
        samples_root=samples,
        output_dir=output,
        source_revision="test",
        start_day=1,
        end_day=1,
    )

    report = json.loads((output / "report.json").read_text())
    assert report["target_users"] == 2
    assert report["rows"] == {"payment": 1, "payment_receipt": 1}
    assert report["unique_brands"] == {
        "payment": 1,
        "payment_receipt": 1,
        "union": 2,
    }
    assert report["item_identity_used"] is False

    unpacked = tmp_path / "unpacked"
    with tarfile.open(output / "eventhub.tar") as archive:
        archive.extractall(unpacked, filter="data")
    event_names = {path.name for path in (unpacked / "eventhub").iterdir()}
    assert event_names == {
        preprocessing.PAYMENT_EVENT,
        preprocessing.RECEIPT_EVENT,
    }
    rows = pl.concat(
        [pl.read_parquet(path) for path in (unpacked / "eventhub").rglob("*.pq")]
    )
    assert set(rows["client_id"]) == {"1", "3"}
    assert "approximate_item_id" not in rows.columns


def _raw_payment_fixture(root: Path, *, days: int = 3) -> tuple[Path, Path]:
    samples = root / "samples/basis/train"
    samples.mkdir(parents=True)
    pl.DataFrame({"client_id": ["1"]}).write_parquet(samples / "samples.pq")
    source = root / "source"
    for kind in ("events", "receipts"):
        directory = source / "dataset/full/payments" / kind
        directory.mkdir(parents=True)
        for day in range(1, days + 1):
            pl.DataFrame(
                {
                    "timestamp": [timedelta(days=day)] * 3,
                    "user_id": [1, 2, 1],
                    "brand_id": [day, 100, day],
                },
                schema_overrides={"timestamp": pl.Duration("us")},
            ).write_parquet(directory / f"{day:05d}.pq")
    return source, root / "samples"


def _archive_frames(path: Path) -> dict[str, pl.DataFrame]:
    with tarfile.open(path) as archive:
        return {
            entry.name: pl.read_parquet(archive.extractfile(entry))
            for entry in archive.getmembers()
            if entry.name.endswith(".pq")
        }


@pytest.mark.parametrize("operation", ["_read_payment_file", "_write_events"])
def test_failed_payment_consumption_keeps_raw_input(tmp_path, monkeypatch, operation):
    source, samples = _raw_payment_fixture(tmp_path, days=1)
    path = source / "dataset/full/payments/events/00001.pq"

    def fail(*args, **kwargs):
        raise RuntimeError("payment consumption failed")

    monkeypatch.setattr(preprocessing, operation, fail)
    with pytest.raises(RuntimeError, match="payment consumption failed"):
        preprocessing._build_full_payments_history(
            source=lambda kind, day: (
                source / f"dataset/full/payments/{kind}/{day:05d}.pq"
            ),
            samples_root=samples,
            output_dir=tmp_path / "output",
            source_revision="test",
            start_day=1,
            end_day=1,
            file_workers=2,
            delete_source_files=True,
        )
    assert path.is_file()


@pytest.mark.parametrize("workers", [2, 4])
def test_parallel_payments_are_bounded_ordered_and_match_sequential(
    tmp_path: Path, monkeypatch, caplog, workers
) -> None:
    source, samples = _raw_payment_fixture(tmp_path, days=4)
    sequential = tmp_path / "sequential"
    preprocessing.build_full_payments_history(
        source_dir=source,
        samples_root=samples,
        output_dir=sequential,
        start_day=1,
        end_day=4,
    )
    read = preprocessing._read_payment_file
    write = preprocessing._write_events
    barrier, lock = Barrier(workers), Lock()
    started = []
    writes = []
    active = peak = 0

    def concurrent_read(kind, day, **kwargs):
        nonlocal active, peak
        with lock:
            started.append((kind, day))
            active += 1
            peak = max(active, peak)
        try:
            barrier.wait(timeout=5)
            return read(kind, day, **kwargs)
        finally:
            with lock:
                active -= 1

    def ordered_write(root, event, values):
        with lock:
            # The next file is submitted only after the current file is written.
            assert len(started) <= len(writes) + workers
            writes.append((event, values["timestamp"][0].day))
        write(root, event, values)

    monkeypatch.setattr(preprocessing, "_read_payment_file", concurrent_read)
    monkeypatch.setattr(preprocessing, "_write_events", ordered_write)
    parallel = tmp_path / "parallel"
    with caplog.at_level(logging.INFO):
        preprocessing.build_full_payments_history(
            source_dir=source,
            samples_root=samples,
            output_dir=parallel,
            start_day=1,
            end_day=4,
            file_workers=workers,
        )
    assert peak == workers
    assert writes == [
        (event, day + 1)
        for day in range(1, 5)
        for event in (preprocessing.PAYMENT_EVENT, preprocessing.RECEIPT_EVENT)
    ]
    expected = _archive_frames(sequential / "eventhub.tar")
    actual = _archive_frames(parallel / "eventhub.tar")
    assert actual.keys() == expected.keys()
    assert all(actual[name].equals(expected[name]) for name in expected)
    report = json.loads((parallel / "report.json").read_text())
    reference = json.loads((sequential / "report.json").read_text())
    assert report["rows"] == reference["rows"] == {"payment": 4, "payment_receipt": 4}
    assert report["unique_brands"] == reference["unique_brands"]
    assert report["file_workers"] == workers
    assert report["source_files"] == 8
    assert report["source_bytes"] > 0
    assert len(list(source.rglob("*.pq"))) == 8  # Local sources must not be deleted.
    assert f"workers={workers}" in caplog.text
    assert "throughput=" in caplog.text
    assert "eta=" in caplog.text


def test_phase_heartbeat_and_failure_are_logged(monkeypatch, caplog) -> None:
    monkeypatch.setattr(preprocessing, "PROGRESS_INTERVAL_SECONDS", 0.01)
    observed = Event()

    class HeartbeatHandler(logging.Handler):
        def emit(self, record):
            if "in progress" in record.getMessage():
                observed.set()

    handler = HeartbeatHandler()
    preprocessing.logger.addHandler(handler)
    try:
        with caplog.at_level(logging.INFO):
            with preprocessing._log_phase("slow read"):
                assert observed.wait(timeout=2)
            with (
                pytest.raises(ValueError, match="failure"),
                preprocessing._log_phase("failed read"),
            ):
                raise ValueError("failure")
    finally:
        preprocessing.logger.removeHandler(handler)
    assert "Starting slow read" in caplog.text
    assert "slow read in progress elapsed=" in caplog.text
    assert "Completed slow read" in caplog.text
    assert "Stopped failed read" in caplog.text
    assert "Completed failed read" not in caplog.text


def test_payments_wait_reports_counts_without_inventing_eta(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    source, samples = _raw_payment_fixture(tmp_path, days=1)
    read = preprocessing._read_payment_file
    observed = Event()
    monkeypatch.setattr(preprocessing, "PROGRESS_INTERVAL_SECONDS", 0.01)

    class ProgressHandler(logging.Handler):
        def emit(self, record):
            if record.getMessage().startswith("Full Payments progress"):
                observed.set()

    def slow_read(kind, day, **kwargs):
        if kind == "events":
            assert observed.wait(timeout=2)
        return read(kind, day, **kwargs)

    handler = ProgressHandler()
    preprocessing.logger.addHandler(handler)
    monkeypatch.setattr(preprocessing, "_read_payment_file", slow_read)
    try:
        with caplog.at_level(logging.INFO):
            preprocessing.build_full_payments_history(
                source_dir=source,
                samples_root=samples,
                output_dir=tmp_path / "history",
                start_day=1,
                end_day=1,
                file_workers=2,
            )
    finally:
        preprocessing.logger.removeHandler(handler)
    assert "Full Payments progress: files=0/2 workers=2" in caplog.text
    assert "eta=unknown (no completed files)" in caplog.text


def test_payments_cli_enables_info_and_extracts_archives(tmp_path: Path) -> None:
    source, samples = _raw_payment_fixture(tmp_path, days=1)
    output = tmp_path / "history"
    env = {**os.environ, "PYTHONPATH": "src"}
    module = [sys.executable, "-m", "model_datasets.tecd.payments"]
    build = subprocess.run(
        [
            *module,
            "build-full-payments-history",
            f"--source-dir={source}",
            f"--samples-root={samples}",
            f"--output-dir={output}",
            "--start-day=1",
            "--end-day=1",
            "--file-workers=2",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "INFO" in build.stderr
    assert "Starting Full Payments files" in build.stderr
    assert "Completed Full Payments files" in build.stderr
    extract = subprocess.run(
        [
            *module,
            "extract-histories",
            f"--history-dir={output}",
            f"--workdir={tmp_path / 'unpacked'}",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Starting Extract history eventhub.tar" in extract.stderr
    assert "Completed Extract history eventhub.tar" in extract.stderr
    assert len(list((tmp_path / "unpacked/eventhub").rglob("*.pq"))) == 2
    assert not (output / "eventhub.tar").exists()


def test_downloaded_raw_files_are_deleted_after_write(tmp_path: Path) -> None:
    source, samples = _raw_payment_fixture(tmp_path, days=1)
    preprocessing._build_full_payments_history(
        samples_root=samples,
        output_dir=tmp_path / "history",
        source_revision="fixture",
        start_day=1,
        end_day=1,
        source=lambda kind, day: source / f"dataset/full/payments/{kind}/{day:05d}.pq",
        delete_source_files=True,
        file_workers=2,
    )
    assert not list(source.rglob("*.pq"))
