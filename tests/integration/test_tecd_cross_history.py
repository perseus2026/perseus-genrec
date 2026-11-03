"""Test the repository-owned T-ECD auxiliary history builder."""

import datetime as dt
import json
import logging
from pathlib import Path

import polars as pl
import pytest
from requests.exceptions import ChunkedEncodingError

from model_datasets.tecd import build_cross_history, cross_history, extract_eventhub
from model_datasets.tecd.cross_history import (
    PAYMENT_WORKERS,
    _payment_ranges,
)


def test_cross_history_builds_offers_and_payments_without_perseus(
    tmp_path: Path, caplog
):
    source = tmp_path / "source"
    offers = source / "dataset/small/offers"
    payments = source / "dataset/full/payments/events"
    (offers / "events").mkdir(parents=True)
    payments.mkdir(parents=True)
    timestamp = dt.timedelta(days=20_000, hours=12)
    pl.DataFrame(
        {
            "action_type": ["click", "view"],
            "timestamp": [timestamp, timestamp],
            "user_id": [1, 1],
            "item_id": [10, 10],
        }
    ).write_parquet(offers / "events/part.pq")
    pl.DataFrame({"item_id": [10], "brand_id": [100]}).write_parquet(
        offers / "items.pq"
    )
    users = source / "dataset/small/users.pq"
    users.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"user_id": [1]}).write_parquet(users)
    pl.DataFrame({"brand_id": [100]}).write_parquet(
        source / "dataset/small/brands.pq"
    )
    pl.DataFrame(
        {
            "timestamp": [timestamp],
            "user_id": [1],
            "brand_id": [100],
            "price": [42.0],
        }
    ).write_parquet(payments / "00002.pq")

    output = tmp_path / "output"
    with caplog.at_level(logging.INFO):
        build_cross_history(
            source_dir=source,
            output_dir=output,
            source_revision="fixed-revision",
            payment_start_day=2,
            payment_end_day=2,
        )
    assert "Processed local payment batch 1/1" in caplog.text
    assert (output / "eventhub.tar").is_file()
    extract_eventhub(basis=output)
    event_names = {
        path.parts[-3] for path in (output / "eventhub").rglob("*.pq")
    }
    assert event_names == {"t-ecd-small-offers-click", "t-ecd-payments"}
    values = pl.concat(
        [pl.read_parquet(path) for path in (output / "eventhub").rglob("*.pq")],
        how="diagonal_relaxed",
    )
    assert values.schema["brand_id"] == pl.String
    payment = pl.concat(
        [
            pl.read_parquet(path)
            for path in (output / "eventhub/t-ecd-payments").rglob("*.pq")
        ]
    )
    assert payment["price"].to_list() == [42.0]
    report = json.loads((output / "report.json").read_text())
    assert report["source_revision"] == "fixed-revision"
    assert report["rows"] == {"offers": 1, "payments": 1}


def test_payment_ranges_cover_every_day_once() -> None:
    ranges = _payment_ranges(1308, 715, PAYMENT_WORKERS)
    assert len(ranges) == 8
    days = [
        day
        for start_day, end_day in ranges
        for day in range(start_day, end_day - 1, -1)
    ]
    assert days == list(range(1308, 714, -1))


def test_payment_workers_use_eight_processes_and_flush_logs(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    calls = []

    class Process:
        def __init__(self, command, env):
            self.command = command
            self.env = env
            self.returncode = 0
            calls.append(self)
            output = Path(
                next(value.split("=", 1)[1] for value in command if value.startswith("--output-dir="))
            )
            output.mkdir(parents=True)
            (output / "report.json").write_text('{"rows": 1}\n')

        def poll(self):
            return self.returncode

    monkeypatch.setattr(cross_history.subprocess, "Popen", Process)
    monkeypatch.setenv("POLARS_MAX_THREADS", "32")
    with caplog.at_level(logging.INFO):
        outputs, rows = cross_history._run_payment_workers(
            source_dir=tmp_path,
            revision="fixed-revision",
            start_day=1308,
            end_day=715,
        )

    assert len(calls) == PAYMENT_WORKERS == 8
    assert len(outputs) == 8
    assert rows == 8
    assert all(process.env["POLARS_MAX_THREADS"] == "4" for process in calls)
    assert all(process.env["PYTHONUNBUFFERED"] == "1" for process in calls)
    assert "Payment workers progress: completed=8/8" in caplog.text


@pytest.mark.parametrize(
    ("failures", "error_type"),
    [
        (0, ChunkedEncodingError),
        (2, ChunkedEncodingError),
        (cross_history.PAYMENT_DOWNLOAD_ATTEMPTS, ChunkedEncodingError),
        (1, ValueError),
    ],
)
def test_payment_download_retries_only_interrupted_streams(
    tmp_path: Path, monkeypatch, caplog, failures: int, error_type: type[Exception]
) -> None:
    import huggingface_hub

    source = tmp_path / "source"
    metadata = source / "dataset/small"
    metadata.mkdir(parents=True)
    pl.DataFrame({"user_id": [1]}).write_parquet(metadata / "users.pq")
    pl.DataFrame({"brand_id": [100]}).write_parquet(metadata / "brands.pq")
    output = tmp_path / "worker"
    filename = "dataset/full/payments/events/00002.pq"
    partial = output / "downloads/.cache/huggingface/download/payment.incomplete"
    error = error_type("Connection broken: IncompleteRead")
    calls, sleeps = [], []

    def download(**kwargs):
        calls.append(kwargs)
        if len(calls) > 1:
            assert partial.read_bytes() == b"partial download"
        if len(calls) <= failures:
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_bytes(b"partial download")
            raise error
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "timestamp": [dt.timedelta(days=20_000)],
                "user_id": [1],
                "brand_id": [100],
                "price": [42.0],
            }
        ).write_parquet(path)
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(cross_history.time, "sleep", sleeps.append)
    arguments = {
        "source_dir": source,
        "output_dir": output,
        "revision": "fixed-revision",
        "start_day": 2,
        "end_day": 2,
    }
    should_fail = (
        failures == cross_history.PAYMENT_DOWNLOAD_ATTEMPTS or error_type is ValueError
    )
    with caplog.at_level(logging.INFO):
        if should_fail:
            with pytest.raises(error_type) as caught:
                cross_history.download_and_build_payment_range(**arguments)
            assert caught.value is error
            assert not (output / "report.json").exists()
            assert not list((output / "eventhub").rglob("*.pq"))
        else:
            cross_history.download_and_build_payment_range(**arguments)
            report = json.loads((output / "report.json").read_text())
            assert report["rows"] == 1
            payments = pl.concat(
                [pl.read_parquet(path) for path in (output / "eventhub").rglob("*.pq")]
            )
            assert payments["price"].to_list() == [42.0]
            assert not (output / "downloads" / filename).exists()
            assert caplog.text.count("processed file 1/1") == 1
    expected_calls = failures if should_fail else failures + 1
    assert len(calls) == expected_calls
    assert all(
        call
        == {
            "repo_id": cross_history.DATASET_REPOSITORY,
            "repo_type": "dataset",
            "revision": "fixed-revision",
            "filename": filename,
            "local_dir": output / "downloads",
        }
        for call in calls
    )
    assert sleeps == [cross_history.PAYMENT_DOWNLOAD_RETRY_DELAY_SECONDS] * (
        expected_calls - 1
    )


@pytest.mark.parametrize(
    ("failures", "error_type"),
    [
        (0, ChunkedEncodingError),
        (2, ChunkedEncodingError),
        (cross_history.SOURCE_DOWNLOAD_ATTEMPTS, ChunkedEncodingError),
        (1, ValueError),
    ],
)
def test_cross_source_download_resumes_and_keeps_failed_cache_in_work(
    tmp_path, monkeypatch, failures, error_type
):
    import huggingface_hub

    output = tmp_path / "work/data/cross-history"
    calls, sleeps, workers = [], [], []
    error = error_type("Connection broken: IncompleteRead")

    def download(**arguments):
        calls.append(arguments)
        source = Path(arguments["local_dir"])
        partial = source / ".cache/huggingface/download/offers.incomplete"
        if len(calls) > 1:
            assert partial.read_bytes() == b"partial offers"
        if len(calls) <= failures:
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_bytes(b"partial offers")
            raise error
        (source / "offers.pq").write_bytes(b"complete")

    def run_workers(**arguments):
        assert (arguments["source_dir"] / "offers.pq").is_file()
        workers.append(arguments)
        return [], 0

    offers = pl.DataFrame(
        {
            "client_id": ["1"],
            "timestamp": [dt.datetime(2025, 1, 1, tzinfo=dt.UTC)],
            "brand_id": ["100"],
            "action": ["click"],
        }
    )
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(cross_history.time, "sleep", sleeps.append)
    monkeypatch.setattr(cross_history, "_run_payment_workers", run_workers)
    monkeypatch.setattr(cross_history, "_load_offers", lambda _: offers)
    should_fail = (
        failures == cross_history.SOURCE_DOWNLOAD_ATTEMPTS or error_type is ValueError
    )
    arguments = {
        "output_dir": output,
        "revision": "pinned",
        "payment_start_day": 1,
        "payment_end_day": 1,
    }
    if should_fail:
        with pytest.raises(error_type) as caught:
            cross_history.download_and_build_cross_history(**arguments)
        assert caught.value is error
        assert not output.exists()
        assert not workers
    else:
        cross_history.download_and_build_cross_history(**arguments)
        assert len(workers) == 1
        assert (output / "eventhub.tar").is_file()
        assert json.loads((output / "report.json").read_text())["rows"] == {
            "offers": 1,
            "payments": 0,
        }
    source = Path(calls[0]["local_dir"])
    assert source.parent == output.parent
    if should_fail:
        assert (
            source / ".cache/huggingface/download/offers.incomplete"
        ).read_bytes() == b"partial offers"
    else:
        assert not source.exists()
    attempts = failures if should_fail else failures + 1
    assert len(calls) == attempts
    assert all(
        call
        == {
            "repo_id": cross_history.DATASET_REPOSITORY,
            "repo_type": "dataset",
            "revision": "pinned",
            "allow_patterns": [
                "dataset/small/offers/**",
                "dataset/small/users.pq",
                "dataset/small/brands.pq",
            ],
            "local_dir": str(source),
            "max_workers": 8,
        }
        for call in calls
    )
    assert sleeps == [cross_history.SOURCE_DOWNLOAD_RETRY_DELAY_SECONDS] * (
        attempts - 1
    )
