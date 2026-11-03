"""Instrumentation preserves CLI signatures and writes separate DDP timings."""

import inspect
import json
import logging
import sys
import types
from threading import Event

import pytest

from genrecsys.utils.progress import log_phase, logged_stage


def test_phase_heartbeat_and_failure_keep_exception_and_do_not_initialize_cuda(
    caplog, monkeypatch
):
    logger = logging.getLogger("test.stage.progress")
    observed = Event()
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_initialized=lambda: False),
        ),
    )

    class Handler(logging.Handler):
        def emit(self, record):
            if "in progress" in record.getMessage():
                observed.set()

    handler = Handler()
    logger.addHandler(handler)
    try:
        with caplog.at_level(logging.INFO):
            with log_phase(logger, "success", interval_seconds=0.01):
                assert observed.wait(timeout=2)
            with (
                pytest.raises(ValueError, match="original"),
                log_phase(logger, "failure"),
            ):
                raise ValueError("original")
    finally:
        logger.removeHandler(handler)
    assert "Starting success" in caplog.text
    assert "Completed success" in caplog.text
    assert "Stopped failure" in caplog.text
    assert "Completed failure" not in caplog.text
    assert "process_peak_rss_gib=" in caplog.text
    assert "cpu_cores=" in caplog.text
    assert "filesystem_free_gib=" in caplog.text
    assert "gpu_allocated" not in caplog.text


def test_logged_stage_keeps_signature_and_separates_rank_files(tmp_path, monkeypatch):
    @logged_stage
    def example(*, value: int = 1):
        if value < 0:
            raise ValueError("failure")
        return value

    assert inspect.signature(example) == inspect.signature(example.__wrapped__)
    monkeypatch.setenv("GENRECSYS_TIMINGS_PATH", str(tmp_path / "timings.json"))
    monkeypatch.setenv("RANK", "3")
    assert example(value=2) == 2
    path = tmp_path / "timings-rank3.json"
    records = json.loads(path.read_text())
    assert records[0]["stage"] == "example"
    assert records[0]["rank"] == 3
    assert records[0]["success"] is True
    assert records[0]["elapsed_seconds"] >= 0
    with pytest.raises(ValueError, match="failure"):
        example(value=-1)
    assert json.loads(path.read_text()) == records
    assert not (tmp_path / "timings.json").exists()
