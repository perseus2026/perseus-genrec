"""Collected phase logs for long CPU/GPU jobs, without changing their execution."""

from __future__ import annotations

import json
import logging
import os
import resource
import shutil
import sys
import threading
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path


@contextmanager
def log_phase(logger, name, *, interval_seconds=30, output_path: Path | None = None):
    if not logger.isEnabledFor(logging.INFO):
        yield
        return
    started = last_time = time.monotonic()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    last_cpu = usage.ru_utime + usage.ru_stime
    finished = threading.Event()

    def measurements():
        nonlocal last_time, last_cpu
        now = time.monotonic()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        cpu = usage.ru_utime + usage.ru_stime
        cores = (cpu - last_cpu) / max(now - last_time, 1e-9)
        last_time, last_cpu = now, cpu
        rss_divisor = 2**30 if sys.platform == "darwin" else 2**20
        disk = shutil.disk_usage(Path.cwd())
        values = (
            f"cpu_cores={cores:.2f} process_peak_rss_gib={usage.ru_maxrss / rss_divisor:.2f} "
            f"filesystem_used_gib={disk.used / 2**30:.2f} filesystem_free_gib={disk.free / 2**30:.2f}"
        )
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            device = int(os.environ.get("LOCAL_RANK", "0"))
            values += (
                f" gpu_allocated_gib={torch.cuda.memory_allocated(device) / 2**30:.2f}"
                f" gpu_peak_reserved_gib={torch.cuda.max_memory_reserved(device) / 2**30:.2f}"
            )
        if output_path is not None and output_path.is_file():
            values += f" output_gib={output_path.stat().st_size / 2**30:.2f}"
        return values

    def heartbeat():
        while not finished.wait(interval_seconds):
            logger.info(
                "%s in progress elapsed=%.1fs %s",
                name,
                time.monotonic() - started,
                measurements(),
            )

    logger.info("Starting %s %s", name, measurements())
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    success = False
    try:
        yield
        success = True
    finally:
        finished.set()
        thread.join()
        logger.info(
            "%s %s elapsed=%.1fs %s",
            "Completed" if success else "Stopped",
            name,
            time.monotonic() - started,
            measurements(),
        )


def logged_stage(function):
    """Instrument CLI stages while preserving signatures for Cyclopts."""

    @wraps(function)
    def run(*args, **kwargs):
        started = time.monotonic()
        success = False
        try:
            with log_phase(
                logging.getLogger(function.__module__),
                function.__name__.replace("_", " "),
            ):
                result = function(*args, **kwargs)
                success = True
                return result
        finally:
            # Explicitly enabled only by job presets. Each DDP rank has its own
            # file; the final report takes the slowest inference rank.
            if success and (destination := os.environ.get("GENRECSYS_TIMINGS_PATH")):
                path = Path(destination)
                rank = os.environ.get("RANK")
                if rank is not None:
                    path = path.with_name(f"{path.stem}-rank{rank}{path.suffix}")
                records = json.loads(path.read_text()) if path.exists() else []
                records.append(
                    {
                        "stage": function.__name__,
                        "elapsed_seconds": time.monotonic() - started,
                        "success": success,
                        "rank": int(rank) if rank is not None else None,
                    }
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(records, indent=2) + "\n")

    return run
