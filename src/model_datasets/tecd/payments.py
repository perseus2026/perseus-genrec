"""Build the Full T-ECD Payments history used by matched MD-to-SD runs."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import polars as pl
from cyclopts import App

from data_sources.event_hub import client_partition_id, partition_filename
from genrecsys.utils.files import discard_files, publish_directory
from genrecsys.utils.progress import log_phase, logged_stage
from model_datasets.tecd.retail import archive_eventhub

logger = logging.getLogger(__name__)
app = App()

DATASET_REPOSITORY = "t-tech/T-ECD"
DATASET_REVISION = "c55530a7a179f06330ff626d35634d20c8a56f43"
START_DAY = 1092
END_DAY = 1307
PROGRESS_INTERVAL_SECONDS = 30

PAYMENT_EVENT = "t-ecd-full-payments-payment"
RECEIPT_EVENT = "t-ecd-full-payments-receipt"


def _log_phase(phase: str, *, output_path: Path | None = None):
    """Keep long blocking I/O phases visible in collected job logs."""

    return log_phase(
        logger,
        phase,
        interval_seconds=PROGRESS_INTERVAL_SECONDS,
        output_path=output_path,
    )


def _target_users(samples_root: Path) -> pl.DataFrame:
    paths = sorted(samples_root.rglob("samples.pq"))
    if not paths:
        raise FileNotFoundError(f"no target basis samples under {samples_root}")
    users = (
        pl.scan_parquet(paths)
        .select(pl.col("client_id").cast(pl.String))
        .drop_nulls()
        .unique()
        .collect(engine="streaming")
        .with_columns(
            partition_id=pl.col("client_id").map_elements(
                client_partition_id,
                return_dtype=pl.UInt32,
            )
        )
    )
    if users.is_empty():
        raise ValueError("target bases contain no users")
    logger.info("Full MD-to-SD payment filter contains %d users", len(users))
    return users


def _timestamp() -> pl.Expr:
    return (
        pl.col("timestamp")
        .dt.total_microseconds()
        .cast(pl.Datetime("us"))
        .dt.truncate("1s")
        .cast(pl.Datetime("ns"))
    )


def _read_payment_events(path: Path, users: pl.DataFrame) -> pl.DataFrame:
    return (
        pl.scan_parquet(path)
        .select(
            _timestamp(),
            pl.col("user_id").cast(pl.String).alias("client_id"),
            pl.col("brand_id").cast(pl.String),
        )
        .drop_nulls(["timestamp", "client_id"])
        .join(users.lazy(), on="client_id", how="inner", validate="m:1")
        .collect(engine="streaming")
        .unique()
    )


def _read_receipts(path: Path, users: pl.DataFrame) -> pl.DataFrame:
    # approximate_item_id deliberately stays out of OneRec global lifelong:
    # auxiliary domains are connected through brand and event type only.
    return (
        pl.scan_parquet(path)
        .select(
            _timestamp(),
            pl.col("user_id").cast(pl.String).alias("client_id"),
            pl.col("brand_id").cast(pl.String),
        )
        .drop_nulls(["timestamp", "client_id"])
        .join(users.lazy(), on="client_id", how="inner", validate="m:1")
        .collect(engine="streaming")
        .unique()
    )


def _write_events(root: Path, event: str, values: pl.DataFrame) -> None:
    if values.is_empty():
        return
    values = values.with_columns(pl.col("timestamp").dt.date().alias("date"))
    for (day, partition), frame in values.group_by(
        "date", "partition_id", maintain_order=True
    ):
        destination = root / event / day.isoformat()
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / partition_filename(int(partition))
        if path.exists():
            raise FileExistsError(f"overlapping raw payment partitions: {path}")
        frame.select("client_id", "timestamp", "brand_id").sort(
            "client_id", "timestamp"
        ).write_parquet(path)


def _read_payment_file(
    kind: str,
    day: int,
    *,
    source: Callable[[str, int], Path],
    users: pl.DataFrame,
) -> tuple[pl.DataFrame, int, Path]:
    with _log_phase(f"Payments source {kind}/{day:05d}"):
        path = source(kind, day)
    source_bytes = path.stat().st_size
    logger.info(
        "Payments source %s/%05d: size=%.2f GiB", kind, day, source_bytes / 2**30
    )
    with _log_phase(f"Payments read/filter {kind}/{day:05d}"):
        values = (
            _read_payment_events(path, users)
            if kind == "events"
            else _read_receipts(path, users)
        )
    return values, source_bytes, path


def _build_full_payments_history(
    *,
    samples_root: Path,
    output_dir: Path,
    source_revision: str,
    start_day: int,
    end_day: int,
    source: Callable[[str, int], Path],
    delete_source_files: bool,
    file_workers: int,
) -> None:
    if start_day > end_day:
        raise ValueError("start_day must not exceed end_day")
    if file_workers < 1:
        raise ValueError("file_workers must be positive")
    started_at = time.monotonic()
    with _log_phase("Payments target user filter"):
        users = _target_users(samples_root)
    payment_rows = 0
    receipt_rows = 0
    payment_brands: set[str] = set()
    receipt_brands: set[str] = set()
    source_bytes = 0
    total_files = (end_day - start_day + 1) * 2
    logger.info(
        "Starting Full Payments files: days=%d-%d files=%d workers=%d users=%d",
        start_day,
        end_day,
        total_files,
        file_workers,
        len(users),
    )

    with publish_directory(output_dir, overwrite=False) as output:
        eventhub = output / "eventhub"
        eventhub.mkdir()
        files = iter(
            (kind, day)
            for day in range(start_day, end_day + 1)
            for kind in ("events", "receipts")
        )
        files_started = time.monotonic()
        # At most file_workers raw files/filtered frames are alive. The consumer
        # writes and aggregates in source order, without a dataset-sized queue.
        with ThreadPoolExecutor(max_workers=file_workers) as executor:
            pending = deque()

            def submit_next() -> None:
                entry = next(files, None)
                if entry is not None:
                    kind, day = entry
                    pending.append(
                        (
                            kind,
                            day,
                            executor.submit(
                                _read_payment_file,
                                kind,
                                day,
                                source=source,
                                users=users,
                            ),
                        )
                    )

            for _ in range(file_workers):
                submit_next()
            completed_files = 0
            while pending:
                kind, day, future = pending.popleft()
                while not wait([future], timeout=PROGRESS_INTERVAL_SECONDS).done:
                    elapsed = time.monotonic() - files_started
                    eta = (
                        f"{elapsed / completed_files * (total_files - completed_files) / 60:.1f}min"
                        if completed_files
                        else "unknown (no completed files)"
                    )
                    logger.info(
                        "Full Payments progress: files=%d/%d workers=%d "
                        "waiting=%s/%05d elapsed=%.1fmin size=%.2f GiB eta=%s",
                        completed_files,
                        total_files,
                        file_workers,
                        kind,
                        day,
                        elapsed / 60,
                        source_bytes / 2**30,
                        eta,
                    )
                values, size, source_path = future.result()
                rows = len(values)
                with _log_phase(f"Payments write {kind}/{day:05d} rows={rows}"):
                    _write_events(
                        eventhub,
                        PAYMENT_EVENT if kind == "events" else RECEIPT_EVENT,
                        values,
                    )
                if delete_source_files:
                    discard_files([source_path], logger=logger)
                brands = set(values["brand_id"].drop_nulls().unique())
                if kind == "events":
                    payment_rows += rows
                    payment_brands.update(brands)
                else:
                    receipt_rows += rows
                    receipt_brands.update(brands)
                source_bytes += size
                completed_files += 1
                elapsed = time.monotonic() - files_started
                logger.info(
                    "Processed Full Payments %s/%05d (%d/%d): rows=%d "
                    "workers=%d elapsed=%.1fmin size=%.2f GiB "
                    "throughput=%.2f MiB/s eta=%.1fmin",
                    kind,
                    day,
                    completed_files,
                    total_files,
                    rows,
                    file_workers,
                    elapsed / 60,
                    source_bytes / 2**30,
                    source_bytes / 2**20 / elapsed,
                    elapsed / completed_files * (total_files - completed_files) / 60,
                )
                del values, future
                submit_next()

        archive = output / "eventhub.tar"
        with _log_phase("Payments archive eventhub.tar", output_path=archive):
            archive_eventhub(basis=output)
        logger.info(
            "Completed Full Payments files: files=%d/%d workers=%d "
            "payments=%d receipts=%d archive=%.2f GiB elapsed=%.1fmin",
            completed_files,
            total_files,
            file_workers,
            payment_rows,
            receipt_rows,
            archive.stat().st_size / 2**30,
            (time.monotonic() - started_at) / 60,
        )
        report = {
            "dataset": "T-ECD Full Payments MD-to-SD history",
            "source_repository": DATASET_REPOSITORY,
            "source_revision": source_revision,
            "days": [start_day, end_day],
            "target_users": len(users),
            "rows": {
                "payment": payment_rows,
                "payment_receipt": receipt_rows,
            },
            "unique_brands": {
                "payment": len(payment_brands),
                "payment_receipt": len(receipt_brands),
                "union": len(payment_brands | receipt_brands),
            },
            "item_identity_used": False,
            "file_workers": file_workers,
            "source_files": completed_files,
            "source_bytes": source_bytes,
            "archive_bytes": archive.stat().st_size,
            "elapsed_seconds": time.monotonic() - started_at,
        }
        (output / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


@app.command
@logged_stage
def download_and_build_full_payments_history(
    *,
    samples_root: Path,
    output_dir: Path,
    revision: str = DATASET_REVISION,
    start_day: int = START_DAY,
    end_day: int = END_DAY,
    file_workers: int = 1,
) -> None:
    """Download/process bounded raw files and delete each after writing history."""

    from huggingface_hub import hf_hub_download

    with tempfile.TemporaryDirectory(prefix="tecd-full-payments-") as temporary:
        download_root = Path(temporary)

        def download(kind: str, day: int) -> Path:
            filename = f"dataset/full/payments/{kind}/{day:05d}.pq"
            return Path(
                hf_hub_download(
                    repo_id=DATASET_REPOSITORY,
                    repo_type="dataset",
                    revision=revision,
                    filename=filename,
                    local_dir=download_root,
                )
            )

        _build_full_payments_history(
            samples_root=samples_root,
            output_dir=output_dir,
            source_revision=revision,
            start_day=start_day,
            end_day=end_day,
            source=download,
            delete_source_files=True,
            file_workers=file_workers,
        )


@app.command
def build_full_payments_history(
    *,
    source_dir: Path,
    samples_root: Path,
    output_dir: Path,
    source_revision: str = "local",
    start_day: int = START_DAY,
    end_day: int = END_DAY,
    file_workers: int = 1,
) -> None:
    """Build a Payments archive from already available raw files."""

    def local(kind: str, day: int) -> Path:
        path = source_dir / f"dataset/full/payments/{kind}/{day:05d}.pq"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    _build_full_payments_history(
        samples_root=samples_root,
        output_dir=output_dir,
        source_revision=source_revision,
        start_day=start_day,
        end_day=end_day,
        source=local,
        delete_source_files=False,
        file_workers=file_workers,
    )


@app.command
@logged_stage
def extract_histories(*, history_dir: Path, workdir: Path) -> None:
    """Extract the four downloaded history archives, then discard each tar."""

    archives = sorted(history_dir.glob("*.tar"))
    if not archives:
        raise FileNotFoundError(f"no history archives under {history_dir}")
    workdir.mkdir(parents=True, exist_ok=True)
    for index, archive in enumerate(archives, start=1):
        phase = (
            f"Extract history {archive.name} ({index}/{len(archives)}) "
            f"size={archive.stat().st_size / 2**30:.2f} GiB"
        )
        with _log_phase(phase):
            subprocess.run(["tar", "-xf", str(archive), "-C", str(workdir)], check=True)
        discard_files([archive], logger=logger)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app()
