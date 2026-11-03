"""Build the auxiliary T-ECD Small histories used by cross-domain OneRec."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Iterator
from itertools import islice
from pathlib import Path

import polars as pl

from data_sources.event_hub import distribute_to_partitions, partition_filename
from genrecsys.utils.files import discard_files, publish_directory
from model_datasets.tecd.retail import DATASET_REPOSITORY

PAYMENT_START_DAY = 1308
PAYMENT_END_DAY = 715
PAYMENT_BATCH_SIZE = 8
PAYMENT_WORKERS = 8
PAYMENT_PROGRESS_INTERVAL_SECONDS = 30
PAYMENT_DOWNLOAD_ATTEMPTS = 5
PAYMENT_DOWNLOAD_RETRY_DELAY_SECONDS = 5
SOURCE_DOWNLOAD_ATTEMPTS = 5
SOURCE_DOWNLOAD_RETRY_DELAY_SECONDS = 5

logger = logging.getLogger(__name__)


def download_and_build_cross_history(
    *,
    output_dir: Path,
    revision: str,
    payment_start_day: int = PAYMENT_START_DAY,
    payment_end_day: int = PAYMENT_END_DAY,
) -> None:
    """Download pinned Offers/Payments inputs and publish one EventHub archive."""

    from huggingface_hub import snapshot_download
    from requests.exceptions import ChunkedEncodingError

    if payment_start_day < payment_end_day:
        raise ValueError("payment_start_day must be greater than payment_end_day")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    source = Path(
        tempfile.mkdtemp(prefix="tecd-small-cross-history-", dir=output_dir.parent)
    )
    started_at = time.monotonic()
    logger.info("Downloading pinned T-ECD Small Offers/users/brands into %s", source)
    for attempt in range(1, SOURCE_DOWNLOAD_ATTEMPTS + 1):
        try:
            snapshot_download(
                repo_id=DATASET_REPOSITORY,
                repo_type="dataset",
                revision=revision,
                allow_patterns=[
                    "dataset/small/offers/**",
                    "dataset/small/users.pq",
                    "dataset/small/brands.pq",
                ],
                local_dir=str(source),
                max_workers=8,
            )
            break
        except ChunkedEncodingError as error:
            if attempt == SOURCE_DOWNLOAD_ATTEMPTS:
                raise
            logger.warning(
                "Interrupted Offers/users/brands download (attempt %d/%d); "
                "retrying in %ds: %s",
                attempt,
                SOURCE_DOWNLOAD_ATTEMPTS,
                SOURCE_DOWNLOAD_RETRY_DELAY_SECONDS,
                error,
            )
            time.sleep(SOURCE_DOWNLOAD_RETRY_DELAY_SECONDS)
    logger.info(
        "Downloaded pinned T-ECD Small Offers/users/brands in %.1fs",
        time.monotonic() - started_at,
    )
    worker_outputs, payment_rows = _run_payment_workers(
        source_dir=source,
        revision=revision,
        start_day=payment_start_day,
        end_day=payment_end_day,
    )
    offers = _load_offers(source)
    with publish_directory(output_dir, overwrite=False) as output:
        eventhub = output / "eventhub"
        eventhub.mkdir()
        _write_offer_events(eventhub, offers)
        for worker_output in worker_outputs:
            _merge_eventhub(worker_output / "eventhub", eventhub)
        _finalize_output(
            output=output,
            source_revision=revision,
            payment_start_day=payment_start_day,
            payment_end_day=payment_end_day,
            offer_rows=len(offers),
            payment_rows=payment_rows,
        )
    # This function owns the directory; preserve failed downloads for diagnosis.
    shutil.rmtree(source)
    logger.info("Removed consumed Cross source: %s", source)


def download_and_build_payment_range(
    *,
    source_dir: Path,
    output_dir: Path,
    revision: str,
    start_day: int,
    end_day: int,
) -> None:
    """Download and process one isolated range of T-ECD payment days."""

    from huggingface_hub import hf_hub_download
    from requests.exceptions import ChunkedEncodingError

    if start_day < end_day:
        raise ValueError("start_day must be greater than end_day")
    users, brands = _load_payment_filters(source_dir)
    eventhub = output_dir / "eventhub"
    eventhub.mkdir(parents=True)
    download_dir = output_dir / "downloads"
    days = range(start_day, end_day - 1, -1)
    total = start_day - end_day + 1
    payment_rows = 0
    logger.info(
        "Payment worker started: days=%d..%d files=%d polars_threads=%s",
        start_day,
        end_day,
        total,
        os.environ.get("POLARS_MAX_THREADS", "default"),
    )
    for index, day in enumerate(days, start=1):
        filename = f"dataset/full/payments/events/{day:05d}.pq"
        started_at = time.monotonic()
        logger.info(
            "Payment worker %d..%d downloading file %d/%d: %s",
            start_day,
            end_day,
            index,
            total,
            filename,
        )
        for attempt in range(1, PAYMENT_DOWNLOAD_ATTEMPTS + 1):
            try:
                path = Path(
                    hf_hub_download(
                        repo_id=DATASET_REPOSITORY,
                        repo_type="dataset",
                        revision=revision,
                        filename=filename,
                        local_dir=download_dir,
                    )
                )
                break
            except ChunkedEncodingError as error:
                if attempt == PAYMENT_DOWNLOAD_ATTEMPTS:
                    raise
                # Keep the partial file: hf_hub_download resumes it on retry.
                logger.warning(
                    "Interrupted payment download %s (attempt %d/%d); "
                    "retrying in %ds: %s",
                    filename,
                    attempt,
                    PAYMENT_DOWNLOAD_ATTEMPTS,
                    PAYMENT_DOWNLOAD_RETRY_DELAY_SECONDS,
                    error,
                )
                time.sleep(PAYMENT_DOWNLOAD_RETRY_DELAY_SECONDS)
        payments = _load_payments((path,), users, brands)
        rows = len(payments)
        if rows:
            payment_rows += rows
            _write_payment_events(eventhub, payments)
        discard_files([path], logger=logger)
        logger.info(
            "Payment worker %d..%d processed file %d/%d: day=%d rows=%d "
            "total_rows=%d elapsed=%.1fs",
            start_day,
            end_day,
            index,
            total,
            day,
            rows,
            payment_rows,
            time.monotonic() - started_at,
        )
    (output_dir / "report.json").write_text(
        json.dumps({"rows": payment_rows}) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Payment worker finished: days=%d..%d files=%d rows=%d",
        start_day,
        end_day,
        total,
        payment_rows,
    )


def build_cross_history(
    *,
    source_dir: Path,
    output_dir: Path,
    source_revision: str,
    payment_start_day: int = PAYMENT_START_DAY,
    payment_end_day: int = PAYMENT_END_DAY,
) -> None:
    """Build Offers/Payments EventHub streams without any Perseus artifacts."""

    payment_paths = [
        source_dir / f"dataset/full/payments/events/{day:05d}.pq"
        for day in range(payment_start_day, payment_end_day - 1, -1)
    ]
    missing = [str(path) for path in payment_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"T-ECD payment inputs are missing ({len(missing)}): {missing[:3]}"
        )
    _build_cross_history(
        source_dir=source_dir,
        output_dir=output_dir,
        source_revision=source_revision,
        payment_paths=iter(payment_paths),
        payment_start_day=payment_start_day,
        payment_end_day=payment_end_day,
    )


def _build_cross_history(
    *,
    source_dir: Path,
    output_dir: Path,
    source_revision: str,
    payment_paths: Iterator[Path],
    payment_start_day: int,
    payment_end_day: int,
) -> None:
    offers = _load_offers(source_dir)
    users, brands = _load_payment_filters(source_dir)
    payment_rows = 0
    payment_file_count = payment_start_day - payment_end_day + 1
    payment_batch_count = (
        payment_file_count + PAYMENT_BATCH_SIZE - 1
    ) // PAYMENT_BATCH_SIZE
    logger.info(
        "Processing %d local payment files in %d batches",
        payment_file_count,
        payment_batch_count,
    )
    with publish_directory(output_dir, overwrite=False) as output:
        eventhub = output / "eventhub"
        eventhub.mkdir()
        _write_offer_events(eventhub, offers)
        for batch_index, paths in enumerate(
            _batches(payment_paths, PAYMENT_BATCH_SIZE), start=1
        ):
            started_at = time.monotonic()
            payments = _load_payments(paths, users, brands)
            batch_rows = len(payments)
            if batch_rows:
                payment_rows += batch_rows
                _write_payment_events(eventhub, payments)
            logger.info(
                "Processed local payment batch %d/%d: files=%d rows=%d "
                "total_rows=%d elapsed=%.1fs",
                batch_index,
                payment_batch_count,
                len(paths),
                batch_rows,
                payment_rows,
                time.monotonic() - started_at,
            )
        _finalize_output(
            output=output,
            source_revision=source_revision,
            payment_start_day=payment_start_day,
            payment_end_day=payment_end_day,
            offer_rows=len(offers),
            payment_rows=payment_rows,
        )


def _load_offers(source_dir: Path) -> pl.DataFrame:
    offers_root = source_dir / "dataset/small/offers"
    users_path = source_dir / "dataset/small/users.pq"
    brands_path = source_dir / "dataset/small/brands.pq"
    required = (
        offers_root / "events",
        offers_root / "items.pq",
        users_path,
        brands_path,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"T-ECD cross-history input does not exist: {path}")

    offer_items = pl.read_parquet(
        offers_root / "items.pq", columns=["item_id", "brand_id"]
    ).select(
        pl.col("item_id").cast(pl.String),
        pl.col("brand_id").cast(pl.String),
    )
    offers = (
        pl.scan_parquet(offers_root / "events/*.pq")
        .filter(pl.col("action_type") != "view")
        .select(
            pl.col("action_type").cast(pl.String).alias("action"),
            _timestamp(),
            pl.col("user_id").cast(pl.String).alias("client_id"),
            pl.col("item_id").cast(pl.String),
        )
        .collect(engine="streaming")
        .join(offer_items, on="item_id", how="left", validate="m:1")
        .drop_nulls(["client_id", "timestamp", "brand_id"])
        .unique()
    )
    return offers


def _load_payment_filters(source_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    users_path = source_dir / "dataset/small/users.pq"
    brands_path = source_dir / "dataset/small/brands.pq"
    for path in (users_path, brands_path):
        if not path.is_file():
            raise FileNotFoundError(f"T-ECD cross-history input does not exist: {path}")
    users = pl.read_parquet(users_path, columns=["user_id"]).select(
        pl.col("user_id").cast(pl.String).alias("client_id")
    )
    brands = pl.read_parquet(brands_path, columns=["brand_id"]).select(
        pl.col("brand_id").cast(pl.String)
    )
    return users, brands


def _load_payments(
    paths: tuple[Path, ...], users: pl.DataFrame, brands: pl.DataFrame
) -> pl.DataFrame:
    return (
        pl.scan_parquet([str(path) for path in paths])
        .select(
            _timestamp(),
            pl.col("user_id").cast(pl.String).alias("client_id"),
            pl.col("brand_id").cast(pl.String),
            pl.col("price"),
        )
        .join(users.lazy(), on="client_id", how="semi")
        .join(brands.lazy(), on="brand_id", how="semi")
        .collect(engine="streaming")
        .drop_nulls(["client_id", "timestamp", "brand_id"])
        .unique()
    )


def _payment_ranges(
    start_day: int, end_day: int, workers: int
) -> tuple[tuple[int, int], ...]:
    day_count = start_day - end_day + 1
    chunk_size = (day_count + workers - 1) // workers
    return tuple(
        (chunk_start, max(end_day, chunk_start - chunk_size + 1))
        for chunk_start in range(start_day, end_day - 1, -chunk_size)
    )


def _run_payment_workers(
    *,
    source_dir: Path,
    revision: str,
    start_day: int,
    end_day: int,
) -> tuple[tuple[Path, ...], int]:
    ranges = _payment_ranges(start_day, end_day, PAYMENT_WORKERS)
    worker_root = source_dir / "payment-workers"
    environment = os.environ.copy()
    total_threads = int(environment.get("POLARS_MAX_THREADS", "32"))
    environment["POLARS_MAX_THREADS"] = str(max(1, total_threads // len(ranges)))
    environment["PYTHONUNBUFFERED"] = "1"
    processes: list[tuple[tuple[int, int], Path, subprocess.Popen]] = []
    logger.info(
        "Starting %d payment workers for %d days: ranges=%s threads_per_worker=%s",
        len(ranges),
        start_day - end_day + 1,
        ranges,
        environment["POLARS_MAX_THREADS"],
    )
    for index, (worker_start, worker_end) in enumerate(ranges, start=1):
        worker_output = worker_root / f"worker-{index:02d}"
        command = [
            sys.executable,
            "-m",
            "model_datasets.tecd",
            "download-and-build-payment-range",
            f"--source-dir={source_dir}",
            f"--output-dir={worker_output}",
            f"--revision={revision}",
            f"--start-day={worker_start}",
            f"--end-day={worker_end}",
        ]
        logger.info(
            "Launching payment worker %d/%d: days=%d..%d",
            index,
            len(ranges),
            worker_start,
            worker_end,
        )
        processes.append(
            (
                (worker_start, worker_end),
                worker_output,
                subprocess.Popen(command, env=environment),
            )
        )
    while True:
        active = [days for days, _, process in processes if process.poll() is None]
        completed = len(processes) - len(active)
        logger.info(
            "Payment workers progress: completed=%d/%d active_ranges=%s",
            completed,
            len(processes),
            active,
        )
        if not active:
            break
        time.sleep(PAYMENT_PROGRESS_INTERVAL_SECONDS)
    return_codes = [process.returncode for _, _, process in processes]
    if any(return_code != 0 for return_code in return_codes):
        raise RuntimeError(f"Payment workers failed: {return_codes}")
    outputs = tuple(output for _, output, _ in processes)
    payment_rows = sum(
        json.loads((output / "report.json").read_text())["rows"] for output in outputs
    )
    logger.info(
        "All payment workers finished: workers=%d rows=%d", len(outputs), payment_rows
    )
    return outputs, payment_rows


def _merge_eventhub(source: Path, destination: Path) -> None:
    """Consume disposable payment-worker output without duplicating it on disk."""
    for path in source.rglob("*.pq"):
        target = destination / path.relative_to(source)
        if target.exists():
            raise FileExistsError(
                f"Payment workers produced overlapping output: {target}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(path, target)


def _write_offer_events(eventhub: Path, offers: pl.DataFrame) -> None:
    _write_events(
        eventhub,
        offers.with_columns(
            (pl.lit("t-ecd-small-offers-") + pl.col("action")).alias("event")
        ).drop("action"),
    )


def _write_payment_events(eventhub: Path, payments: pl.DataFrame) -> None:
    _write_events(
        eventhub,
        payments.with_columns(pl.lit("t-ecd-payments").alias("event")),
    )


def _finalize_output(
    *,
    output: Path,
    source_revision: str,
    payment_start_day: int,
    payment_end_day: int,
    offer_rows: int,
    payment_rows: int,
) -> None:
    eventhub = output / "eventhub"
    archive = output / "eventhub.tar"
    logger.info("Archiving EventHub: %s", archive)
    with tarfile.open(archive, mode="w") as target:
        target.add(eventhub, arcname="eventhub")
    shutil.rmtree(eventhub)
    (output / "report.json").write_text(
        json.dumps(
            {
                "dataset": "T-ECD Small cross-domain auxiliary history",
                "source_repository": DATASET_REPOSITORY,
                "source_revision": source_revision,
                "payment_days": {
                    "start": payment_start_day,
                    "end": payment_end_day,
                },
                "rows": {"offers": offer_rows, "payments": payment_rows},
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Cross-history ready: offers=%d payments=%d archive=%s",
        offer_rows,
        payment_rows,
        archive,
    )


def _batches(paths: Iterator[Path], size: int) -> Iterator[tuple[Path, ...]]:
    while batch := tuple(islice(paths, size)):
        yield batch


def _timestamp() -> pl.Expr:
    return (
        pl.col("timestamp")
        .dt.total_microseconds()
        .cast(pl.Datetime("us"))
        .dt.truncate("1s")
        .cast(pl.Datetime("ns"))
        .alias("timestamp")
    )


def _write_events(root: Path, events: pl.DataFrame) -> None:
    values = distribute_to_partitions(
        events.with_columns(pl.col("timestamp").dt.date().alias("date"))
    )
    for (event, day, partition), frame in values.group_by(
        "event", "date", "partition_id", maintain_order=True
    ):
        destination = root / str(event) / day.isoformat()
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / partition_filename(int(partition))
        frame = frame.drop("event", "date", "partition_id")
        if path.exists():
            frame = pl.concat([pl.read_parquet(path), frame], how="diagonal_relaxed")
        frame.unique().sort("client_id", "timestamp").write_parquet(path)


__all__ = [
    "build_cross_history",
    "download_and_build_cross_history",
    "download_and_build_payment_range",
]
