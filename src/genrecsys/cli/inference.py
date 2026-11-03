"""Partitioned fused inference over the shared GenRecSys workdir."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import polars as pl
import pyarrow.parquet as pq
from cyclopts import App

from data_sources import event_hub
from genrecsys import api, core
from genrecsys.utils.progress import logged_stage

app = App()
logger = logging.getLogger(__name__)


@app.command
@logged_stage
def distribute_samples(
    *,
    history_source: Literal["event_hub", "event_store"] = "event_hub",
    workdir: Path = Path("workdir/"),
) -> None:
    """Partition test samples by stable client hash and write samples/*.pq.

    Read raw basis for EventHub inference or prepared samples for EventStore inference.
    """

    samples_path = (
        workdir / "basis" / "test" / "samples.pq"
        if history_source == "event_hub"
        else workdir / "dataset" / "test" / "samples.pq"
    )
    samples_dir = workdir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    writers = {}
    try:
        for batch in pq.ParquetFile(samples_path).iter_batches(batch_size=4096):
            for partition_id, partition_samples in api.distribute_samples(
                pl.from_arrow(batch)
            ):
                path = samples_dir / event_hub.partition_filename(partition_id)
                table = partition_samples.to_arrow()
                if partition_id not in writers:
                    writers[partition_id] = pq.ParquetWriter(
                        path, table.schema, compression="zstd"
                    )
                writers[partition_id].write_table(table)
    finally:
        for writer in writers.values():
            writer.close()


@app.command
@logged_stage
def make_predictions(
    *,
    partitions_bin_idx: int = 1,
    partitions_num_bins: int = 1,
    prediction_write_batch_size: int = 1024,
    input_batch_size: int = 0,
    history_source: Literal["event_hub", "event_store"] = "event_hub",
    workdir: Path = Path("workdir/"),
) -> None:
    """Predict assigned partitions with one session and publish predictions/*.pq.

    Use test artifacts and the selected history source. input_batch_size=0 loads an
    entire partition; a positive value limits input rows held in RAM.
    """
    if prediction_write_batch_size < 1 or input_batch_size < 0:
        raise ValueError("write batch size must be positive; input batch size >= 0")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1 and partitions_num_bins == 1 and partitions_bin_idx == 1:
        partitions_num_bins = world_size
        partitions_bin_idx = rank + 1
    checkpoint = core.checkpoint.Trained.load(workdir / "checkpoint")
    artifacts_cls = core.tasks.retrieval.Artifacts
    test_artifacts = artifacts_cls.load(workdir / "dataset" / "test" / "artifacts")
    predictions_dir = workdir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    event_store_path = workdir / "dataset" / "events"
    event_store = None
    if history_source == "event_store":
        if not event_store_path.exists():
            raise FileNotFoundError(event_store_path)
        event_store = core.EventStore(event_store_path, backend="disk")
    assigned_paths = []
    for path in sorted((workdir / "samples").iterdir()):
        if not path.is_file() or path.suffix != ".pq":
            continue
        partition_id = event_hub.partition_id_from_filename(path.name)
        if (
            partition_id % partitions_num_bins
            != partitions_bin_idx % partitions_num_bins
        ):
            continue
        assigned_paths.append((partition_id, path))
    if not assigned_paths:
        logger.warning(
            "rank=%s/%s has no assigned inference partitions",
            rank,
            world_size,
        )
        return
    inference_session = api.create_inference_session(
        checkpoint,
        artifacts=test_artifacts,
    )
    logger.info(
        "rank=%s/%s loaded inference session for %s partitions",
        rank,
        world_size,
        len(assigned_paths),
    )
    for index, (partition_id, path) in enumerate(assigned_paths, start=1):
        started = time.perf_counter()

        def prediction_batches(path=path, partition_id=partition_id):
            """Read one sample partition in batches, reusing the inference session."""

            parquet = pq.ParquetFile(path)
            batch_size = input_batch_size or max(1, parquet.metadata.num_rows)
            for batch in parquet.iter_batches(batch_size=batch_size):
                yield from api.iter_predictions(
                    partition_id,
                    pl.from_arrow(batch),
                    checkpoint,
                    event_store=event_store,
                    samples_prepared=history_source == "event_store",
                    inference_session=inference_session,
                )

        rows = _write_prediction_batches(
            predictions_dir / path.name,
            prediction_batches(),
            batch_size=prediction_write_batch_size,
        )
        logger.info(
            "rank=%s/%s wrote inference partition %s/%s: id=%s, rows=%s in %.1fs",
            rank,
            world_size,
            index,
            len(assigned_paths),
            partition_id,
            rows,
            time.perf_counter() - started,
        )


def _write_prediction_batches(
    path: Path,
    batches: Iterable[pl.DataFrame],
    /,
    *,
    batch_size: int,
) -> int:
    """Atomically write prediction batches to Parquet and return the row count.

    Remove the temporary file on failure; do not publish a partial destination.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.stem}-",
        suffix=".pq",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    writer = None
    total_rows = 0
    try:
        for batch in batches:
            if batch.is_empty():
                continue
            table = batch.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary_path, table.schema, compression="zstd"
                )
            writer.write_table(table, row_group_size=batch_size)
            total_rows += len(batch)
        if writer is None:
            return 0
        writer.close()
        writer = None
        temporary_path.replace(path)
        return total_rows
    finally:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)


@app.command
@logged_stage
def evaluate(
    *,
    workdir: Path = Path("workdir/"),
) -> None:
    """Verify complete test prediction coverage and write metrics.json.

    Use dataset/test/samples.pq as ground truth and the test catalog only as the
    denominator of global coverage. Do not load model weights.
    """

    config = core.Config.load_data(workdir / "checkpoint" / "config.yaml")
    api.evaluate_predictions(
        config,
        (workdir / "predictions").glob("*.pq"),
        expected_samples=workdir / "dataset" / "test" / "samples.pq",
        candidate_items=workdir / "dataset" / "test" / "artifacts" / "items.pq",
        output_path=workdir / "metrics.json",
    )


__all__ = ["distribute_samples", "evaluate", "make_predictions"]
