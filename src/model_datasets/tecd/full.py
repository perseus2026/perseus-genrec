"""Build the missing T-ECD Full SD cells from Table 6.

Views are history-only. Targets always come from non-view events, matching the
official tecd-baselines preprocessing configurations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from cyclopts import App

from data_sources.event_hub import distribute_to_partitions, partition_filename
from genrecsys.core import Config
from genrecsys.core.dataset_manifest import finalize_bundle
from genrecsys.core.encoders.semantic_id import Preprocessor as SIDPreprocessor
from genrecsys.core.resources import copy_resource
from genrecsys.utils.files import discard_files, publish_directory
from genrecsys.utils.progress import logged_stage
from model_datasets.tecd.retail import archive_eventhub

logger = logging.getLogger(__name__)
app = App()

DATASET_REPOSITORY = "t-tech/T-ECD"
DATASET_REVISION = "c55530a7a179f06330ff626d35634d20c8a56f43"

TRAIN_START_DAY = 1092
TRAIN_END_DAY = 1291
VALIDATION_GAP_DAY = 1292
VALIDATION_START_DAY = 1293
VALIDATION_END_DAY = 1299
TEST_GAP_DAY = 1300
TEST_START_DAY = 1301
TEST_END_DAY = 1307

POSITIVE_MODE = "positives"
ALL_EVENTS_MODE = "all-events"
REPRESENTATION_BATCH_SIZE = 100_000
DOWNLOAD_PROGRESS_INTERVAL_SECONDS = 60.0

DOMAIN_SPECS: dict[str, dict[str, Any]] = {
    "retail": {
        "title": "Retail",
        "root": Path("dataset/full/retail"),
        "supported_actions": ("view", "click", "added_to_cart", "order"),
        "positive_actions": ("click", "added_to_cart", "order"),
        "like_actions": ("added_to_cart", "order"),
        "event_names": {
            "view": "t-ecd-full-retail-view",
            "click": "t-ecd-full-retail-click",
            "added_to_cart": "t-ecd-full-retail-added-to-cart",
            "order": "t-ecd-full-retail-order",
        },
        "require_brand": True,
    },
    "marketplace": {
        "title": "Marketplace",
        "root": Path("dataset/full/marketplace"),
        "supported_actions": ("view", "click", "like", "clickout"),
        "positive_actions": ("click", "like", "clickout"),
        "like_actions": ("like", "clickout"),
        "event_names": {
            "view": "t-ecd-full-marketplace-view",
            "click": "t-ecd-full-marketplace-click",
            "like": "t-ecd-full-marketplace-like",
            "clickout": "t-ecd-full-marketplace-clickout",
        },
        "require_brand": False,
    },
    "offers": {
        "title": "Offers",
        "root": Path("dataset/full/offers"),
        "supported_actions": ("view", "click", "like", "clickout"),
        "positive_actions": ("click", "like", "clickout"),
        "like_actions": ("like", "clickout"),
        "event_names": {
            "view": "t-ecd-full-offers-view",
            "click": "t-ecd-full-offers-click",
            "like": "t-ecd-full-offers-like",
            "clickout": "t-ecd-full-offers-clickout",
        },
        "require_brand": False,
    },
}


def _spec(domain: str) -> dict[str, Any]:
    try:
        return DOMAIN_SPECS[domain]
    except KeyError as error:
        raise ValueError(
            f"unsupported Full T-ECD domain {domain!r}; "
            f"expected one of {sorted(DOMAIN_SPECS)}"
        ) from error


@app.command
@logged_stage
def download_and_build_full_table6(
    *,
    domain: str,
    output_dir: Path,
    revision: str = DATASET_REVISION,
    max_workers: int = 8,
) -> None:
    """Build both Full histories with the same positives-defined holdout."""
    with tempfile.TemporaryDirectory(prefix=f"tecd-full-{domain}-") as temporary:
        source = Path(temporary)
        _download_article_window(
            domain=domain, output_dir=source, revision=revision, max_workers=max_workers
        )
        build_full_table6_sd(
            source_dir=source,
            output_dir=output_dir,
            domain=domain,
            history_modes=(POSITIVE_MODE, ALL_EVENTS_MODE),
            source_revision=revision,
            shared_holdout=True,
            delete_source_files=True,
        )


def _download_article_window(
    *,
    domain: str,
    output_dir: Path,
    revision: str,
    max_workers: int,
) -> None:
    from huggingface_hub import snapshot_download

    spec = _spec(domain)
    root = spec["root"]
    event_patterns = [
        f"{root}/events/{day:05d}.pq"
        for day in range(TRAIN_START_DAY, TEST_END_DAY + 1)
    ]
    patterns = [*event_patterns, f"{root}/items.pq"]
    paths = [output_dir / pattern for pattern in patterns]
    started = time.monotonic()
    logger.info(
        "Starting T-ECD Full %s download: revision=%s files=%d workers=%d",
        spec["title"],
        revision,
        len(patterns),
        max_workers,
    )
    download_finished = threading.Event()
    progress = threading.Thread(
        target=_log_download_progress,
        kwargs={
            "finished": download_finished,
            "title": spec["title"],
            "paths": paths,
            "started": started,
        },
        name=f"tecd-{domain}-download-progress",
        daemon=True,
    )
    progress.start()
    try:
        snapshot_download(
            repo_id=DATASET_REPOSITORY,
            repo_type="dataset",
            revision=revision,
            allow_patterns=patterns,
            local_dir=str(output_dir),
            max_workers=max_workers,
        )
    finally:
        download_finished.set()
        progress.join()
    downloaded_files, downloaded_bytes = _download_statistics(paths)
    logger.info(
        "Completed T-ECD Full %s download: files=%d/%d size=%.2f GiB elapsed=%.1f min",
        spec["title"],
        downloaded_files,
        len(paths),
        downloaded_bytes / 2**30,
        (time.monotonic() - started) / 60,
    )


def _log_download_progress(
    *,
    finished: threading.Event,
    title: str,
    paths: list[Path],
    started: float,
) -> None:
    while not finished.wait(DOWNLOAD_PROGRESS_INTERVAL_SECONDS):
        downloaded_files, downloaded_bytes = _download_statistics(paths)
        logger.info(
            "Downloading T-ECD Full %s: files=%d/%d size=%.2f GiB elapsed=%.1f min",
            title,
            downloaded_files,
            len(paths),
            downloaded_bytes / 2**30,
            (time.monotonic() - started) / 60,
        )


def _download_statistics(paths: list[Path]) -> tuple[int, int]:
    downloaded = [path for path in paths if path.is_file()]
    return len(downloaded), sum(path.stat().st_size for path in downloaded)


@logged_stage
def build_full_table6_sd(
    *,
    source_dir: Path,
    output_dir: Path,
    domain: str,
    history_modes: tuple[str, ...],
    source_revision: str = "local",
    shared_holdout: bool = False,
    delete_source_files: bool = False,
) -> None:
    """Build one or both SD history modes while keeping positive targets fixed."""

    spec = _spec(domain)
    modes = tuple(dict.fromkeys(history_modes))
    if not modes or set(modes) - {POSITIVE_MODE, ALL_EVENTS_MODE}:
        raise ValueError(f"invalid history modes: {modes}")
    if shared_holdout and POSITIVE_MODE not in modes:
        raise ValueError("shared holdout requires the positives history mode")
    nested_variants = len(modes) > 1
    started = time.monotonic()
    root = source_dir / spec["root"]
    items_path = root / "items.pq"
    event_paths = [
        root / "events" / f"{day:05d}.pq"
        for day in range(TRAIN_START_DAY, TEST_END_DAY + 1)
    ]
    if not items_path.is_file():
        raise FileNotFoundError(
            f"T-ECD Full {spec['title']} catalog does not exist: {items_path}"
        )
    if missing := [path for path in event_paths if not path.is_file()]:
        raise FileNotFoundError(
            f"T-ECD Full {spec['title']} article window is incomplete; missing: "
            + ", ".join(path.name for path in missing[:10])
        )

    logger.info(
        "Starting Full %s Table-6 build: modes=%s partitions=%d revision=%s",
        spec["title"],
        modes,
        len(event_paths),
        source_revision,
    )
    event_compressed_bytes = sum(path.stat().st_size for path in event_paths)
    items_compressed_bytes = items_path.stat().st_size
    items, embedding_dimension = _load_catalog(
        items_path,
        domain=domain,
        require_brand=bool(spec["require_brand"]),
    )
    if delete_source_files:
        discard_files([items_path], logger=logger)
    logger.info(
        "Loaded Full %s catalog: items=%d brands=%d missing_brands=%d "
        "embedding_dimension=%d",
        spec["title"],
        len(items),
        items["brand_id"].drop_nulls().n_unique(),
        items["brand_id"].null_count(),
        embedding_dimension,
    )
    item_brands = items.select("item_id", "brand_id")
    candidate_catalog = item_brands.sort("item_id")
    candidate_ids = candidate_catalog.select("item_id")
    positive_actions = tuple(spec["positive_actions"])
    supported_actions = tuple(spec["supported_actions"])

    validation_targets: list[pl.DataFrame] = []
    test_targets: list[pl.DataFrame] = []
    train_positive_item_ids: set[str] = set()
    positive_rows = 0
    target_rows = 0
    raw_supported_rows = 0
    unknown_item_rows = 0

    with publish_directory(output_dir, overwrite=False) as output:
        states: dict[str, dict[str, Any]] = {}
        for mode in modes:
            variant = output / mode if nested_variants else output
            for split in ("train", "validation", "test"):
                destination = variant / "basis" / split
                (destination / "artifacts").mkdir(parents=True)
                candidate_catalog.write_parquet(destination / "artifacts" / "items.pq")
            candidate_catalog.write_parquet(variant / "item_features.pq")
            states[mode] = {
                "variant": variant,
                "eventhub": variant / "eventhub",
                "train_seen_users": set(),
                "test_history_users": set(),
                "train_writer": None,
                "train_rows": 0,
                "train_target_items": 0,
                "train_digest": hashlib.sha256(),
                "history_rows": 0,
                "action_rows": {
                    action: 0
                    for action in (
                        positive_actions if mode == POSITIVE_MODE else supported_actions
                    )
                },
            }

        try:
            for index, path in enumerate(event_paths, start=1):
                partition_started = time.monotonic()
                day = int(path.stem)
                raw = _read_partition(path, supported_actions)
                known = raw.join(candidate_ids, on="item_id", how="semi").join(
                    item_brands,
                    on="item_id",
                    how="left",
                    validate="m:1",
                )
                unknown_item_rows += len(raw) - len(known)
                positives = known.filter(pl.col("action").is_in(positive_actions))
                targets = positives.filter(pl.col("brand_id").is_not_null())
                raw_supported_rows += len(known)
                positive_rows += len(positives)
                target_rows += len(targets)
                if day <= TRAIN_END_DAY:
                    train_positive_item_ids.update(
                        positives["item_id"].unique().to_list()
                    )
                elif VALIDATION_START_DAY <= day <= VALIDATION_END_DAY:
                    validation_targets.append(
                        targets.select("client_id", "timestamp", "item_id")
                    )
                elif TEST_START_DAY <= day <= TEST_END_DAY:
                    test_targets.append(
                        targets.select("client_id", "timestamp", "item_id")
                    )

                daily_samples = (
                    _daily_basis_samples(targets, day, domain=domain)
                    if not targets.is_empty()
                    else None
                )
                for mode, state in states.items():
                    history = known if mode == ALL_EVENTS_MODE else positives
                    if not history.is_empty():
                        _append_eventhub_partition(
                            state["eventhub"],
                            history,
                            event_names=spec["event_names"],
                        )
                    state["history_rows"] += len(history)
                    counts = dict(history.group_by("action").len().iter_rows())
                    for action in state["action_rows"]:
                        state["action_rows"][action] += int(counts.get(action, 0))
                    day_users = set(history["client_id"].unique().to_list())

                    if day <= TRAIN_END_DAY:
                        if daily_samples is not None:
                            eligible = _filter_seen_users(
                                daily_samples,
                                state["train_seen_users"],
                            )
                            if not eligible.is_empty():
                                table = eligible.to_arrow()
                                if state["train_writer"] is None:
                                    state["train_writer"] = pq.ParquetWriter(
                                        state["variant"] / "basis/train/samples.pq",
                                        table.schema,
                                        compression="zstd",
                                    )
                                state["train_writer"].write_table(table)
                                state["train_rows"] += len(eligible)
                                state["train_target_items"] += int(
                                    eligible["target"].list.len().sum()
                                )
                                _update_target_digest(state["train_digest"], eligible)
                        state["train_seen_users"].update(day_users)
                    if day <= VALIDATION_END_DAY:
                        state["test_history_users"].update(day_users)

                logger.info(
                    "Full %s Table-6 partition %d/%d %s: raw=%d known=%d "
                    "positive=%d targets=%d elapsed=%.1fs",
                    spec["title"],
                    index,
                    len(event_paths),
                    path.name,
                    len(raw),
                    len(known),
                    len(positives),
                    len(targets),
                    time.monotonic() - partition_started,
                )
                if delete_source_files:
                    discard_files([path], logger=logger)
        finally:
            for state in states.values():
                if state["train_writer"] is not None:
                    state["train_writer"].close()

        if not validation_targets or not test_targets:
            raise ValueError(f"Full {spec['title']} holdout target window is empty")
        representations = _write_representations(
            output / "representations",
            items,
            train_positive_item_ids,
            embedding_dimension=embedding_dimension,
            domain=domain,
        )

        validation_events = pl.concat(validation_targets)
        test_events = pl.concat(test_targets)
        for mode, state in states.items():
            if state["train_writer"] is None:
                raise ValueError(f"Full {spec['title']} {mode} train basis is empty")
            holdout_state = states[POSITIVE_MODE] if shared_holdout else state
            validation = _window_basis_samples(
                validation_events,
                timestamp=_day_start(VALIDATION_START_DAY),
                history_timestamp=_day_start(VALIDATION_GAP_DAY),
            ).join(
                pl.DataFrame({"client_id": list(holdout_state["train_seen_users"])}),
                on="client_id",
                how="semi",
            )
            test = _window_basis_samples(
                test_events,
                timestamp=_day_start(TEST_START_DAY),
                history_timestamp=_day_start(TEST_GAP_DAY),
            ).join(
                pl.DataFrame({"client_id": list(holdout_state["test_history_users"])}),
                on="client_id",
                how="semi",
            )
            _validate_basis_samples(domain, "validation", validation)
            _validate_basis_samples(domain, "test", test)
            validation.write_parquet(state["variant"] / "basis/validation/samples.pq")
            test.write_parquet(state["variant"] / "basis/test/samples.pq")

            eventhub_files = sum(
                path.is_file() for path in state["eventhub"].rglob("*.pq")
            )
            archive_eventhub(basis=state["variant"])
            archive_path = state["variant"] / "eventhub.tar"
            history_actions = (
                positive_actions if mode == POSITIVE_MODE else supported_actions
            )
            report = {
                "dataset": f"T-ECD Full {spec['title']}",
                "domain": domain,
                "source_repository": DATASET_REPOSITORY,
                "source_revision": source_revision,
                "license": "CC-BY-NC-SA-4.0",
                "protocol": f"article-window-{domain}-sd-{mode}-daily-train",
                "history_mode": mode,
                "history_actions": list(history_actions),
                "like_actions": list(spec["like_actions"]),
                "target_actions": list(positive_actions),
                "views_included": mode == ALL_EVENTS_MODE,
                "views_are_targets": False,
                "shared_holdout": shared_holdout,
                "train_days": [TRAIN_START_DAY, TRAIN_END_DAY],
                "validation_gap_day": VALIDATION_GAP_DAY,
                "validation_days": [VALIDATION_START_DAY, VALIDATION_END_DAY],
                "test_gap_day": TEST_GAP_DAY,
                "test_days": [TEST_START_DAY, TEST_END_DAY],
                "train_sample": "one user-day with unique positive item targets",
                "validation_sample": (
                    "one user with unique positive targets over seven days"
                ),
                "test_sample": (
                    "one user with unique positive targets over seven days"
                ),
                "target_mode": "all_targets_independent",
                "target_reduction": "mean_tokens_then_targets_global",
                "source": {
                    "event_partitions": len(event_paths),
                    "event_compressed_bytes": event_compressed_bytes,
                    "items_compressed_bytes": items_compressed_bytes,
                },
                "candidate_policy": (f"all items from Full {spec['title']} items.pq"),
                "warm_item_filter": False,
                "rows": {
                    "train": state["train_rows"],
                    "validation": len(validation),
                    "test": len(test),
                },
                "target_items": {
                    "train": state["train_target_items"],
                    "validation": int(validation["target"].list.len().sum()),
                    "test": int(test["target"].list.len().sum()),
                },
                "target_digest": {
                    "train": state["train_digest"].hexdigest(),
                    "validation": _sample_target_digest(validation),
                    "test": _sample_target_digest(test),
                },
                "supported_event_rows": raw_supported_rows,
                "positive_event_rows": positive_rows,
                "target_eligible_event_rows": target_rows,
                "history_event_rows": state["history_rows"],
                "unknown_item_rows_dropped": unknown_item_rows,
                "event_rows": state["action_rows"],
                "candidate_items": len(candidate_catalog),
                "train_visible_items": len(train_positive_item_ids),
                "validation_cold_items": _cold_target_items(
                    validation, train_positive_item_ids
                ),
                "test_cold_items": _cold_target_items(test, train_positive_item_ids),
                "brand_ids": candidate_catalog["brand_id"].drop_nulls().n_unique(),
                "representations": representations,
                "eventhub_files_before_archive": eventhub_files,
                "eventhub_archive_bytes": archive_path.stat().st_size,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            (state["variant"] / "report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    logger.info(
        "Full %s Table-6 bases completed in %.1f minutes: modes=%s",
        spec["title"],
        (time.monotonic() - started) / 60,
        modes,
    )


@app.command
@logged_stage
def materialize_full_table6_basis(
    *,
    config: Path,
    basis: Path,
    sid_mapping: Path,
    workdir: Path,
) -> None:
    """Bind one Table-6 basis to its frozen complete-catalog SID mapping."""

    recipe = Config.load_data(config)
    report = json.loads((basis / "report.json").read_text(encoding="utf-8"))
    item_features = pl.read_parquet(basis / "item_features.pq").select(
        pl.col("item_id").cast(pl.String),
        pl.col("brand_id").cast(pl.String),
    )
    if item_features["item_id"].null_count() or item_features[
        "item_id"
    ].n_unique() != len(item_features):
        raise ValueError("T-ECD Full catalog requires unique non-null item_id")

    mapping = pl.read_parquet(sid_mapping).with_columns(
        pl.col("item_id").cast(pl.String)
    )
    sid = SIDPreprocessor(mapping, item_id_column="item_id")
    missing = item_features.select("item_id").join(
        mapping.select("item_id"), on="item_id", how="anti"
    )
    extra = mapping.select("item_id").join(
        item_features.select("item_id"), on="item_id", how="anti"
    )
    if len(missing) or len(extra):
        raise ValueError(
            "T-ECD Full SID mapping and candidate catalog differ: "
            f"missing={len(missing)} extra={len(extra)}"
        )

    protocol = {
        "name": report["protocol"],
        "source_revision": report["source_revision"],
        "history_mode": report["history_mode"],
        "history_actions": report["history_actions"],
        "target_actions": report["target_actions"],
        "views_included": report["views_included"],
        "views_are_targets": report["views_are_targets"],
        "train_days": report["train_days"],
        "validation_gap_day": report["validation_gap_day"],
        "validation_days": report["validation_days"],
        "test_gap_day": report["test_gap_day"],
        "test_days": report["test_days"],
        "target_digest": report["target_digest"],
        "candidate_policy": report["candidate_policy"],
        "warm_item_filter": report["warm_item_filter"],
        "sid_fit_scope": "train-visible positive items with source embeddings",
        "sid_apply_scope": "complete domain candidate catalog",
        "sid_width": sid.code_length,
        "query_aggregation": {"train": "user-day", "holdout": "user-seven-day-window"},
    }

    with publish_directory(workdir, overwrite=False) as output:
        resources = output / "resources"
        resources.mkdir()
        mapping.sort("item_id").write_parquet(resources / "item_to_sid.parquet")
        (resources / "manifest.json").write_text(
            json.dumps({"item_to_sid": "item_to_sid.parquet"}, indent=2) + "\n",
            encoding="utf-8",
        )
        for split in ("train", "validation", "test"):
            source = basis / "basis" / split
            destination = output / "basis" / split
            (destination / "artifacts").mkdir(parents=True)
            catalog = pl.read_parquet(source / "artifacts/items.pq").select(
                pl.col("item_id").cast(pl.String),
                pl.col("brand_id").cast(pl.String),
            )
            if not catalog.sort("item_id").equals(item_features.sort("item_id")):
                raise ValueError(
                    f"T-ECD Full {split} catalog differs from item_features.pq"
                )
            copy_resource(source / "samples.pq", destination / "samples.pq")
            catalog.write_parquet(destination / "artifacts/items.pq")

        finalize_bundle(
            output,
            recipe,
            dataset=report["dataset"],
            source={
                "repository": report["source_repository"],
                "revision": report["source_revision"],
            },
            protocol=protocol,
        )
        provenance = {
            "dataset": report["dataset"],
            "domain": report["domain"],
            "protocol": protocol,
            "rows": report["rows"],
            "target_items": report["target_items"],
            "candidate_items": report["candidate_items"],
            "train_visible_items": report["train_visible_items"],
            "validation_cold_items": report["validation_cold_items"],
            "test_cold_items": report["test_cold_items"],
            "brand_ids": report["brand_ids"],
            "representations": report["representations"],
            "sid_items": len(mapping),
            "sid_vocabulary_size": sid.vocab_size,
        }
        (output / "provenance.json").write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def _filter_seen_users(
    samples: pl.DataFrame,
    seen_users: set[str],
) -> pl.DataFrame:
    """Keep users with earlier history without rebuilding a huge lookup table."""

    if not seen_users or samples.is_empty():
        return samples.head(0)
    mask = pl.Series(
        "eligible",
        (client_id in seen_users for client_id in samples["client_id"]),
        dtype=pl.Boolean,
    )
    return samples.filter(mask)


def _load_catalog(
    path: Path,
    *,
    domain: str,
    require_brand: bool,
) -> tuple[pl.DataFrame, int]:
    items = pl.read_parquet(path, columns=["item_id", "brand_id", "embedding"]).select(
        pl.col("item_id").cast(pl.String),
        pl.col("brand_id").cast(pl.String),
        "embedding",
    )
    if items["item_id"].null_count() or items["item_id"].n_unique() != len(items):
        raise ValueError(f"Full {domain} catalog requires unique non-null item_id")
    if require_brand and items["brand_id"].null_count():
        raise ValueError(f"Full {domain} catalog requires non-null brand_id")
    embedding_type = items.schema["embedding"]
    available = items["embedding"].drop_nulls()
    if isinstance(embedding_type, pl.Array):
        lengths = available.arr.len().unique().to_list()
    elif isinstance(embedding_type, pl.List):
        lengths = available.list.len().unique().to_list()
    else:
        lengths = []
    if len(lengths) != 1 or lengths[0] is None or lengths[0] < 1:
        raise ValueError(
            f"Full {domain} embeddings require one positive dimension: {lengths}"
        )
    return items.sort("item_id"), int(lengths[0])


def _read_partition(path: Path, actions: tuple[str, ...]) -> pl.DataFrame:
    return (
        pl.read_parquet(
            path,
            columns=["timestamp", "user_id", "item_id", "action_type"],
        )
        .select(
            pl.col("timestamp")
            .dt.total_microseconds()
            .cast(pl.Datetime("us"))
            .dt.truncate("1s")
            .cast(pl.Datetime("ns")),
            pl.col("user_id").cast(pl.String).alias("client_id"),
            pl.col("item_id").cast(pl.String),
            pl.col("action_type").cast(pl.String).alias("action"),
        )
        .drop_nulls(["timestamp", "client_id", "item_id", "action"])
        .filter(pl.col("action").is_in(actions))
        .unique()
    )


def _day_start(day: int) -> datetime:
    return datetime.fromtimestamp(day * 86400, tz=UTC).replace(tzinfo=None)


def _targets(events: pl.DataFrame) -> pl.DataFrame:
    return (
        events.select("client_id", "timestamp", "item_id")
        .sort("client_id", "timestamp", "item_id")
        .unique(subset=["client_id", "item_id"], keep="first", maintain_order=True)
        .group_by("client_id", maintain_order=True)
        .agg(
            pl.col("item_id").alias("target"),
            pl.col("timestamp").alias("target_ts"),
        )
        .with_columns(
            pl.col("target")
            .list.eval(pl.lit(1.0, dtype=pl.Float32))
            .alias("target_labels")
        )
    )


def _daily_basis_samples(
    events: pl.DataFrame,
    day: int,
    *,
    domain: str,
) -> pl.DataFrame:
    timestamp = _day_start(day)
    if events.filter(
        (pl.col("timestamp") < timestamp)
        | (pl.col("timestamp") >= timestamp + timedelta(days=1))
    ).height:
        raise ValueError(f"{domain} partition {day:05d}.pq contains another day")
    return (
        _targets(events)
        .with_columns(pl.lit(timestamp).cast(pl.Datetime("ns")).alias("timestamp"))
        .with_columns(
            pl.struct("client_id", "timestamp").hash(seed=0).alias("sample_id")
        )
        .select(
            "sample_id",
            "client_id",
            "timestamp",
            "target",
            "target_labels",
            "target_ts",
        )
    )


def _window_basis_samples(
    events: pl.DataFrame,
    *,
    timestamp: datetime,
    history_timestamp: datetime,
) -> pl.DataFrame:
    return (
        _targets(events)
        .with_columns(
            pl.lit(timestamp).cast(pl.Datetime("ns")).alias("timestamp"),
            pl.lit(history_timestamp)
            .cast(pl.Datetime("ns"))
            .alias("history_timestamp"),
        )
        .with_columns(
            pl.struct("client_id", "timestamp").hash(seed=0).alias("sample_id")
        )
        .select(
            "sample_id",
            "client_id",
            "timestamp",
            "history_timestamp",
            "target",
            "target_labels",
            "target_ts",
        )
        .sort("client_id")
    )


def _validate_basis_samples(
    domain: str,
    split: str,
    samples: pl.DataFrame,
) -> None:
    if samples.is_empty():
        raise ValueError(f"Full {domain} {split} basis is empty")
    if samples["sample_id"].n_unique() != len(samples):
        raise ValueError(f"Full {domain} {split} sample_id values are not unique")
    if samples["client_id"].null_count() or samples["target"].list.len().min() < 1:
        raise ValueError(f"Full {domain} {split} has invalid users or targets")
    for column in ("target_ts", "target_labels"):
        if samples.filter(
            pl.col("target").list.len() != pl.col(column).list.len()
        ).height:
            raise ValueError(
                f"Full {domain} {split} target and {column} lengths differ"
            )
    if samples.filter(
        pl.col("timestamp") - pl.col("history_timestamp") != pl.duration(days=1)
    ).height:
        raise ValueError(f"Full {domain} {split} history gap is not one day")


def _append_eventhub_partition(
    root: Path,
    events: pl.DataFrame,
    *,
    event_names: dict[str, str],
) -> None:
    root.mkdir(exist_ok=True)
    values = distribute_to_partitions(
        events.with_columns(
            pl.col("action").replace_strict(event_names).alias("event"),
            pl.col("timestamp").dt.date().alias("date"),
        )
    )
    for (event, day, partition), frame in values.group_by(
        "event", "date", "partition_id", maintain_order=True
    ):
        destination = root / str(event) / day.isoformat()
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / partition_filename(int(partition))
        payload = frame.select("client_id", "timestamp", "item_id", "brand_id")
        if path.exists():
            payload = pl.concat([pl.read_parquet(path), payload]).unique()
        payload.sort("client_id", "timestamp", "item_id").write_parquet(path)


def _write_representations(
    root: Path,
    items: pl.DataFrame,
    train_item_ids: set[str],
    *,
    embedding_dimension: int,
    domain: str,
) -> dict[str, int | str]:
    train_ids = pl.DataFrame({"item_id": sorted(train_item_ids)})
    train = train_ids.join(items, on="item_id", how="left", validate="1:1")
    missing_train = train["embedding"].null_count()
    train = train.filter(pl.col("embedding").is_not_null())
    if train.is_empty():
        raise ValueError(f"Full {domain} train items have no embeddings")
    missing_all = items["embedding"].null_count()
    zero = pl.lit(
        [0.0] * embedding_dimension,
        dtype=items.schema["embedding"],
    )
    root.mkdir()
    for name, frame in (("all", items), ("train", train)):
        frame = frame.sort("item_id")
        destination = root / name
        destination.mkdir()
        frame.select("item_id").write_parquet(destination / "items.pq")
        _write_embedding_matrix(
            destination / "embeddings.npy",
            frame,
            embedding_dimension=embedding_dimension,
            zero=zero,
            label=f"{domain}-{name}",
        )
    return {
        "dimension": embedding_dimension,
        "sid_fit_items": len(train),
        "sid_fit_missing_excluded": missing_train,
        "frozen_apply_items": len(items),
        "frozen_apply_zero_filled": missing_all,
        "missing_policy": "exclude-from-fit-zero-fill-for-apply",
    }


def _write_embedding_matrix(
    path: Path,
    frame: pl.DataFrame,
    *,
    embedding_dimension: int,
    zero: pl.Expr,
    label: str,
) -> None:
    output = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(len(frame), embedding_dimension),
    )
    try:
        for start in range(0, len(frame), REPRESENTATION_BATCH_SIZE):
            batch = frame.slice(start, REPRESENTATION_BATCH_SIZE)
            embedding = pl.col("embedding").fill_null(zero)
            if isinstance(frame.schema["embedding"], pl.List):
                embedding = embedding.list.to_array(embedding_dimension)
            values = np.asarray(
                batch.select(embedding)["embedding"].to_numpy(),
                dtype=np.float32,
            )
            expected = (len(batch), embedding_dimension)
            if values.shape != expected or not np.isfinite(values).all():
                raise ValueError(
                    f"invalid {label} embedding batch at row {start}: {values.shape}"
                )
            output[start : start + len(batch)] = values
    finally:
        output.flush()
        del output


def _update_target_digest(digest: Any, samples: pl.DataFrame) -> None:
    rows = (
        samples.select("client_id", "target", "target_ts")
        .explode("target", "target_ts")
        .sort("target_ts", "client_id", "target")
    )
    for client_id, item_id, timestamp in rows.iter_rows():
        digest.update(f"{client_id}\0{timestamp.isoformat()}\0{item_id}\n".encode())


def _sample_target_digest(samples: pl.DataFrame) -> str:
    digest = hashlib.sha256()
    _update_target_digest(digest, samples)
    return digest.hexdigest()


def _cold_target_items(samples: pl.DataFrame, train_items: set[str]) -> int:
    targets = samples.select(pl.col("target").explode().alias("item_id")).unique()
    known = pl.DataFrame({"item_id": sorted(train_items)})
    return len(targets.join(known, on="item_id", how="anti"))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app()
