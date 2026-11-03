"""Prepare the T-ECD Small Marketplace single-domain Perseus comparison."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from data_sources.event_hub import distribute_to_partitions, partition_filename
from genrecsys.utils.files import publish_directory
from model_datasets.tecd.retail import (
    DATASET_REPOSITORY,
    HISTORY_GAP_HOURS,
    ITEM_FEATURES,
    USERS_PATH,
    _basis_target_digest,
    _filter_active_clients,
    _prepare_basis,
    _redistribute_target_timestamps,
    _samples_with_context,
    _target_digest,
    _validate_protocol,
)

logger = logging.getLogger(__name__)

MARKETPLACE_ROOT = Path("dataset/small/marketplace")
TARGET_ACTION = "click"
EVENT_NAMES = {
    "view": "t-ecd-small-marketplace-view",
    "click": "t-ecd-small-marketplace-click",
    "like": "t-ecd-small-marketplace-like",
    "clickout": "t-ecd-small-marketplace-clickout",
}


def download_small_marketplace(
    *,
    output_dir: Path,
    revision: str = "main",
) -> None:
    """Download T-ECD Small Marketplace events, catalog, and user context."""

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=DATASET_REPOSITORY,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[
            f"{MARKETPLACE_ROOT}/events/*.pq",
            f"{MARKETPLACE_ROOT}/items.pq",
            str(USERS_PATH),
        ],
        local_dir=str(output_dir),
    )


def download_and_build_small_marketplace(
    *,
    output_dir: Path,
    revision: str = "main",
    history_gap_hours: int = HISTORY_GAP_HOURS,
    include_views: bool = True,
) -> None:
    """Download Marketplace into temporary storage and publish its adapter output."""

    with tempfile.TemporaryDirectory(prefix="tecd-small-marketplace-") as temporary:
        source = Path(temporary)
        download_small_marketplace(output_dir=source, revision=revision)
        build_small_marketplace(
            source_dir=source,
            output_dir=output_dir,
            source_revision=revision,
            history_gap_hours=history_gap_hours,
            include_views=include_views,
        )


def build_small_marketplace(
    *,
    source_dir: Path,
    output_dir: Path,
    source_revision: str = "main",
    history_gap_hours: int = HISTORY_GAP_HOURS,
    include_views: bool = True,
    train_ratio: float = 0.7,
) -> None:
    """Build Marketplace splits preserving Perseus's final 20% test."""

    if history_gap_hours < 0:
        raise ValueError("history_gap_hours must be non-negative")

    interactions, items, users = _load_source(source_dir)
    items, item_catalog_report = _complete_item_catalog(items, interactions)
    target_events = _redistribute_target_timestamps(
        _filter_active_clients(interactions.filter(pl.col("action") == TARGET_ACTION))
    )
    other_events = interactions.filter(pl.col("action") != TARGET_ACTION)
    if not include_views:
        other_events = other_events.filter(pl.col("action") != "view")
    train_basis, validation_basis, test_basis, candidate_ids, reference = (
        _prepare_basis(target_events, train_ratio=train_ratio)
    )
    samples = {
        split: _samples_with_context(
            frame,
            users,
            history_gap_hours=history_gap_hours,
        )
        for split, frame in {
            "train": train_basis,
            "validation": validation_basis,
            "test": test_basis,
        }.items()
    }
    _validate_protocol(
        samples,
        reference,
        history_gap_hours=history_gap_hours,
    )

    candidate_catalog = candidate_ids.join(
        items.drop("embedding").with_columns(_catalog_match=pl.lit(True)),
        on="item_id",
        how="left",
        validate="1:1",
    ).sort("item_id")
    if candidate_catalog["_catalog_match"].null_count():
        raise AssertionError("completed Marketplace item catalog misses candidates")
    candidate_catalog = candidate_catalog.drop("_catalog_match")

    retained_targets = pl.concat([train_basis, validation_basis, test_basis]).select(
        "client_id", "target_timestamp", "item_id"
    )
    retained_clicks = target_events.join(
        retained_targets,
        left_on=["client_id", "timestamp", "item_id"],
        right_on=["client_id", "target_timestamp", "item_id"],
        how="semi",
    )
    eventhub_events = pl.concat([retained_clicks, other_events], how="vertical").join(
        items.select("item_id", "brand_id"),
        on="item_id",
        how="left",
        validate="m:1",
    )
    train_item_ids = (
        samples["train"].select(pl.col("target").explode().alias("item_id")).unique()
    )
    with publish_directory(output_dir, overwrite=False) as output:
        for split, frame in samples.items():
            destination = output / "basis" / split
            (destination / "artifacts").mkdir(parents=True)
            frame.write_parquet(destination / "samples.pq")
            candidate_catalog.write_parquet(destination / "artifacts" / "items.pq")
        items.drop("embedding").write_parquet(output / "item_features.pq")
        embedding_report = _write_representations(
            output / "representations", items, train_item_ids
        )
        _write_eventhub(output / "eventhub", eventhub_events)
        report = {
            "dataset": "T-ECD Small Marketplace",
            "source_repository": DATASET_REPOSITORY,
            "source_revision": source_revision,
            "license": "CC-BY-NC-SA-4.0",
            "protocol": (
                "perseus-repro-with-validation"
                if history_gap_hours == 0
                else (
                    "perseus-repro-with-validation-and-12h-history-gap"
                    if history_gap_hours == HISTORY_GAP_HOURS
                    else "perseus-repro-with-validation-and-history-gap"
                )
            ),
            "target_action": TARGET_ACTION,
            "model_target_event": EVENT_NAMES[TARGET_ACTION],
            "split": {
                "train_ratio": train_ratio,
                "validation_ratio": round(0.8 - train_ratio, 10),
                "test_ratio": 0.2,
                "test_created_before_validation_split": True,
            },
            "history_gap_hours": history_gap_hours,
            "includes_views": include_views,
            "candidate_policy": "public Perseus retained 80% development target items",
            "holdout_targets_filtered": True,
            "sid_fit_scope": (
                f"{train_ratio:.0%} train target items with non-null source embeddings only"
            ),
            "test_target_digest": _target_digest(samples["test"]),
            "perseus_reference_test_target_digest": _basis_target_digest(reference),
            "rows": {split: len(frame) for split, frame in samples.items()},
            "candidate_items": len(candidate_catalog),
            "full_catalog_items": len(items),
            "item_catalog": item_catalog_report,
            "embeddings": embedding_report,
            "event_rows": {
                action: eventhub_events.filter(pl.col("action") == action).height
                for action in EVENT_NAMES
            },
        }
        (output / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def _load_source(source_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Read and normalize Marketplace events, item embeddings, and users."""

    root = source_dir / MARKETPLACE_ROOT
    events_path = root / "events"
    items_path = root / "items.pq"
    users_path = source_dir / USERS_PATH
    for path in (events_path, items_path, users_path):
        if not path.exists():
            raise FileNotFoundError(f"T-ECD source path does not exist: {path}")
    event_paths = sorted(events_path.glob("*.pq"))
    if not event_paths:
        raise FileNotFoundError(f"Marketplace events do not exist in: {events_path}")
    source_size_gib = sum(path.stat().st_size for path in event_paths) / 1024**3
    logger.info(
        "Reading %d Marketplace event partitions (%.2f GiB compressed)",
        len(event_paths),
        source_size_gib,
    )
    interactions = (
        pl.scan_parquet([str(path) for path in event_paths])
        .select(
            pl.col("timestamp")
            .dt.total_microseconds()
            .cast(pl.Datetime("us"))
            .dt.truncate("1s")
            .cast(pl.Datetime("ns")),
            pl.col("user_id").cast(pl.String).alias("client_id"),
            pl.col("item_id").cast(pl.String),
            pl.col("action_type").cast(pl.String).alias("action"),
            pl.col("subdomain").cast(pl.String),
            pl.col("os").cast(pl.String),
        )
        .drop_nulls(["timestamp", "client_id", "item_id", "action"])
        .filter(pl.col("action").is_in(list(EVENT_NAMES)))
        .unique()
        .collect(engine="streaming")
    )
    logger.info("Loaded and normalized %d Marketplace interactions", len(interactions))
    items = (
        pl.read_parquet(items_path, columns=["item_id", *ITEM_FEATURES, "embedding"])
        .with_columns(
            pl.col("item_id").cast(pl.String),
            pl.col("brand_id").cast(pl.String),
            pl.col("category").cast(pl.String),
            pl.col("subcategory").cast(pl.String),
            pl.col("price").cast(pl.Float32),
        )
        .sort("item_id")
    )
    if items["item_id"].null_count() or items["item_id"].n_unique() != len(items):
        raise ValueError("Marketplace catalog requires unique non-null item_id")
    _embedding_dimension(items)
    users = pl.read_parquet(
        users_path,
        columns=["user_id", "socdem_cluster", "region"],
    ).select(
        pl.col("user_id").cast(pl.String).alias("client_id"),
        pl.col("socdem_cluster").cast(pl.String),
        pl.col("region").cast(pl.String),
    )
    if users["client_id"].null_count() or users["client_id"].n_unique() != len(users):
        raise ValueError("T-ECD users require unique non-null user_id")
    logger.info("Loaded %d Marketplace items and %d users", len(items), len(users))
    return interactions, items, users


def _embedding_dimension(items: pl.DataFrame) -> int:
    """Validate the available item embeddings and return their common dimension."""

    dtype = items.schema["embedding"]
    available = items["embedding"].drop_nulls()
    if isinstance(dtype, pl.Array):
        lengths = available.arr.len().unique().to_list()
    elif isinstance(dtype, pl.List):
        lengths = available.list.len().unique().to_list()
    else:
        lengths = []
    if len(lengths) != 1 or lengths[0] is None or lengths[0] < 1:
        raise ValueError(
            "Marketplace non-null embeddings must have one positive dimension: "
            f"{lengths}"
        )
    return int(lengths[0])


def _complete_item_catalog(
    items: pl.DataFrame,
    interactions: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, int | str]]:
    """Preserve event items absent from the published catalog for later imputation."""

    dimension = _embedding_dimension(items)
    event_items = interactions.select("item_id").unique().sort("item_id")
    missing = event_items.join(items.select("item_id"), on="item_id", how="anti")
    report: dict[str, int | str] = {
        "source_items": len(items),
        "source_null_embeddings": items["embedding"].null_count(),
        "event_items": len(event_items),
        "imputed_items": len(missing),
        "imputation": f"null-features-null-{dimension}d-embedding",
    }
    if missing.is_empty():
        return items, report
    logger.warning(
        "Marketplace catalog misses %d/%d event items; examples=%s",
        len(missing),
        len(event_items),
        missing["item_id"].head(10).to_list(),
    )
    fallback = missing.with_columns(
        pl.lit(None, dtype=pl.String).alias("brand_id"),
        pl.lit(None, dtype=pl.String).alias("category"),
        pl.lit(None, dtype=pl.String).alias("subcategory"),
        pl.lit(None, dtype=pl.Float32).alias("price"),
        pl.lit(None, dtype=items.schema["embedding"]).alias("embedding"),
    ).select(items.columns)
    return pl.concat([items, fallback], how="vertical").sort("item_id"), report


def _write_representations(
    root: Path,
    items: pl.DataFrame,
    train_item_ids: pl.DataFrame,
) -> dict[str, int | str]:
    """Fit on available train vectors and zero-fill missing vectors only for apply."""

    dimension = _embedding_dimension(items)
    train = train_item_ids.join(items, on="item_id", how="left", validate="1:1")
    missing_train = train["embedding"].null_count()
    train = train.filter(pl.col("embedding").is_not_null())
    if train.is_empty():
        raise ValueError("Marketplace train targets have no non-null embeddings")
    missing_all = items["embedding"].null_count()
    zero = pl.lit(
        [0.0] * dimension,
        dtype=items.schema["embedding"],
    )
    root.mkdir()
    for name, frame in (
        ("all", items),
        ("train", train),
    ):
        embedding = pl.col("embedding").fill_null(zero)
        if isinstance(items.schema["embedding"], pl.List):
            embedding = embedding.list.to_array(dimension)
        frame = frame.sort("item_id").with_columns(embedding)
        destination = root / name
        destination.mkdir()
        frame.select("item_id").write_parquet(destination / "items.pq")
        matrix = np.asarray(frame["embedding"].to_numpy(), dtype=np.float32)
        if matrix.shape != (len(frame), dimension) or not np.isfinite(matrix).all():
            raise ValueError(f"invalid {name} Marketplace embeddings: {matrix.shape}")
        np.save(destination / "embeddings.npy", matrix, allow_pickle=False)
    logger.warning(
        "Marketplace SID fit excludes %d/%d train items without embeddings; "
        "frozen apply zero-fills %d/%d catalog embeddings",
        missing_train,
        len(train) + missing_train,
        missing_all,
        len(items),
    )
    return {
        "dimension": dimension,
        "sid_fit_items": len(train),
        "sid_fit_missing_excluded": missing_train,
        "frozen_apply_items": len(items),
        "frozen_apply_zero_filled": missing_all,
        "missing_policy": "exclude-from-fit-zero-fill-for-apply",
    }


def _write_eventhub(root: Path, events: pl.DataFrame) -> None:
    """Publish local EventHub-compatible action/date/hash partitions."""

    root.mkdir()
    values = distribute_to_partitions(
        events.with_columns(
            pl.col("action").replace_strict(EVENT_NAMES).alias("event"),
            pl.col("timestamp").dt.date().alias("date"),
        )
    )
    for (event, day, partition), frame in values.group_by(
        "event", "date", "partition_id", maintain_order=True
    ):
        destination = root / str(event) / day.isoformat()
        destination.mkdir(parents=True, exist_ok=True)
        frame.select(
            "client_id", "timestamp", "item_id", "subdomain", "os", "brand_id"
        ).sort("client_id", "timestamp", "item_id").write_parquet(
            destination / partition_filename(int(partition))
        )


__all__ = [
    "build_small_marketplace",
    "download_and_build_small_marketplace",
    "download_small_marketplace",
]
