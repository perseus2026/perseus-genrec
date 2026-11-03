"""Prepare the T-ECD Small Retail single-domain Perseus comparison."""

from __future__ import annotations

import json
import logging
import shutil
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from data_sources.event_hub import distribute_to_partitions, partition_filename
from genrecsys.utils.files import discard_files, publish_directory
from genrecsys.utils.progress import logged_stage

logger = logging.getLogger(__name__)

DATASET_REPOSITORY = "t-tech/T-ECD"
RETAIL_ROOT = Path("dataset/small/retail")
USERS_PATH = Path("dataset/small/users.pq")
HISTORY_GAP_HOURS = 12
TARGET_ACTION = "order"
EVENT_NAMES = {
    "view": "t-ecd-small-retail-view",
    "click": "t-ecd-small-retail-click",
    "added_to_cart": "t-ecd-small-retail-added-to-cart",
    # Name the event optimized as the target `like`.
    "order": "t-ecd-small-retail-like",
}
ITEM_FEATURES = ("brand_id", "category", "subcategory", "price")
EVENTHUB_ARCHIVE = "eventhub.tar"


def download_small_retail(
    *,
    output_dir: Path,
    revision: str = "main",
) -> None:
    """Download the complete T-ECD Small Retail source, including item embeddings."""

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=DATASET_REPOSITORY,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[
            f"{RETAIL_ROOT}/events/*.pq",
            f"{RETAIL_ROOT}/items.pq",
            str(USERS_PATH),
        ],
        local_dir=str(output_dir),
        max_workers=1,
    )


def download_and_build_small_retail(
    *,
    output_dir: Path,
    revision: str = "main",
    history_gap_hours: int = HISTORY_GAP_HOURS,
    include_views: bool = True,
) -> None:
    """Download T-ECD Small Retail into temporary storage and publish its adapter output."""

    with tempfile.TemporaryDirectory(prefix="tecd-small-retail-") as temporary:
        source = Path(temporary)
        download_small_retail(output_dir=source, revision=revision)
        build_small_retail(
            source_dir=source,
            output_dir=output_dir,
            source_revision=revision,
            history_gap_hours=history_gap_hours,
            include_views=include_views,
        )


@logged_stage
def archive_eventhub(*, basis: Path) -> None:
    """Pack EventHub into one uncompressed transport object and remove its source."""

    eventhub = basis / "eventhub"
    archive = basis / EVENTHUB_ARCHIVE
    if not eventhub.is_dir():
        raise FileNotFoundError(f"T-ECD EventHub directory does not exist: {eventhub}")
    if archive.exists():
        raise FileExistsError(f"T-ECD EventHub archive already exists: {archive}")
    file_count = sum(path.is_file() for path in eventhub.rglob("*"))
    if not file_count:
        raise ValueError(f"T-ECD EventHub directory is empty: {eventhub}")

    with tarfile.open(archive, mode="w") as output:
        output.add(eventhub, arcname="eventhub")
    shutil.rmtree(eventhub)
    logger.info(
        "Archived %d EventHub files into %s (%.2f GiB)",
        file_count,
        archive,
        archive.stat().st_size / 1024**3,
    )


@logged_stage
def extract_eventhub(*, basis: Path) -> None:
    """Restore the local EventHub tree from its single-object transport archive."""

    eventhub = basis / "eventhub"
    archive = basis / EVENTHUB_ARCHIVE
    if eventhub.exists():
        raise FileExistsError(f"T-ECD EventHub directory already exists: {eventhub}")
    if not archive.is_file():
        raise FileNotFoundError(f"T-ECD EventHub archive does not exist: {archive}")

    with tarfile.open(archive, mode="r") as source:
        source.extractall(path=basis, filter="data")
    if not eventhub.is_dir():
        raise ValueError(f"T-ECD EventHub archive has no eventhub directory: {archive}")
    discard_files([archive], logger=logger)
    logger.info("Extracted T-ECD EventHub into %s", eventhub)


def build_small_retail(
    *,
    source_dir: Path,
    output_dir: Path,
    source_revision: str = "main",
    history_gap_hours: int = HISTORY_GAP_HOURS,
    include_views: bool = True,
    train_ratio: float = 0.7,
) -> None:
    """Build Retail splits while preserving Perseus's final 20% test.

    Reproduce the public 80/20 target filtering first. Split its retained development
    pool at the requested global train timestamp only afterwards. The test target rows
    therefore stay identical to the public preprocessing. ``history_gap_hours=0``
    reproduces Perseus's history cutoff at the target timestamp.
    """

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
    missing = candidate_catalog.filter(pl.col("_catalog_match").is_null())
    if len(missing):
        raise AssertionError("completed Retail item catalog does not cover candidates")
    candidate_catalog = candidate_catalog.drop("_catalog_match")
    retained_targets = pl.concat([train_basis, validation_basis, test_basis]).select(
        "client_id", "target_timestamp", "item_id"
    )
    retained_order_events = target_events.join(
        retained_targets,
        left_on=["client_id", "timestamp", "item_id"],
        right_on=["client_id", "target_timestamp", "item_id"],
        how="semi",
    )
    eventhub_events = pl.concat(
        [retained_order_events, other_events], how="vertical"
    ).join(
        items.select("item_id", "brand_id"),
        on="item_id",
        how="left",
        validate="m:1",
    )
    train_item_ids = (
        samples["train"].select(pl.col("target").explode().alias("item_id")).unique()
    )
    test_target_digest = _target_digest(samples["test"])
    reference_digest = _basis_target_digest(reference)
    with publish_directory(output_dir, overwrite=False) as output:
        for split, frame in samples.items():
            destination = output / "basis" / split
            (destination / "artifacts").mkdir(parents=True)
            frame.write_parquet(destination / "samples.pq")
            candidate_catalog.write_parquet(destination / "artifacts" / "items.pq")
        items.drop("embedding").write_parquet(output / "item_features.pq")
        _write_representations(output / "representations", items, train_item_ids)
        _write_eventhub(output / "eventhub", eventhub_events)
        report = {
            "dataset": "T-ECD Small Retail",
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
            "sid_fit_scope": f"{train_ratio:.0%} train target items only",
            "test_target_digest": test_target_digest,
            "perseus_reference_test_target_digest": reference_digest,
            "rows": {split: len(frame) for split, frame in samples.items()},
            "candidate_items": len(candidate_catalog),
            "full_catalog_items": len(items),
            "item_catalog": item_catalog_report,
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
    """Read and normalize the public Retail events, catalog and user context."""

    root = source_dir / RETAIL_ROOT
    events_path = root / "events"
    items_path = root / "items.pq"
    users_path = source_dir / USERS_PATH
    for path in (events_path, items_path, users_path):
        if not path.exists():
            raise FileNotFoundError(f"T-ECD source path does not exist: {path}")

    event_paths = sorted(events_path.glob("*.pq"))
    if not event_paths:
        raise FileNotFoundError(f"T-ECD Retail events do not exist in: {events_path}")
    source_size_gib = sum(path.stat().st_size for path in event_paths) / 1024**3
    logger.info(
        "Reading %d Retail event partitions (%.2f GiB compressed) with Polars streaming",
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
        # Preserving source-row order disables Polars streaming. All order-sensitive
        # downstream operations sort explicitly, so only the unique row set matters.
        .unique()
        .collect(engine="streaming")
    )
    logger.info("Loaded and normalized %d Retail interactions", len(interactions))
    items = (
        pl.read_parquet(
            items_path,
            columns=["item_id", *ITEM_FEATURES, "embedding"],
        )
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
        raise ValueError("Retail catalog requires unique non-null item_id")
    embedding_type = items.schema["embedding"]
    if isinstance(embedding_type, pl.Array):
        embedding_lengths = items["embedding"].arr.len().unique().to_list()
    elif isinstance(embedding_type, pl.List):
        embedding_lengths = items["embedding"].list.len().unique().to_list()
    else:
        embedding_lengths = []
    if embedding_lengths != [300] or items["embedding"].null_count():
        raise ValueError(
            f"Retail embeddings must be non-null 300d vectors, got {embedding_lengths}"
        )
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
    logger.info("Loaded %d Retail items and %d users", len(items), len(users))
    return interactions, items, users


def _complete_item_catalog(
    items: pl.DataFrame,
    interactions: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, int | str]]:
    """Give every event item a transparent fallback representation for SID mapping."""

    event_items = interactions.select("item_id").unique().sort("item_id")
    missing = event_items.join(items.select("item_id"), on="item_id", how="anti")
    report: dict[str, int | str] = {
        "source_items": len(items),
        "event_items": len(event_items),
        "imputed_items": len(missing),
        "imputation": "null-features-zero-300d-embedding",
    }
    if missing.is_empty():
        return items, report

    logger.warning(
        "Retail source catalog misses %d/%d event items; preserving them with "
        "null features and a zero 300d SID embedding; examples=%s",
        len(missing),
        len(event_items),
        missing["item_id"].head(10).to_list(),
    )
    fallback = missing.with_columns(
        pl.lit(None, dtype=pl.String).alias("brand_id"),
        pl.lit(None, dtype=pl.String).alias("category"),
        pl.lit(None, dtype=pl.String).alias("subcategory"),
        pl.lit(None, dtype=pl.Float32).alias("price"),
        pl.lit([0.0] * 300, dtype=pl.Array(pl.Float32, 300)).alias("embedding"),
    ).select(items.columns)
    return pl.concat([items, fallback], how="vertical").sort("item_id"), report


def _filter_active_clients(interactions: pl.DataFrame) -> pl.DataFrame:
    """Match Perseus by removing target users above the 99th percentile."""

    counts = interactions.group_by("client_id").len(name="event_count")
    threshold = counts["event_count"].quantile(0.99)
    clients = counts.filter(pl.col("event_count") <= threshold).select("client_id")
    return interactions.join(clients, on="client_id", how="semi")


def _redistribute_target_timestamps(interactions: pl.DataFrame) -> pl.DataFrame:
    """Retain Perseus sample identity by adding stable offsets to second-level ties."""

    return interactions.sort(
        "timestamp", "client_id", maintain_order=True
    ).with_columns(
        (
            pl.col("timestamp")
            + pl.duration(
                microseconds=pl.int_range(pl.len()).over("timestamp", "client_id")
            )
        ).cast(pl.Datetime("ns"))
    )


def _prepare_basis(
    target_events: pl.DataFrame,
    *,
    train_ratio: float = 0.7,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split development targets without changing the public final 20% test."""

    if not 0 < train_ratio < 0.8:
        raise ValueError("train_ratio must be between 0 and 0.8")

    basis = (
        target_events.group_by("timestamp", "client_id")
        .agg(pl.col("item_id").unique().alias("target"))
        .sort("timestamp")
        .with_columns(item_id=pl.col("target").list.first())
    )
    if basis.is_empty() or basis["target"].list.len().max() != 1:
        raise ValueError(
            "target timestamp redistribution must yield one item per sample"
        )
    total = len(basis)
    validation_index, test_index = int(total * train_ratio), int(total * 0.8)
    if not 0 < validation_index < test_index < total:
        raise ValueError(
            f"not enough target rows for train_ratio={train_ratio}: {total}"
        )
    validation_timestamp = basis["timestamp"][validation_index]
    test_timestamp = basis["timestamp"][test_index]
    development = basis.filter(pl.col("timestamp") < test_timestamp)
    test = basis.filter(pl.col("timestamp") >= test_timestamp)

    items = development["item_id"].unique()
    while True:
        development = development.filter(pl.len().over("client_id") > 1)
        positive_items = development.filter(
            pl.int_range(pl.len()).over("client_id") > 0
        )["item_id"].unique()
        if len(positive_items) == len(items):
            break
        if not len(positive_items):
            raise ValueError("Perseus warm-item filtering removed every target item")
        items = positive_items
        development = development.filter(pl.col("item_id").is_in(items))
        test = test.filter(pl.col("item_id").is_in(items))

    reference_test = test.select(
        "client_id",
        pl.col("timestamp").alias("target_timestamp"),
        "item_id",
    )
    development = development.select(
        "client_id",
        pl.col("timestamp").alias("target_timestamp"),
        "item_id",
    )
    train = development.filter(pl.col("target_timestamp") < validation_timestamp)
    validation = development.filter(pl.col("target_timestamp") >= validation_timestamp)
    if train.is_empty() or validation.is_empty() or reference_test.is_empty():
        raise ValueError(f"train_ratio={train_ratio} filtering produced an empty split")
    candidate_ids = development.select("item_id").unique()
    return train, validation, reference_test, candidate_ids, reference_test.clone()


def _samples_with_context(
    basis: pl.DataFrame,
    users: pl.DataFrame,
    *,
    history_gap_hours: int = HISTORY_GAP_HOURS,
) -> pl.DataFrame:
    """Attach immutable user features and an optional history cutoff gap."""

    samples = basis.join(users, on="client_id", how="left", validate="m:1")
    samples = samples.with_columns(
        timestamp=pl.col("target_timestamp"),
        target=pl.concat_list("item_id"),
        target_ts=pl.concat_list("target_timestamp"),
        target_labels=pl.concat_list(pl.lit(1.0, dtype=pl.Float32)),
        context=pl.struct("socdem_cluster", "region"),
    )
    if history_gap_hours:
        samples = samples.with_columns(
            history_timestamp=pl.col("target_timestamp")
            - pl.duration(hours=history_gap_hours)
        )
    columns = [
        "sample_id",
        "client_id",
        "timestamp",
        "target",
        "target_labels",
        "target_ts",
        "context",
    ]
    if history_gap_hours:
        columns.insert(3, "history_timestamp")
    return (
        samples.with_columns(
            pl.struct("client_id", "target_timestamp", "item_id")
            .hash(seed=0)
            .alias("sample_id")
        )
        .select(columns)
        .sort("target_ts", "client_id")
    )


def _validate_protocol(
    samples: dict[str, pl.DataFrame],
    reference_test: pl.DataFrame,
    *,
    history_gap_hours: int = HISTORY_GAP_HOURS,
) -> None:
    """Fail if cutoff semantics or the public test-target population changes."""

    for split, frame in samples.items():
        if history_gap_hours:
            invalid = frame.filter(
                pl.col("target_ts").list.first() - pl.col("history_timestamp")
                != pl.duration(hours=history_gap_hours)
            )
            if len(invalid):
                raise ValueError(
                    f"{split} contains an invalid {history_gap_hours}-hour gap"
                )
        elif "history_timestamp" in frame.columns:
            raise ValueError(f"{split} unexpectedly contains a history cutoff gap")
        if frame.filter(pl.col("target_ts").list.first() != pl.col("timestamp")).height:
            raise ValueError(f"{split} changed prediction timestamps")
    expected = _basis_target_digest(reference_test)
    actual = _target_digest(samples["test"])
    if actual != expected:
        raise AssertionError("validation split changed the public Perseus test targets")


def _target_digest(samples: pl.DataFrame) -> str:
    """Hash target identity independently of the 12-hour history cutoff timestamp."""

    rows = (
        samples.select("client_id", "target_ts", "target")
        .explode("target_ts", "target")
        .sort("target_ts", "client_id", "target")
    )
    return _target_rows_digest(rows)


def _basis_target_digest(basis: pl.DataFrame) -> str:
    """Hash raw target rows before the history cutoff is added."""

    rows = basis.select(
        "client_id",
        pl.col("target_timestamp").alias("target_ts"),
        pl.col("item_id").alias("target"),
    ).sort("target_ts", "client_id", "target")
    return _target_rows_digest(rows)


def _target_rows_digest(rows: pl.DataFrame) -> str:
    """Hash normalized target identity rows."""

    import hashlib

    digest = hashlib.sha256()
    for client_id, timestamp, item_id in rows.iter_rows():
        digest.update(f"{client_id}\0{timestamp.isoformat()}\0{item_id}\n".encode())
    return digest.hexdigest()


def _write_representations(
    root: Path,
    items: pl.DataFrame,
    train_item_ids: pl.DataFrame,
) -> None:
    """Write row-aligned published 300d vectors for train-only fit and full apply."""

    root.mkdir()
    for name, frame in (
        ("all", items),
        (
            "train",
            train_item_ids.join(items, on="item_id", how="left", validate="1:1"),
        ),
    ):
        frame = frame.sort("item_id")
        destination = root / name
        destination.mkdir()
        frame.select("item_id").write_parquet(destination / "items.pq")
        matrix = np.asarray(frame["embedding"].to_numpy(), dtype=np.float32)
        if matrix.shape != (len(frame), 300) or not np.isfinite(matrix).all():
            raise ValueError(f"invalid {name} Retail embedding matrix: {matrix.shape}")
        np.save(destination / "embeddings.npy", matrix, allow_pickle=False)


def _write_eventhub(root: Path, events: pl.DataFrame) -> None:
    """Publish local EventHub-compatible action/date/MurmurHash partitions."""

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
