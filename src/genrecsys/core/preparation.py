"""Shared operations for sequential and sharded dataset preparation."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Generator, Iterable
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from data_sources import event_hub
from genrecsys import core

logger = logging.getLogger(__name__)
PARTITION_PROGRESS_INTERVAL_SECONDS = 30


def load_basis(workdir: Path, config: core.Config) -> dict:
    """Load the same validated query/catalog bundle for either execution path."""
    manifest = core.DatasetManifest.load(workdir / "manifest.json")
    manifest.validate(workdir, config)
    result = {}
    for split in ("train", "validation", "test"):
        root = workdir / "basis" / split
        samples = pl.read_parquet(root / "samples.pq")
        if split != "train":
            samples = normalise_validation_targets(samples)
        result[f"{split}_samples"] = normalise_sample_timestamps(samples)
        result[f"{split}_artifacts"] = core.tasks.retrieval.Artifacts.load(
            root / "artifacts"
        )
    return result


def create_observers(config: core.Config) -> dict[str, core.encoders.Observer]:
    """Use encoder-owned statistics in both sequential and sharded preparation."""
    return {
        name: core.encoders.registry[description.type].observer()
        for name, description in config.encoders.items()
    }


def normalise_validation_targets(samples: pl.DataFrame) -> pl.DataFrame:
    """Separate relevance labels before the retrieval target encoder runs."""

    target_type = samples.schema.get("target")
    if not isinstance(target_type, pl.List) or not isinstance(
        target_type.inner,
        pl.Struct,
    ):
        if "target_labels" in samples.columns:
            return samples
        return samples.with_columns(
            target_labels=pl.col("target").list.eval(pl.lit(1.0))
        )
    fields = {field.name for field in target_type.inner.fields}
    if not {"item", "relevance"}.issubset(fields):
        raise ValueError("validation target structs require item and relevance")
    return samples.with_columns(
        target_labels=pl.col("target").list.eval(
            pl.element().struct.field("relevance").cast(pl.Float64)
        ),
        target=pl.col("target").list.eval(pl.element().struct.field("item")),
    )


def distribute_samples(
    train_samples: pl.DataFrame,
    validation_samples: pl.DataFrame,
    test_samples: pl.DataFrame | None = None,
) -> Generator[tuple[int, int, int, pl.DataFrame], None, None]:
    """Combine splits with fold labels and yield EventHub history partitions.

    Keep repeated user-hour samples; fold labels distinguish train observations.
    """

    folds = [
        history_samples(train_samples, "train"),
        history_samples(validation_samples, "validation"),
    ]
    if test_samples is not None:
        folds.append(history_samples(test_samples, "test"))
    samples = pl.concat(folds)
    samples = event_hub.distribute_to_partitions(samples).sort("partition_id")
    logger.info(
        "has %s samples for %s partitions",
        len(samples),
        samples["partition_id"].n_unique(),
    )
    partition_count = int(samples["partition_id"].n_unique())
    partitions = samples.group_by("partition_id", maintain_order=True)
    for index, ((partition,), partition_samples) in enumerate(partitions, start=1):
        yield (
            index,
            partition_count,
            int(partition),
            partition_samples.drop("partition_id"),
        )


def read_event_partitions(
    reader: event_hub.Reader,
    partitions: Iterable[tuple[int, int, int, pl.DataFrame]],
) -> Generator[tuple[int, pl.DataFrame, pl.DataFrame | None], None, None]:
    """Read one partition at a time, with a logging-only heartbeat thread."""

    overall_started = time.perf_counter()
    for index, partition_count, partition, samples in partitions:
        logger.info(
            "processing EventHub partition %s/%s: id=%s, %s samples, %s clients",
            index,
            partition_count,
            partition,
            len(samples),
            samples["client_id"].n_unique(),
        )
        started = time.perf_counter()
        finished = threading.Event()

        def heartbeat(finished, index, partition_count, partition):
            while not finished.wait(PARTITION_PROGRESS_INTERVAL_SECONDS):
                elapsed = time.perf_counter() - overall_started
                eta = (
                    f"{elapsed / (index - 1) * (partition_count - index + 1) / 60:.1f}min"
                    if index > 1
                    else "unknown (no completed partitions)"
                )
                logger.info(
                    "waiting for EventHub partition id=%s: completed=%s/%s "
                    "elapsed=%.1fmin eta=%s",
                    partition,
                    index - 1,
                    partition_count,
                    elapsed / 60,
                    eta,
                )

        thread = threading.Thread(
            target=heartbeat,
            args=(finished, index, partition_count, partition),
            daemon=True,
        )
        thread.start()
        try:
            events = reader.read(partition, samples)
        finally:
            finished.set()
            thread.join()
        yield partition, samples, events
        total_elapsed = time.perf_counter() - overall_started
        eta_seconds = total_elapsed / index * (partition_count - index)
        logger.info(
            "processed EventHub partition %s/%s: id=%s in %.1fs; "
            "elapsed=%.1fmin eta=%.1fmin",
            index,
            partition_count,
            partition,
            time.perf_counter() - started,
            total_elapsed / 60,
            eta_seconds / 60,
        )


def normalise_sample_timestamps(samples: pl.DataFrame) -> pl.DataFrame:
    """Normalize prediction time and an optional earlier history cutoff."""

    columns = [event_hub.timestamp_expression("timestamp")]
    if "history_timestamp" in samples.columns:
        columns.append(event_hub.timestamp_expression("history_timestamp"))
    return samples.with_columns(columns)


def history_samples(samples: pl.DataFrame, fold: str) -> pl.DataFrame:
    """Select the explicit history cutoff without changing prediction time."""

    cutoff = (
        "history_timestamp" if "history_timestamp" in samples.columns else "timestamp"
    )
    return samples.select(
        "client_id",
        pl.col(cutoff).alias("_sample_timestamp"),
        _fold=pl.lit(fold),
    )


def observe_events(
    events: pl.DataFrame,
    samples: pl.DataFrame,
    encoder_to_observer: dict[str, core.encoders.Observer],
    config: core.Config,
    *,
    feature_count_parts: dict[str, list[pl.DataFrame]] | None = None,
) -> None:
    """Observe events from train users strictly before their last train sample.

    Exclude holdout-only users and post-train events from preprocessor fitting.
    Dataset adapters do not collect these history statistics.
    """

    observed_features = {
        feature: encoder
        for feature, encoder in config.feature_to_encoder(event=True).items()
        if not has_frozen_mapping(config.encoders[encoder])
    }
    if not observed_features and feature_count_parts is None:
        return
    train_events = (
        events.lazy()
        .join(
            samples.lazy()
            .filter(pl.col("_fold") == "train")
            .group_by("client_id")
            .agg(_last_sample_timestamp=pl.col("_sample_timestamp").max()),
            how="inner",
            on="client_id",
        )
        .filter(pl.col("timestamp") < pl.col("_last_sample_timestamp"))
        .drop("_last_sample_timestamp")
        .collect()
    )
    for feature, encoder in config.feature_to_encoder(event=True).items():
        if feature in observed_features:
            encoder_to_observer[encoder].observe(train_events[feature])
        if (
            feature_count_parts is not None
            and feature != config.task.objective.target
            and config.encoders[encoder].type == "id"
        ):
            feature_count_parts.setdefault(feature, []).append(
                train_events[feature]
                .drop_nulls()
                .rename("value")
                .value_counts(name="count")
            )


def observe_samples(
    train_samples: pl.DataFrame,
    encoder_to_observer: dict[str, core.encoders.Observer],
    config: core.Config,
) -> None:
    """Observe train context features after removing repeated user-context pairs."""

    features = config.feature_to_encoder(context=True)
    if not features:
        return
    context = (
        train_samples.lazy()
        .select("client_id", "context")
        .unique()
        .select(pl.col("context").struct.unnest())
        .collect()
    )
    for feature, encoder in features.items():
        if not has_frozen_mapping(config.encoders[encoder]):
            encoder_to_observer[encoder].observe(context[feature])


def fit_task_preprocessor(
    train_samples: pl.DataFrame,
    train_artifacts: core.tasks.retrieval.Artifacts,
    encoder_to_observer: dict[str, core.encoders.Observer],
    config: core.Config,
) -> core.tasks.retrieval.Preprocessor:
    """Fit train target indices and frequencies, checking train catalog coverage."""

    feature_to_observer = {
        feature: encoder_to_observer[encoder]
        for feature, encoder in config.feature_to_encoder(artifacts=True).items()
        if not has_frozen_mapping(config.encoders[encoder])
    }
    return core.tasks.retrieval.Preprocessor.fit(
        train_samples["target"],
        train_artifacts,
        feature_to_observer,
    )


def fit_encoder_preprocessors(
    encoder_to_observer: dict[str, core.encoders.Observer],
    resources: core.Resources,
    config: core.Config,
) -> dict[str, core.encoders.Preprocessor]:
    """Build feature preprocessors from train statistics or frozen resources.

    SID and categorical artifacts/categories are reused without fitting. Unfrozen
    ID vocabularies and numeric mean/std are learned from selected train data.
    """

    result: dict[str, core.encoders.Preprocessor] = {}
    for name, description in config.encoders.items():
        result[name] = core.encoders.registry[description.type].preprocessor.fit(
            encoder_to_observer[name],
            **core.feature_engineering.resolve_encoder_kwargs(
                description,
                resources,
            ),
        )
    return result


def static_transform_basis(
    train_samples: pl.DataFrame,
    train_artifacts: core.tasks.retrieval.Artifacts,
    validation_samples: pl.DataFrame,
    validation_artifacts: core.tasks.retrieval.Artifacts,
    test_samples: pl.DataFrame | None,
    test_artifacts: core.tasks.retrieval.Artifacts | None,
    encoder_to_preprocessor: dict[str, core.encoders.Preprocessor],
    task_preprocessor: core.tasks.retrieval.Preprocessor,
    config: core.Config,
) -> tuple[
    pl.DataFrame,
    core.tasks.retrieval.Artifacts,
    pl.DataFrame,
    core.tasks.retrieval.Artifacts,
    pl.DataFrame | None,
    core.tasks.retrieval.Artifacts | None,
]:
    """Transform all splits with the same frozen feature preprocessors.

    Assign task indices to train targets and preserve holdout labels for evaluation.
    """

    context_features = config.feature_to_encoder(context=True)
    if context_features:
        train_samples = core.feature_engineering.transform_context(
            train_samples,
            context_features,
            encoder_to_preprocessor,
        )
        validation_samples = core.feature_engineering.transform_context(
            validation_samples,
            context_features,
            encoder_to_preprocessor,
        )
        if test_samples is not None:
            test_samples = core.feature_engineering.transform_context(
                test_samples,
                context_features,
                encoder_to_preprocessor,
            )
    artifact_preprocessors = {
        feature: encoder_to_preprocessor[encoder]
        for feature, encoder in config.feature_to_encoder(artifacts=True).items()
        if not has_frozen_mapping(config.encoders[encoder])
    }
    train_artifacts = train_artifacts.static_transform(artifact_preprocessors)
    validation_artifacts = validation_artifacts.static_transform(artifact_preprocessors)
    if test_artifacts is not None:
        test_artifacts = test_artifacts.static_transform(artifact_preprocessors)
    train_samples = train_samples.with_columns(
        pl.col("target").map_batches(task_preprocessor.static_transform)
    )
    return (
        train_samples.sort("timestamp", "client_id"),
        train_artifacts,
        validation_samples.sort("timestamp", "client_id"),
        validation_artifacts,
        (
            test_samples.sort("timestamp", "client_id")
            if test_samples is not None
            else None
        ),
        test_artifacts,
    )


def has_frozen_mapping(encoder: core.config.Encoder) -> bool:
    """Return whether preparation consumes an existing vocabulary/SID artifact."""
    return "artifact" in encoder.preprocessor or "categories" in encoder.preprocessor


def resolve_representations(resources: core.Resources) -> None:
    """Apply deferred SID mapper recipes and update resources.tables in place.

    Use fitted checkpoint/item/embedding references in bounded CPU batches. Reuse
    existing tables; do not train SID codebooks or embeddings.
    """

    if not resources.representations:
        return
    from item_representations.semantic_ids.mapping import apply_sid_mapper

    for name, recipe in resources.representations.items():
        with TemporaryDirectory(prefix="genrecsys-sid-apply-") as temporary:
            root = Path(temporary)
            resources.resource(recipe["items"]).write_parquet(root / "items.pq")
            apply_sid_mapper(
                recipe["path"],
                root / "items.pq",
                resources.files[recipe["embeddings"]],
                root / "mapping",
                eval_batch_size=recipe.get("batch_size", 8192),
                device="cpu",
            )
            resources.tables[name] = pl.read_parquet(
                root / "mapping/item_to_sid.parquet"
            )
