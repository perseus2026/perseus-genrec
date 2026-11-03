"""Native dataset preparation for the shared GenRecSys API contract."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from data_sources import event_hub
from genrecsys import core
from genrecsys.core import preparation
from genrecsys.core.basis import validate_samples
from genrecsys.utils.files import discard_files

logger = logging.getLogger(__name__)


def prepare_dataset(
    config: core.Config,
    train_samples: pl.DataFrame,
    validation_samples: pl.DataFrame,
    /,
    *,
    train_artifacts: core.tasks.retrieval.Artifacts,
    validation_artifacts: core.tasks.retrieval.Artifacts,
    test_samples: pl.DataFrame | None = None,
    test_artifacts: core.tasks.retrieval.Artifacts | None = None,
    resources: core.Resources | dict[str, pl.DataFrame] | None = None,
    event_feature_mappings: tuple[core.event_feature_mapping.Spec, ...] = (),
    collect_statistics: bool = False,
) -> tuple[core.checkpoint.Prepared, core.Dataset]:
    """Build frozen preprocessors and a point-in-time EventStore.

    Fit unfrozen vocabularies, numeric statistics, and target frequencies on train
    only; transform holdout splits with that state. Apply fitted SID resources
    without training codebooks, embeddings, or recommendation weights.
    """

    config.validate_data()
    train_samples = preparation.normalise_sample_timestamps(train_samples)
    validation_samples = preparation.normalise_sample_timestamps(
        preparation.normalise_validation_targets(validation_samples)
    )
    if not isinstance(resources, core.Resources):
        resources = core.Resources(resources or {})
    if (test_samples is None) != (test_artifacts is None):
        raise ValueError("test_samples and test_artifacts must be provided together")
    if test_samples is not None:
        test_samples = preparation.normalise_sample_timestamps(
            preparation.normalise_validation_targets(test_samples)
        )
    for split, samples, artifacts in (
        ("train", train_samples, train_artifacts),
        ("validation", validation_samples, validation_artifacts),
        ("test", test_samples, test_artifacts),
    ):
        if samples is not None:
            validate_samples(samples, artifacts.items, split=split)
    preparation.resolve_representations(resources)
    encoder_to_observer = preparation.create_observers(config)
    # Target/catalog compatibility depends only on the basis. Fail before the
    # expensive EventHub pass rather than after reading every partition.
    task_preprocessor = preparation.fit_task_preprocessor(
        train_samples,
        train_artifacts,
        encoder_to_observer,
        config,
    )
    event_reader = event_hub.Reader(config)
    event_store = core.EventStore(backend="disk")
    event_store_rows = 0
    event_store_clients = 0
    event_feature_count_parts: dict[str, list[pl.DataFrame]] = {}
    train_event_feature_count_parts: dict[str, list[pl.DataFrame]] = {}
    event_feature_mappers = core.event_feature_mapping.build_mappers(
        event_feature_mappings,
        config,
        resources,
    )

    with TemporaryDirectory(prefix="genrecsys-prepare-") as cache_value:
        cache = Path(cache_value)
        partitions = preparation.distribute_samples(
            train_samples, validation_samples, test_samples
        )
        for partition, samples, events in preparation.read_event_partitions(
            event_reader,
            partitions,
        ):
            if events is None:
                continue
            events = core.feature_engineering.prepare_events(
                events,
                config,
                event_feature_mappers=event_feature_mappers,
            )
            event_store_rows += len(events)
            event_store_clients += events["client_id"].n_unique()
            for feature in (
                config.feature_to_encoder(event=True) if collect_statistics else ()
            ):
                if (
                    feature == config.task.objective.target
                    or config.encoders[config.features[feature].encoder].type != "id"
                ):
                    continue
                event_feature_count_parts.setdefault(feature, []).append(
                    events.group_by(feature).agg(
                        rows=pl.len(),
                        clients=pl.col("client_id").n_unique(),
                    )
                )
            preparation.observe_events(
                events,
                samples,
                encoder_to_observer,
                config,
                feature_count_parts=train_event_feature_count_parts
                if collect_statistics
                else None,
            )
            events.write_parquet(cache / event_hub.partition_filename(partition))
        if not event_store_rows:
            raise ValueError("EventHub contains no histories for basis samples")
        preparation.observe_samples(train_samples, encoder_to_observer, config)
        encoder_to_preprocessor = preparation.fit_encoder_preprocessors(
            encoder_to_observer,
            resources,
            config,
        )

        with event_store.open_writer() as writer:
            partition_paths = sorted(cache.iterdir())
            for index, partition_path in enumerate(partition_paths, start=1):
                started = time.perf_counter()
                logger.info(
                    "encoding EventStore partition %s/%s: %s",
                    index,
                    len(partition_paths),
                    partition_path.name,
                )
                events = core.feature_engineering.encode_events(
                    pl.read_parquet(partition_path),
                    encoder_to_preprocessor,
                    config,
                )
                writer.write(events)
                discard_files([partition_path], logger=logger)
                logger.info(
                    "encoded EventStore partition %s/%s: %s rows, %s clients in %.1fs",
                    index,
                    len(partition_paths),
                    len(events),
                    events["client_id"].n_unique(),
                    time.perf_counter() - started,
                )

    (
        train_samples,
        train_artifacts,
        validation_samples,
        validation_artifacts,
        test_samples,
        test_artifacts,
    ) = preparation.static_transform_basis(
        train_samples,
        train_artifacts,
        validation_samples,
        validation_artifacts,
        test_samples,
        test_artifacts,
        encoder_to_preprocessor,
        task_preprocessor,
        config,
    )
    from genrecsys.core.data_statistics import preparation_statistics

    statistics = (
        preparation_statistics(
            config,
            encoder_to_preprocessor,
            train_samples,
            validation_samples,
            test_samples,
            train_artifacts,
            validation_artifacts,
            test_artifacts,
            event_store_rows,
            event_store_clients,
            event_feature_count_parts,
            train_event_feature_count_parts,
        )
        if collect_statistics
        else {}
    )
    if statistics:
        logger.info(
            "prepared dataset statistics: train_rows=%s validation_rows=%s "
            "test_rows=%s event_features=%s history_window=%s",
            statistics["samples"]["train"]["rows"],
            statistics["samples"]["validation"]["rows"],
            statistics["samples"].get("test", {}).get("rows", 0),
            {name: value["vocab_size"] for name, value in statistics["events"].items()},
            statistics["history"]["max_duration_per_sequence"],
        )
    return (
        core.checkpoint.Prepared(
            config,
            encoder_to_preprocessor,
            core.DataContract.from_config(config),
            event_feature_mappers,
            files=resources.files,
        ),
        core.Dataset(
            train_samples=train_samples,
            train_artifacts=train_artifacts,
            validation_samples=validation_samples,
            validation_artifacts=validation_artifacts,
            event_store=event_store,
            task_preprocessor=task_preprocessor,
            test_samples=test_samples,
            test_artifacts=test_artifacts,
            statistics=statistics,
        ),
    )


__all__ = ["prepare_dataset"]
