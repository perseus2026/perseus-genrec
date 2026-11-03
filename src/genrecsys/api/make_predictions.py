"""Config-selected generative inference over EventHub sample partitions."""

from __future__ import annotations

from collections.abc import Generator, Iterator

import polars as pl

from data_sources import event_hub
from genrecsys import core
from genrecsys.core.checkpoint import Trained
from genrecsys.core.models import get
from genrecsys.core.models.base import InferenceSession


def distribute_samples(
    samples: pl.DataFrame,
    /,
) -> Generator[tuple[int, pl.DataFrame], None, None]:
    """Yield stable EventHub partitions without dropping repeated user samples.

    Remove the temporary partition_id column from each returned frame.
    """

    partitioned = event_hub.distribute_to_partitions(samples).sort("partition_id")
    for (partition_id,), values in partitioned.group_by(
        "partition_id", maintain_order=True
    ):
        yield int(partition_id), values.drop("partition_id")


def create_inference_session(
    checkpoint: Trained,
    /,
    *,
    artifacts: core.tasks.retrieval.Artifacts | None = None,
) -> InferenceSession:
    """Create a reusable inference session, optionally replacing the split catalog.

    Reuse the session across batches to avoid reloading model weights and embeddings.
    """

    checkpoint.prepared.config.validate()
    implementation = get(checkpoint.prepared.config.model.type)
    return implementation.create_inference_session(checkpoint, artifacts=artifacts)


def iter_predictions(
    partition_id: int,
    samples: pl.DataFrame,
    checkpoint: Trained,
    /,
    *,
    event_store: core.EventStore | None = None,
    samples_prepared: bool = False,
    inference_session: InferenceSession | None = None,
) -> Iterator[pl.DataFrame]:
    """Yield prediction batches joined to the original sample rows.

    Live inference encodes EventHub history with frozen preprocessors. Prepared
    EventStore inference requires samples_prepared=True. Empty input yields nothing.
    """

    indexed = samples.with_row_index("__sample_index")
    if event_store is not None:
        if not samples_prepared:
            raise ValueError(
                "prepared EventStore inference requires samples_prepared=True"
            )
        model_samples = indexed
        events: pl.DataFrame | core.EventStore = event_store
    else:
        if samples_prepared:
            raise ValueError("samples_prepared=True requires an EventStore")
        timestamp_columns = [event_hub.timestamp_expression("timestamp")]
        if "history_timestamp" in indexed.columns:
            timestamp_columns.append(
                event_hub.timestamp_expression("history_timestamp")
            )
        model_samples = core.feature_engineering.transform_context(
            indexed.with_columns(timestamp_columns),
            checkpoint.prepared.config.feature_to_encoder(context=True),
            checkpoint.prepared.encoder_to_preprocessor,
        )
        cutoff = (
            "history_timestamp"
            if "history_timestamp" in model_samples.columns
            else "timestamp"
        )
        event_samples = model_samples.select(
            "client_id",
            pl.col(cutoff).alias("_sample_timestamp"),
        )
        live_events = event_hub.Reader(checkpoint.prepared.config).read(
            partition_id,
            event_samples,
        )
        if live_events is None:
            events = pl.DataFrame(schema={"client_id": pl.String})
        else:
            events = core.feature_engineering.prepare_events(
                live_events,
                checkpoint.prepared.config,
                event_feature_mappers=checkpoint.prepared.event_feature_mappers,
            )
            events = core.feature_engineering.encode_events(
                events,
                checkpoint.prepared.encoder_to_preprocessor,
                checkpoint.prepared.config,
            )
    if model_samples.is_empty():
        return
    session = inference_session or create_inference_session(checkpoint)
    index_dtype = indexed.schema["__sample_index"]
    for predictions in session.iter_predictions(model_samples, events):
        if predictions.is_empty():
            continue
        if "__sample_index" not in predictions.columns:
            raise ValueError("prediction batch misses __sample_index")
        predictions = predictions.with_columns(
            pl.col("__sample_index").cast(index_dtype)
        )
        yield (
            indexed.join(predictions, how="inner", on="__sample_index")
            .sort("__sample_index")
            .drop("__sample_index")
        )


def make_predictions(
    partition_id: int,
    samples: pl.DataFrame,
    checkpoint: Trained,
    /,
    *,
    event_store: core.EventStore | None = None,
    samples_prepared: bool = False,
) -> pl.DataFrame | None:
    """Collect one partition's predictions into a DataFrame, or return None.

    All output batches stay in RAM; use iter_predictions for streaming output.
    """

    frames = list(
        iter_predictions(
            partition_id,
            samples,
            checkpoint,
            event_store=event_store,
            samples_prepared=samples_prepared,
        )
    )
    return pl.concat(frames) if frames else None


__all__ = [
    "create_inference_session",
    "distribute_samples",
    "iter_predictions",
    "make_predictions",
]
