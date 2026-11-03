"""Config-declared raw feature derivation before generic encoders run."""

from __future__ import annotations

from typing import Any

import polars as pl

from genrecsys.core import encoders, feature_transforms
from genrecsys.core.config import Config
from genrecsys.core.event_feature_mapping import EventFeatureMapper
from genrecsys.core.resources import Resources


def prepare_events(
    events: pl.DataFrame,
    config: Config,
    /,
    *,
    event_feature_mappers: dict[str, EventFeatureMapper] | None = None,
) -> pl.DataFrame:
    """Materialize configured model features from raw EventHub attributes."""

    model_event_features = set(config.feature_to_encoder(event=True))
    prepared = events
    mapped_attributes: set[str] = set()
    for mapper in (event_feature_mappers or {}).values():
        prepared = mapper.transform(prepared)
        mapped_attributes.update(mapper.attributes)
    derived = []
    for name in model_event_features:
        feature = config.features[name]
        if feature.transform is not None:
            derived.append(
                feature_transforms.expression(
                    feature.transform.type,
                    feature.transform.params,
                ).alias(name)
            )
    if derived:
        prepared = prepared.with_columns(*derived)
    disposable = sorted(
        (mapped_attributes - model_event_features) & set(prepared.columns)
    )
    return prepared.drop(*disposable)


def encode_events(
    events: pl.DataFrame,
    encoder_to_preprocessor: dict[str, encoders.Preprocessor],
    config: Config,
    /,
) -> pl.DataFrame:
    """Encode prepared raw features identically for Stage 08 and live inference.

    Preserve the raw event name for routing before its categorical encoding.
    """

    return events.with_columns(_event=pl.col("event")).with_columns(
        pl.col(feature).map_batches(encoder_to_preprocessor[encoder].static_transform)
        for feature, encoder in config.feature_to_encoder(event=True).items()
    )


def resolve_encoder_kwargs(
    description: Any,
    resources: Resources,
    /,
) -> dict[str, Any]:
    """Replace logical artifact names with shared immutable resources."""

    kwargs = dict(description.preprocessor)
    if isinstance(kwargs.get("artifact"), str):
        kwargs["artifact"] = resources.resource(kwargs["artifact"])
    return kwargs


def transform_context(
    samples: pl.DataFrame,
    feature_to_encoder: dict[str, str],
    encoder_to_preprocessor: dict[str, encoders.Preprocessor],
    /,
) -> pl.DataFrame:
    """Encode context struct fields with frozen feature preprocessors.

    Require context when context features are configured; preserve other sample columns.
    """

    if not feature_to_encoder:
        return samples
    if "context" not in samples.columns:
        raise ValueError("configured context features require a context struct")
    return samples.with_columns(
        context=samples["context"]
        .struct.unnest()
        .with_columns(
            pl.col(feature).map_batches(
                encoder_to_preprocessor[encoder].static_transform
            )
            for feature, encoder in feature_to_encoder.items()
        )
        .to_struct()
    )


__all__ = [
    "encode_events",
    "prepare_events",
    "resolve_encoder_kwargs",
    "transform_context",
]
