"""Resolve frozen GRID candidates, target indices and generation constraints."""

from __future__ import annotations

from datetime import UTC
from typing import Any

import polars as pl

from genrecsys.core import encoders
from genrecsys.core.checkpoint import Prepared
from genrecsys.core.dataset import Dataset
from genrecsys.core.tasks import retrieval


def _catalog(
    artifacts: retrieval.Artifacts,
    item: encoders.semantic_id.Preprocessor,
) -> pl.DataFrame:
    """Join candidates to frozen SID codes and preserve candidate metadata.

    Require unique item IDs and complete mapping coverage. GRID always consumes
    full SIDs, including the collision level.
    """

    required = {"item_id", item.item_id_column}
    if missing := required - set(artifacts.items.columns):
        raise ValueError(
            "GRID retrieval items require external item_id and SID key: "
            f"missing {sorted(missing)}"
        )
    extra = [
        name
        for name in ("created_at", "generation_allowed")
        if name in artifacts.items.columns
    ]
    columns = list(dict.fromkeys(("item_id", item.item_id_column, *extra)))
    candidates = artifacts.items.select(columns)
    if (
        candidates.select(pl.any_horizontal(pl.col(list(required)).is_null()))
        .to_series()
        .any()
    ):
        raise ValueError("retrieval catalog contains null item identities")
    if candidates["item_id"].n_unique() != len(candidates):
        raise ValueError("retrieval catalog contains duplicate external items")
    catalog = candidates.join(
        item.mapping,
        on=item.item_id_column,
        how="left",
        validate="m:1",
    )
    missing_mapping = catalog.filter(pl.col(item.sid_column).is_null())
    if not missing_mapping.is_empty():
        examples = missing_mapping[item.item_id_column].head(20).to_list()
        raise ValueError(
            "retrieval catalog contains items absent from item_to_sid: "
            f"count={len(missing_mapping)}, examples={examples}"
        )
    catalog = catalog.select(
        "item_id",
        pl.col(item.sid_column).alias("sid"),
        *extra,
    ).sort("item_id")
    if catalog.is_empty():
        raise ValueError("candidate catalog contains no SID-mapped items")
    if "created_at" in catalog.columns and not isinstance(
        catalog.schema["created_at"], pl.Datetime
    ):
        raise ValueError("catalog created_at must have Datetime dtype")
    if "generation_allowed" in catalog.columns and (
        catalog.schema["generation_allowed"] != pl.Boolean
        or catalog["generation_allowed"].null_count()
    ):
        raise ValueError("catalog generation_allowed must be non-null Boolean")
    return catalog


def _training_targets(
    dataset: Dataset,
    artifacts: retrieval.Artifacts,
    item: encoders.semantic_id.Preprocessor,
) -> dict[int, tuple[int, ...]]:
    """Map train target indices to SID codes, requiring complete coverage."""

    if not isinstance(dataset.task_preprocessor, retrieval.Preprocessor):
        raise TypeError("GRID requires retrieval.Preprocessor")
    targets = dataset.task_preprocessor.items.select("index", "item_id").join(
        _catalog(artifacts, item),
        on="item_id",
        how="left",
        validate="1:1",
    )
    missing_mapping = targets.filter(pl.col("sid").is_null())
    if not missing_mapping.is_empty():
        examples = missing_mapping["item_id"].head(20).to_list()
        raise ValueError(
            "train targets contain items absent from the SID catalog: "
            f"count={len(missing_mapping)}, examples={examples}"
        )
    return {
        int(index): tuple(int(value) for value in sid)
        for index, sid in targets.select("index", "sid").iter_rows()
    }


def _item_preprocessor(checkpoint: Prepared) -> encoders.semantic_id.Preprocessor:
    """Return the target feature's semantic-ID preprocessor, checking its type."""

    feature = checkpoint.config.task.objective.target
    name = checkpoint.config.features[feature].encoder
    value = checkpoint.encoder_to_preprocessor[name]
    if not isinstance(value, encoders.semantic_id.Preprocessor):
        raise TypeError(f"encoders.{name} must use semantic_id.Preprocessor")
    return value


def _items_by_sid(catalog: pl.DataFrame) -> dict[tuple[int, ...], list[Any]]:
    """Group external item IDs by SID, preserving catalog order."""

    result: dict[tuple[int, ...], list[Any]] = {}
    for item_id, sid in catalog.select("item_id", "sid").iter_rows():
        result.setdefault(tuple(int(value) for value in sid), []).append(item_id)
    return result


def _item_creation_times(catalog: pl.DataFrame) -> dict[Any, float]:
    """Return item creation times as UTC epoch seconds, or an empty mapping."""

    if "created_at" not in catalog.columns:
        return {}
    return {
        item_id: (
            created_at.replace(tzinfo=UTC) if created_at.tzinfo is None else created_at
        ).timestamp()
        for item_id, created_at in catalog.select("item_id", "created_at").iter_rows()
        if created_at is not None
    }


def _allowed_codes(catalog: pl.DataFrame) -> tuple[tuple[int, ...], ...]:
    """Return distinct SID codes allowed for candidate generation in catalog order."""

    if "generation_allowed" in catalog.columns:
        catalog = catalog.filter("generation_allowed")
    return tuple(
        sorted({tuple(int(token) for token in sid) for sid in catalog["sid"].to_list()})
    )


def _generation_catalog(catalog: pl.DataFrame) -> pl.DataFrame:
    """Use the same eligible candidates for SID search and external ID expansion."""

    return (
        catalog.filter("generation_allowed")
        if "generation_allowed" in catalog.columns
        else catalog
    )
