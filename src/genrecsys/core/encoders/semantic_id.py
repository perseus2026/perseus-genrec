"""Semantic-ID feature encoder for an independently fitted item mapping."""

from __future__ import annotations

import json
import logging
from functools import cached_property
from pathlib import Path
from typing import Self

import polars as pl

from genrecsys.core.encoders import base

logger = logging.getLogger(__name__)


class Observer(base.Observer):
    """Collect item identities seen in train events for coverage diagnostics."""

    def __init__(self) -> None:
        """Initialize an empty set of item IDs for train coverage checks."""

        self.values = pl.Series("item", dtype=pl.Null)

    def observe(self, values: pl.Series, /) -> None:
        """Collect unique non-null item IDs from a training batch."""

        values = values.drop_nulls().unique()
        if self.values.dtype == pl.Null:
            self.values = values.rename("item")
        else:
            self.values = pl.concat([self.values, values]).unique()

    def merge(self, other: Self, /) -> None:
        if not other.values.is_empty():
            self.observe(other.values)

    def save(self, path: Path, /) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.values.to_frame().write_parquet(path / "items.pq")

    @classmethod
    def load(cls, path: Path, /) -> Self:
        result = cls()
        result.values = pl.read_parquet(path / "items.pq").to_series()
        return result


class Preprocessor(base.Preprocessor):
    """Map canonical item IDs to fixed-width SID tokens using a frozen artifact.

    SID codebooks are fitted upstream, not by this preprocessor.
    """

    def __init__(
        self,
        mapping: pl.DataFrame,
        /,
        *,
        item_id_column: str = "item_key",
        sid_column: str = "sid",
    ) -> None:
        """Validate and store a fitted item-to-SID mapping; do not train SID codebooks."""

        self.item_id_column = item_id_column
        self.sid_column = sid_column
        self.mapping = _normalise_mapping(
            mapping,
            item_id_column=item_id_column,
            sid_column=sid_column,
        )
        _validate_mapping(self.mapping, item_id_column, sid_column)

    @cached_property
    def code_length(self) -> int:
        """Return the common SID length, rejecting mixed code lengths."""

        lengths = self.mapping[self.sid_column].list.len().unique().to_list()
        if len(lengths) != 1:
            raise ValueError(f"semantic IDs must have one fixed width, got {lengths}")
        return int(lengths[0])

    @cached_property
    def vocab_size(self) -> int:
        """Return the largest SID token plus one."""

        maximum = self.mapping[self.sid_column].list.max().max()
        return 0 if maximum is None else int(maximum) + 1

    @classmethod
    def fit(
        cls,
        observer: Observer,
        /,
        *,
        artifact: str | Path | pl.DataFrame | pl.Series | None = None,
        item_id_column: str = "item_key",
        sid_column: str = "sid",
    ) -> Self:
        """Load frozen mapping/identity artifacts and validate train coverage."""

        if artifact is None:
            raise ValueError("semantic_id requires an explicit SID mapping artifact")
        if isinstance(artifact, pl.Series):
            artifact_frame = artifact.to_frame()
        elif isinstance(artifact, pl.DataFrame):
            artifact_frame = artifact
        else:
            artifact_frame = pl.read_parquet(Path(artifact))
        preprocessor = cls(
            artifact_frame,
            item_id_column=item_id_column,
            sid_column=sid_column,
        )
        preprocessor.validate_observed(observer)
        return preprocessor

    def validate_observed(self, observer: Observer, /) -> None:
        """Fail when the frozen mapping covers none of the observed train items."""

        if observer.values.is_empty():
            return
        missing = observer.values.filter(
            observer.values.is_in(self.mapping[self.item_id_column].implode()).not_()
        )
        if len(missing) == len(observer.values):
            raise ValueError("semantic-ID artifact covers no observed train items")
        if len(missing):
            logger.warning(
                "semantic-ID artifact misses %s/%s observed train items; their "
                "history events will be discarded",
                len(missing),
                len(observer.values),
            )

    def static_transform(self, values: pl.Series, /) -> pl.Series:
        """Replace item identities with SID lists; unknown items become null."""

        return (
            values.to_frame(self.item_id_column)
            .join(
                self.mapping,
                on=self.item_id_column,
                how="left",
                maintain_order="left",
            )[self.sid_column]
            .rename(values.name)
        )

    def save(self, path: Path, /) -> None:
        """Save mapping.pq and its column names in config.json."""

        path.mkdir(parents=True, exist_ok=True)
        self.mapping.write_parquet(path / "mapping.pq")
        (path / "config.json").write_text(
            json.dumps(
                {
                    "item_id_column": self.item_id_column,
                    "sid_column": self.sid_column,
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path, /) -> Self:
        """Load and validate the frozen SID mapping."""

        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        return cls(
            pl.read_parquet(path / "mapping.pq"),
            **config,
        )


def _normalise_mapping(
    mapping: pl.DataFrame,
    *,
    item_id_column: str,
    sid_column: str,
) -> pl.DataFrame:
    """Select identity/SID columns and require list-valued SID codes."""

    if item_id_column not in mapping.columns:
        raise ValueError(
            f"semantic-ID mapping has no {item_id_column!r} column: {mapping.columns}"
        )
    if sid_column not in mapping.columns:
        raise ValueError(
            f"semantic-ID mapping has no {sid_column!r} column: {mapping.columns}"
        )
    if mapping.schema[sid_column].base_type() != pl.List:
        raise TypeError(f"semantic-ID {sid_column!r} column must be a list")
    return mapping.select(item_id_column, sid_column)


def _validate_mapping(
    mapping: pl.DataFrame,
    item_id_column: str,
    sid_column: str,
) -> None:
    """Reject empty mappings, nulls, duplicate identities/codes, and empty SID codes."""

    if mapping.is_empty():
        raise ValueError("semantic-ID mapping is empty")
    if mapping[item_id_column].null_count():
        raise ValueError("semantic-ID mapping contains null item identities")
    if mapping[sid_column].null_count():
        raise ValueError("semantic-ID mapping contains null SID codes")
    if mapping[item_id_column].n_unique() != len(mapping):
        raise ValueError("semantic-ID mapping contains duplicate item identities")
    if mapping[sid_column].n_unique() != len(mapping):
        raise ValueError(
            "semantic-ID mapping is not bijective; add or repair the collision level"
        )
    if mapping[sid_column].list.len().min() == 0:
        raise ValueError("semantic-ID mapping contains an empty code")


encoder = base.Encoder(Observer, Preprocessor)
