"""Categorical ID encoder shared by events, items and context features."""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import Any, Self

import polars as pl

from genrecsys.core.encoders import base


class Observer(base.Observer):
    def __init__(self) -> None:
        """Initialize an empty category frequency table."""

        self.value_counts = pl.DataFrame()

    def observe(self, values: pl.Series, /) -> None:
        """Accumulate frequencies of non-null training values."""

        counts = values.drop_nulls().rename("value").value_counts(name="count")
        if self.value_counts.is_empty():
            self.value_counts = counts
        else:
            self.value_counts = (
                pl.concat([self.value_counts, counts])
                .group_by("value")
                .agg(pl.col("count").sum())
            )

    def merge(self, other: Self, /) -> None:
        if not other.value_counts.width:
            return
        self.value_counts = (
            (
                pl.concat([self.value_counts, other.value_counts])
                if self.value_counts.width
                else other.value_counts
            )
            .group_by("value")
            .agg(pl.col("count").sum())
        )

    def save(self, path: Path, /) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.value_counts.write_parquet(path / "counts.pq")

    @classmethod
    def load(cls, path: Path, /) -> Self:
        result = cls()
        result.value_counts = pl.read_parquet(path / "counts.pq")
        return result


class Preprocessor(base.Preprocessor):
    def __init__(
        self,
        values: pl.DataFrame,
        /,
        *,
        unknown_index: int | None = None,
        vocab_size: int | None = None,
    ) -> None:
        """Store the frozen vocabulary, unknown index, and embedding table size."""

        self.values = values
        self.unknown_index = unknown_index
        self.vocab_size = vocab_size

    @cached_property
    def cardinality(self) -> int:
        """Return the largest known category index, or zero for an empty vocabulary."""

        maximum = self.values["index"].max()
        return 0 if maximum is None else int(maximum)

    @property
    def num_embeddings(self) -> int:
        """Return the embedding table size, including the unknown index."""

        return max(self.cardinality + 1, int(self.vocab_size or 0))

    @classmethod
    def fit(
        cls,
        observer: Observer,
        /,
        *,
        top_k: int | None = None,
        min_count: int | None = None,
        artifact: pl.DataFrame | None = None,
        vocab_size: int | None = None,
        categories: list[Any] | tuple[Any, ...] | None = None,
    ) -> Self:
        """Freeze a vocabulary from categories, an artifact, or training frequencies.

        Apply min_count/top_k only to observed categories. Reserve index 0 for unknowns.
        """

        if artifact is not None and categories is not None:
            raise ValueError(
                "id preprocessor cannot use artifact and categories together"
            )
        if categories is not None:
            values = pl.DataFrame(
                {
                    "value": categories,
                    "index": range(1, len(categories) + 1),
                }
            )
            if values["value"].null_count() or values["value"].n_unique() != len(
                values
            ):
                raise ValueError(
                    "id preprocessor categories must be unique and non-null"
                )
            return cls(values)
        if artifact is not None:
            required = {"value", "index"}
            if missing := required - set(artifact.columns):
                raise ValueError(
                    f"categorical artifact misses columns: {sorted(missing)}"
                )
            required_size = int(vocab_size or 0)
            preprocessor = cls(
                artifact,
                unknown_index=0,
                vocab_size=required_size or None,
            )
            if required_size and preprocessor.cardinality >= required_size:
                raise ValueError(
                    "frozen categorical ids do not fit declared vocab_size: "
                    f"max_id={preprocessor.cardinality}, vocab_size={required_size}"
                )
            return preprocessor
        value_counts = observer.value_counts
        if min_count is not None:
            value_counts = value_counts.filter(pl.col("count") >= min_count)
        value_counts = value_counts.sort(
            "count",
            "value",
            descending=[True, False],
        )
        if top_k is not None:
            value_counts = value_counts.head(top_k)
        return cls(
            value_counts.select("value").with_row_index("index", offset=1),
        )

    def static_transform(self, values: pl.Series, /) -> pl.Series:
        """Replace categories with indices, preserving row order and mapping unknowns."""

        mapping = self.values
        if "feature" in mapping.columns:
            if not (mapping["feature"] == values.name).any():
                raise ValueError(
                    f"frozen categorical artifact has no feature {values.name!r}"
                )
            mapping = mapping.filter(pl.col("feature") == values.name).drop("feature")
        if mapping.is_empty():
            return pl.Series(
                values.name,
                [self._unknown_index] * len(values),
                dtype=pl.Int64,
            )
        dtype = mapping.schema["value"]
        return (
            values.cast(dtype, strict=False)
            .to_frame("value")
            .lazy()
            .join(
                mapping.lazy(),
                how="left",
                on="value",
                maintain_order="left",
            )
            .select(pl.col("index").fill_null(self._unknown_index))
            .collect()
            .to_series()
            .rename(values.name)
        )

    @property
    def _unknown_index(self) -> int:
        # Index 0 is reserved by every model embedding as padding/unknown.  The
        # previous ``cardinality + 1`` fallback produced an index equal to
        # ``num_embeddings`` and therefore failed on unseen validation values.
        """Return the configured unknown index, defaulting to zero."""

        return 0 if self.unknown_index is None else self.unknown_index

    def save(self, path: Path, /) -> None:
        """Save the vocabulary and indexing settings to the encoder directory."""

        path.mkdir(parents=True, exist_ok=True)
        self.values.write_parquet(path / "values.pq")
        (path / "kwargs.json").write_text(
            json.dumps(
                {
                    "unknown_index": self.unknown_index,
                    "vocab_size": self.vocab_size,
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path, /) -> Self:
        """Load the vocabulary and indexing settings."""

        kwargs = json.loads((path / "kwargs.json").read_text(encoding="utf-8"))
        return cls(pl.read_parquet(path / "values.pq"), **kwargs)


encoder = base.Encoder(Observer, Preprocessor)
