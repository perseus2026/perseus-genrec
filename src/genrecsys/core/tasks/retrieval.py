"""Retrieval artifacts and target preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import polars as pl

from genrecsys.core import encoders


class Artifacts:
    """Split-specific retrieval catalog. Shared resources belong to the bundle."""

    def __init__(
        self,
        items: pl.DataFrame,
    ) -> None:
        """Store the candidate catalog, requiring an item_id column."""

        if "item_id" not in items.columns:
            raise ValueError("retrieval artifacts require an 'item_id' column")
        self.items = items

    def static_transform(
        self,
        feature_to_preprocessor: dict[str, encoders.Preprocessor],
        /,
    ) -> Self:
        """Transform catalog item features in place and return self."""

        self.items = self.items.with_columns(
            pl.col(feature).map_batches(preprocessor.static_transform)
            for feature, preprocessor in feature_to_preprocessor.items()
        )
        return self

    def save(self, path: Path, /) -> None:
        """Write the candidate catalog to items.pq."""

        path.mkdir(parents=True, exist_ok=True)
        self.items.write_parquet(path / "items.pq")

    @classmethod
    def load(cls, path: Path, /) -> Self:
        """Load the candidate catalog from items.pq."""

        return cls(pl.read_parquet(path / "items.pq"))


class Preprocessor:
    """Map training target items to stable catalog row indices."""

    def __init__(self, items: pl.DataFrame, /) -> None:
        """Store the indexed training catalog and log sampling probabilities."""

        self.items = items

    @classmethod
    def fit(
        cls,
        values: pl.Series,
        artifacts: Artifacts,
        feature_to_observer: dict[str, encoders.Observer],
        /,
    ) -> Self:
        """Fit catalog indices and target frequencies from training data.

        Observe train catalog features, retain items without train targets, and reject
        training targets missing from the catalog.
        """

        counts = (
            values.explode(empty_as_null=True)
            .drop_nulls()
            .rename("item_id")
            .value_counts()
        )
        return cls.from_counts(counts, artifacts, feature_to_observer)

    @classmethod
    def from_counts(
        cls,
        counts: pl.DataFrame,
        artifacts: Artifacts,
        feature_to_observer: dict[str, encoders.Observer],
    ) -> Self:
        """Fit from streaming target counts without retaining sample histories."""
        items = (
            counts.lazy()
            .select(
                "item_id",
                logq=(pl.col("count") / pl.col("count").sum()).log(),
            )
            .sort("item_id")
            .with_row_index("index")
            .collect()
        )
        missing = items.select("item_id").join(
            artifacts.items.select("item_id"),
            on="item_id",
            how="anti",
        )
        if not missing.is_empty():
            raise ValueError(
                "training targets are absent from the external candidate universe: "
                f"count={len(missing)}, "
                f"examples={missing['item_id'].head(20).to_list()}"
            )
        artifacts.items = artifacts.items.sort("item_id")
        for feature, observer in feature_to_observer.items():
            observer.observe(artifacts.items[feature])
        return cls(items)

    def static_transform(self, values: pl.Series, /) -> pl.Series:
        """Map item ID lists to train catalog index lists; unknown IDs become null."""

        return (
            values.to_frame("items")
            .lazy()
            .with_row_index("row_index")
            .explode("items", empty_as_null=True)
            .rename({"items": "item_id"})
            .join(
                self.items.lazy().select("item_id", item_index="index"),
                how="left",
                on="item_id",
                maintain_order="left",
            )
            .group_by("row_index", maintain_order=True)
            .agg(item_indices="item_index")
            .select("item_indices")
            .collect()
            .to_series()
        )

    def save(self, path: Path, /) -> None:
        """Save the indexed training catalog to items.pq."""

        path.mkdir(parents=True, exist_ok=True)
        self.items.write_parquet(path / "items.pq")

    @classmethod
    def load(cls, path: Path, /) -> Self:
        """Load the frozen training catalog and its indices."""

        return cls(pl.read_parquet(path / "items.pq"))
