"""Perseus-compatible piecewise-linear encoding for numeric features."""

from __future__ import annotations

import bisect
import json
import math
from pathlib import Path
from typing import Self

import polars as pl
import torch

from genrecsys.core.encoders import base


class Observer(base.Observer):
    """Collect the exact Float32 value distribution without retaining duplicates."""

    def __init__(self) -> None:
        self.counts: dict[float, int] = {}
        self.count = 0

    def observe(self, values: pl.Series, /) -> None:
        """Accumulate finite train values and their exact occurrence counts."""

        numeric = values.cast(pl.Float32, strict=False).drop_nulls()
        numeric = numeric.filter(numeric.is_finite())
        for value, count in numeric.value_counts(sort=False).iter_rows():
            key = float(value)
            occurrences = int(count)
            self.counts[key] = self.counts.get(key, 0) + occurrences
            self.count += occurrences

    def merge(self, other: Self, /) -> None:
        for value, count in other.counts.items():
            self.counts[value] = self.counts.get(value, 0) + count
        self.count += other.count

    def save(self, path: Path, /) -> None:
        path.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {"value": list(self.counts), "count": list(self.counts.values())},
            schema={"value": pl.Float32, "count": pl.Int64},
        ).write_parquet(path / "counts.pq")

    @classmethod
    def load(cls, path: Path, /) -> Self:
        result = cls()
        result.counts = dict(pl.read_parquet(path / "counts.pq").iter_rows())
        result.count = sum(result.counts.values())
        return result


class Preprocessor(base.Preprocessor):
    """Store train-fitted quantile thresholds for dynamic PLE in the model."""

    def __init__(self, thresholds: list[float], /) -> None:
        if len(thresholds) < 2:
            raise ValueError("PLE requires at least two distinct thresholds")
        self.thresholds = thresholds

    @property
    def num_bins(self) -> int:
        """Return the number of PLE coordinates produced for one scalar."""

        return len(self.thresholds) - 1

    @classmethod
    def fit(
        cls,
        observer: Observer,
        /,
        *,
        max_bins: int = 256,
    ) -> Self:
        """Fit Perseus-style linear-interpolated quantile thresholds on train."""

        if not isinstance(max_bins, int) or isinstance(max_bins, bool) or max_bins <= 0:
            raise ValueError("PLE max_bins must be a positive integer")
        if not observer.count:
            raise ValueError("PLE cannot be fitted without finite train values")

        values = sorted(observer.counts)
        cumulative_counts: list[int] = []
        total = 0
        for value in values:
            total += observer.counts[value]
            cumulative_counts.append(total)

        def value_at(index: int) -> float:
            return values[bisect.bisect_right(cumulative_counts, index)]

        thresholds = []
        for bin_ in range(max_bins + 1):
            position = bin_ / max_bins * (observer.count - 1)
            index = int(position)
            weight = position - index
            left = value_at(index)
            threshold = (
                left if not weight else left + weight * (value_at(index + 1) - left)
            )
            thresholds.append(threshold)
        thresholds = sorted(set(thresholds))
        if len(thresholds) < 2:
            raise ValueError("PLE requires at least two distinct train values")
        if not all(math.isfinite(value) for value in thresholds):
            raise ValueError("PLE thresholds must be finite")
        return cls(thresholds)

    def static_transform(self, values: pl.Series, /) -> pl.Series:
        """Keep one scalar per row; expansion happens dynamically on the GPU."""

        return values.cast(pl.Float32, strict=False)

    def save(self, path: Path, /) -> None:
        """Write fitted thresholds in the same format as Perseus."""

        path.mkdir(parents=True, exist_ok=True)
        (path / "thresholds.json").write_text(
            json.dumps(self.thresholds),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path, /) -> Self:
        """Load fitted thresholds without observing validation or test."""

        return cls(json.loads((path / "thresholds.json").read_text(encoding="utf-8")))


def encode(values: torch.Tensor, thresholds: torch.Tensor, /) -> torch.Tensor:
    """Expand scalar values into Perseus piecewise-linear coordinates."""

    return (
        ((values - thresholds[:-1]) / (thresholds[1:] - thresholds[:-1]))
        .clamp(0.0, 1.0)
        .nan_to_num(0.0)
    )


encoder = base.Encoder(Observer, Preprocessor)
