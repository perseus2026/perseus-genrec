"""Train-fitted scalar numeric feature encoder."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Self

import polars as pl

from genrecsys.core.encoders import base


class Observer(base.Observer):
    def __init__(self) -> None:
        """Initialize count, sum, and squared-sum accumulators for training values."""

        self.count = 0
        self.total = 0.0
        self.total_squares = 0.0

    def observe(self, values: pl.Series, /) -> None:
        """Accumulate statistics from finite numeric values."""

        numeric = values.cast(pl.Float64, strict=False).drop_nulls()
        numeric = numeric.filter(numeric.is_finite())
        if numeric.is_empty():
            return
        self.count += len(numeric)
        self.total += float(numeric.sum())
        self.total_squares += float((numeric * numeric).sum())

    def merge(self, other: Self, /) -> None:
        self.count += other.count
        self.total += other.total
        self.total_squares += other.total_squares

    def save(self, path: Path, /) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "statistics.json").write_text(
            json.dumps(
                {
                    "count": self.count,
                    "total": self.total,
                    "total_squares": self.total_squares,
                }
            )
        )

    @classmethod
    def load(cls, path: Path, /) -> Self:
        state = json.loads((path / "statistics.json").read_text())
        result = cls()
        result.count = state["count"]
        result.total = state["total"]
        result.total_squares = state["total_squares"]
        return result


class Preprocessor(base.Preprocessor):
    def __init__(
        self,
        /,
        *,
        mean: float,
        scale: float,
        normalization: str,
        clip: float | None,
    ) -> None:
        """Store mean, scale, normalization mode, and optional clipping bounds."""

        self.mean = mean
        self.scale = scale
        self.normalization = normalization
        self.clip = clip

    @classmethod
    def fit(
        cls,
        observer: Observer,
        /,
        *,
        normalization: str = "standard",
        clip: float | None = None,
    ) -> Self:
        """Fit numeric normalization from training statistics."""

        if normalization not in {"none", "standard"}:
            raise ValueError("numeric normalization must be none or standard")
        if clip is not None and (not math.isfinite(clip) or clip <= 0):
            raise ValueError("numeric clip must be a positive finite number")
        mean = observer.total / observer.count if observer.count else 0.0
        variance = (
            max(observer.total_squares / observer.count - mean * mean, 0.0)
            if observer.count
            else 0.0
        )
        scale = math.sqrt(variance) or 1.0
        return cls(
            mean=mean,
            scale=scale,
            normalization=normalization,
            clip=clip,
        )

    def static_transform(self, values: pl.Series, /) -> pl.Series:
        """Return Float32 values with missing-value imputation, normalization, and clipping."""

        numeric = values.cast(pl.Float32, strict=False)
        transformed = numeric.set(
            ~numeric.is_finite().fill_null(False), self.mean
        ).fill_null(self.mean)
        if self.normalization == "standard":
            transformed = (transformed - self.mean) / self.scale
        if self.clip is not None:
            transformed = transformed.clip(-self.clip, self.clip)
        return transformed.fill_nan(0.0).rename(values.name)

    def save(self, path: Path, /) -> None:
        """Write numeric transformation settings to config.json."""

        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(
            json.dumps(
                {
                    "mean": self.mean,
                    "scale": self.scale,
                    "normalization": self.normalization,
                    "clip": self.clip,
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path, /) -> Self:
        """Load numeric transformation settings from config.json."""

        return cls(**json.loads((path / "config.json").read_text(encoding="utf-8")))


encoder = base.Encoder(Observer, Preprocessor)
