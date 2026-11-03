"""Feature encoder lifecycle shared by GenRecSys models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import polars as pl


class Observer(ABC):
    """Collect train-only statistics used to fit a preprocessor."""

    @abstractmethod
    def observe(self, values: pl.Series, /) -> None:
        """Accumulate statistics from a batch of training values."""
        ...

    @abstractmethod
    def merge(self, other: Self, /) -> None:
        """Combine train statistics without fitting a worker-local preprocessor."""
        ...

    @abstractmethod
    def save(self, path: Path, /) -> None:
        """Persist the worker's statistics for the shared fit stage."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path, /) -> Self:
        """Restore statistics in the encoder's own format."""
        ...


class Preprocessor(ABC):
    """Persisted static feature transformation; models collate their own tensors."""

    @classmethod
    @abstractmethod
    def fit(cls, observer: Observer, /, **kwargs: Any) -> Self:
        """Fit a value transformation from training statistics and encoder settings."""
        ...

    @abstractmethod
    def static_transform(self, values: pl.Series, /) -> pl.Series:
        """Transform values without changing the Series length or row order."""
        ...

    @abstractmethod
    def save(self, path: Path, /) -> None:
        """Save the fitted transformation for reuse."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path, /) -> Self:
        """Load a saved transformation without fitting it again."""
        ...


@dataclass(frozen=True)
class Encoder:
    """Implementations registered under ``encoders.<name>.type``."""

    observer: type[Observer]
    preprocessor: type[Preprocessor]
