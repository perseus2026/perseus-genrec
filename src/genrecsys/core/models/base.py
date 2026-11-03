"""Model implementation boundary for the shared recommendation lifecycle."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self

import polars as pl

if TYPE_CHECKING:
    from genrecsys.core.checkpoint import Prepared
    from genrecsys.core.config import Config
    from genrecsys.core.dataset import Dataset
    from genrecsys.core.event_store import EventStore


class TrainedModel(ABC):
    """Serializable state of one indivisible recommendation model."""

    @abstractmethod
    def save(self, path: Path, /) -> None:
        """Save model-owned checkpoint state to the given directory."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path, /) -> Self:
        """Load model-owned checkpoint state without training."""
        ...


class InferenceSession(Protocol):
    """Reusable model runtime for partitioned inference."""

    def iter_predictions(
        self,
        samples: pl.DataFrame,
        events: pl.DataFrame | EventStore,
        /,
    ) -> Iterator[pl.DataFrame]:
        """Yield recommendation batches for samples and their event history."""
        ...


@dataclass(frozen=True)
class DataRequirements:
    """Declarative compatibility contract checked at the model boundary."""

    artifact_columns: tuple[str, ...] = ("item_id",)
    checkpoint_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class Implementation:
    """Model plugin consumed by the shared lifecycle and data APIs."""

    trained_model: type[TrainedModel]
    fit_model: Callable[[Prepared, Dataset], TrainedModel]
    validate_config: Callable[[Config], None]
    data_requirements: Callable[[Config], DataRequirements]
    create_inference_session: Callable[..., InferenceSession]
    validate_prepared: Callable[[Prepared, Dataset], None] | None = None
