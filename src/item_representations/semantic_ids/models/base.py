"""Shared contracts for semantic-ID builders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import numpy as np
import torch
from torch import nn

from item_representations.semantic_ids.models.torch_kmeans import (
    pairwise_squared_distances,
)


class BaseSIDSGen(ABC):
    """Persistable semantic-ID builder used by the fit/apply lifecycle."""

    method: str
    codebook_size: list[int]
    _n_levels: int

    @abstractmethod
    def fit(self, embeddings: np.ndarray, **kwargs: object) -> Self:
        """Fit the builder on an aligned embedding matrix."""

    @abstractmethod
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Return one integer code per item and level."""

    @abstractmethod
    def reconstruct(self, embeddings: np.ndarray) -> np.ndarray:
        """Return the builder's reconstruction of the input embeddings."""

    @property
    def n_levels(self) -> int:
        """Return the residual codebook depth, excluding the collision suffix."""

        return self._n_levels

    @staticmethod
    def normalize_codebook_size(
        value: int | Sequence[int],
        n_levels: int,
    ) -> list[int]:
        """Expand scalar/singleton codebook sizes to n_levels and require sizes >= 2."""

        if n_levels < 1:
            raise ValueError("n_levels must be positive")
        sizes = [value] * n_levels if isinstance(value, int) else list(value)
        if len(sizes) == 1:
            sizes *= n_levels
        if len(sizes) != n_levels:
            raise ValueError(
                f"codebook_size length {len(sizes)} must be 1 or "
                f"equal n_levels={n_levels}"
            )
        if any(size < 2 for size in sizes):
            raise ValueError("every codebook_size must be at least 2")
        return sizes

    def save(self, path: str | Path) -> None:
        """Save the SID builder with torch.save, creating parent directories."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, destination)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a trusted checkpoint through CPU and restore the model's usable device."""

        model = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(model, cls):
            raise TypeError(f"expected {cls.__name__}, got {type(model).__name__}")
        if isinstance(model, nn.Module):
            device = getattr(model, "device", torch.device("cpu"))
            if device.type == "cuda" and not torch.cuda.is_available():
                device = torch.device("cpu")
                model.device = device
            model.to(device)
        return model


class ResidualQuantizer(BaseSIDSGen):
    """Shared inference for residual codebook methods."""

    device: torch.device
    codebooks: list[np.ndarray]
    _codebooks_t: list[torch.Tensor] | None

    def _codebooks_to_device(self) -> list[torch.Tensor]:
        """Copy frozen FP32 codebooks to the current device on first use."""

        needs_rebuild = (
            self._codebooks_t is None
            or len(self._codebooks_t) != len(self.codebooks)
            or (bool(self._codebooks_t) and self._codebooks_t[0].device != self.device)
        )
        if needs_rebuild:
            self._codebooks_t = [
                torch.as_tensor(codebook, dtype=torch.float32, device=self.device)
                for codebook in self.codebooks
            ]
        return self._codebooks_t

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        # ``embeddings`` is normally a read-only mmap slice.  ``torch.tensor``
        # makes the bounded working copy explicit and avoids sharing a
        # non-writable NumPy buffer on CPU.
        """Assign nearest-codebook indices at each residual level.

        Return NumPy (rows, levels). Copy the input batch to the device; external batch
        sizes bound the rows * codebook_size distance matrix.
        """

        residuals = torch.tensor(
            embeddings,
            dtype=torch.float32,
            device=self.device,
        )
        indices_per_level: list[torch.Tensor] = []
        with torch.no_grad():
            for level, codebook in enumerate(self._codebooks_to_device()):
                indices = pairwise_squared_distances(
                    residuals,
                    codebook,
                ).argmin(dim=1)
                indices_per_level.append(indices)
                if level + 1 < self.n_levels:
                    residuals = residuals - codebook[indices]
        return torch.stack(indices_per_level, dim=1).cpu().numpy()

    def reconstruct(self, embeddings: np.ndarray) -> np.ndarray:
        """Sum selected codebook vectors to reconstruct the input embeddings."""

        indices = self.transform(embeddings)
        reconstructed = np.zeros_like(embeddings)
        for level, codebook in enumerate(self.codebooks):
            reconstructed += codebook[indices[:, level]]
        return reconstructed


__all__ = ["BaseSIDSGen", "ResidualQuantizer"]
