"""Deterministic random-hash baseline for semantic IDs."""

from __future__ import annotations

from typing import Self, Sequence

import numpy as np
from sklearn.utils import murmurhash3_32

from item_representations.semantic_ids.models.base import BaseSIDSGen


class RandomHashing(BaseSIDSGen):
    """Content-independent baseline with stable codes for identical vectors."""

    method = "random_hashing"

    def __init__(
        self,
        n_levels: int = 3,
        codebook_size: int | Sequence[int] = 64,
        random_state: int = 42,
    ) -> None:
        """Configure hash levels, bucket sizes, and the random seed."""

        self._n_levels = n_levels
        self.codebook_size = self.normalize_codebook_size(
            codebook_size,
            n_levels,
        )
        self.random_state = random_state
        self.codebooks: list[np.ndarray] = []

    def fit(self, embeddings: np.ndarray, **_: object) -> Self:
        """Create random reconstruction codebooks without learning embedding structure."""

        rng = np.random.default_rng(self.random_state)
        self.codebooks = [
            rng.standard_normal((size, embeddings.shape[1])).astype(embeddings.dtype)
            for size in self.codebook_size
        ]
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Hash each vector's FP32 bytes into stable codes shaped (rows, levels)."""

        rows = np.ascontiguousarray(embeddings, dtype=np.float32)
        codes = np.empty((len(rows), self.n_levels), dtype=np.int64)
        for row_index, row in enumerate(rows):
            payload = row.tobytes()
            for level, size in enumerate(self.codebook_size):
                codes[row_index, level] = murmurhash3_32(
                    payload,
                    seed=self.random_state + level,
                    positive=True,
                ) % size
        return codes

    def reconstruct(self, embeddings: np.ndarray) -> np.ndarray:
        """Sum random vectors for the hashed codes; requires fitted codebooks."""

        if not self.codebooks:
            raise RuntimeError("RandomHashing must be fitted before reconstruction")
        codes = self.transform(embeddings)
        reconstructed = np.zeros_like(embeddings)
        for level, codebook in enumerate(self.codebooks):
            reconstructed += codebook[codes[:, level]]
        return reconstructed


__all__ = ["RandomHashing"]
