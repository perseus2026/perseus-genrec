"""Host-memory-bounded mini-batch residual K-means SID builder."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator, Sequence
from typing import Self

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from item_representations.semantic_ids.models.base import ResidualQuantizer

logger = logging.getLogger(__name__)


class RQKMeans(ResidualQuantizer):
    """Fit residual codebooks without materializing the full matrix in RAM."""

    method = "rq_kmeans"

    def __init__(
        self,
        n_levels: int = 3,
        codebook_size: int | Sequence[int] = 64,
        random_state: int = 42,
        batch_size: int = 1024,
        max_iter: int = 100,
        n_init: int = 3,
        tol: float = 1e-4,
        device: str | None = None,
    ) -> None:
        """Configure streaming MiniBatchKMeans and the device used for SID transformation."""

        self._n_levels = n_levels
        self.codebook_size = self.normalize_codebook_size(
            codebook_size,
            n_levels,
        )
        self.random_state = random_state
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.codebooks: list[np.ndarray] = []
        self._codebooks_t: list[torch.Tensor] | None = None

    def fit(self, embeddings: np.ndarray, verbose: int = 0, **_: object) -> Self:
        """Fit residual codebooks from bounded CPU mini-batches.

        Do not load the complete embedding matrix into RAM. max_iter sets the mini-batch
        pass budget, not full Lloyd iterations on GPU.
        """

        if embeddings.ndim != 2 or len(embeddings) == 0:
            raise ValueError("embeddings must be a non-empty 2-D matrix")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")
        if len(embeddings) < max(self.codebook_size):
            raise ValueError(
                "number of embeddings must be at least the largest codebook: "
                f"rows={len(embeddings)}, codebook_size={max(self.codebook_size)}"
            )

        self.codebooks = []
        self._codebooks_t = None
        fitted_levels: list[MiniBatchKMeans] = []
        for level in range(self.n_levels):
            kmeans = MiniBatchKMeans(
                n_clusters=self.codebook_size[level],
                init="k-means++",
                n_init=self.n_init,
                # fit() is used only to initialize from one bounded batch.
                # The complete catalog is consumed below through partial_fit().
                max_iter=1,
                tol=self.tol,
                batch_size=self.batch_size,
                random_state=self.random_state + level,
                verbose=0,
                reassignment_ratio=0.01,
                max_no_improvement=10,
            )
            steps = self._fit_level(
                kmeans,
                embeddings,
                fitted_levels,
                level=level,
                verbose=verbose,
            )
            codebook = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
            self.codebooks.append(codebook)
            fitted_levels.append(kmeans)
            logger.info(
                "fitted RQ-KMeans level %s/%s in %s mini-batches",
                level + 1,
                self.n_levels,
                steps,
            )
            if verbose:
                tqdm.write(
                    f"Fitted RQ-KMeans level {level + 1}/{self.n_levels} "
                    f"in {steps} mini-batches"
                )
        return self

    def _fit_level(
        self,
        kmeans: MiniBatchKMeans,
        embeddings: np.ndarray,
        fitted_levels: list[MiniBatchKMeans],
        *,
        level: int,
        verbose: int,
    ) -> int:
        """Initialize and update one codebook from bounded contiguous batches."""

        rows = len(embeddings)
        batch_rows = min(max(self.batch_size, kmeans.n_clusters), rows)
        random_state = np.random.RandomState(self.random_state + level)
        batches_per_pass = math.ceil(rows / batch_rows)
        max_steps = max(1, self.max_iter * batches_per_pass)
        batches = self._batch_slices(
            rows,
            batch_rows,
            random_state=random_state,
        )

        initial_rows = next(batches)
        # A shuffled short tail cannot initialize K centroids. Updates may
        # still consume it after initialization; no catalog-sized copy is made.
        if initial_rows.stop - initial_rows.start < kmeans.n_clusters:
            initial_rows = slice(0, batch_rows)
        initial = self._residual_batch(
            embeddings,
            initial_rows,
            fitted_levels,
        )
        kmeans.fit(initial)
        tolerance = (
            float(np.var(initial, axis=0, dtype=np.float64).mean()) * self.tol
            if self.tol > 0
            else 0.0
        )
        ewa_inertia: float | None = None
        best_ewa_inertia: float | None = None
        no_improvement = 0
        steps = 1

        progress = tqdm(
            total=max_steps,
            initial=steps,
            desc=f"Fitting RQ-KMeans level {level + 1}/{self.n_levels}",
            leave=True,
            disable=not verbose,
            mininterval=10.0,
        )
        try:
            while steps < max_steps:
                residuals = self._residual_batch(
                    embeddings,
                    next(batches),
                    fitted_levels,
                )
                previous_centers = kmeans.cluster_centers_.copy()
                kmeans.partial_fit(residuals)
                steps += 1
                progress.update()

                centers_squared_diff = float(
                    np.square(kmeans.cluster_centers_ - previous_centers).sum()
                )
                if tolerance and centers_squared_diff <= tolerance:
                    break

                mean_inertia = float(kmeans.inertia_) / len(residuals)
                if ewa_inertia is None:
                    ewa_inertia = mean_inertia
                else:
                    alpha = min(len(residuals) * 2.0 / (rows + 1), 1.0)
                    ewa_inertia = ewa_inertia * (1.0 - alpha) + mean_inertia * alpha
                if best_ewa_inertia is None or ewa_inertia < best_ewa_inertia:
                    best_ewa_inertia = ewa_inertia
                    no_improvement = 0
                else:
                    no_improvement += 1
                    if no_improvement >= 10:
                        break
        finally:
            progress.close()
        return steps

    @staticmethod
    def _batch_slices(
        rows: int,
        batch_rows: int,
        *,
        random_state: np.random.RandomState,
    ) -> Iterator[slice]:
        """Yield shuffled contiguous blocks forever for efficient mmap reads."""

        blocks = math.ceil(rows / batch_rows)
        while True:
            for block in random_state.permutation(blocks):
                start = int(block) * batch_rows
                yield slice(start, min(start + batch_rows, rows))

    @staticmethod
    def _residual_batch(
        embeddings: np.ndarray,
        rows: slice,
        fitted_levels: list[MiniBatchKMeans],
    ) -> np.ndarray:
        """Load one writable float32 batch and subtract earlier codebooks."""

        residuals = np.array(
            embeddings[rows],
            dtype=np.float32,
            order="C",
            copy=True,
        )
        for fitted in fitted_levels:
            labels = fitted.predict(residuals)
            residuals -= fitted.cluster_centers_[labels]
        return residuals


__all__ = ["RQKMeans"]
