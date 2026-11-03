"""Full-batch residual K-means semantic-ID builder."""

import logging
from collections.abc import Sequence

import numpy as np
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm

from .base import ResidualQuantizer
from .torch_kmeans import TorchKMeans

logger = logging.getLogger(__name__)


class RQKMeansFull(ResidualQuantizer):
    """Build hierarchical SID codes with full-batch residual KMeans.

    Fit each codebook on the residuals left by the preceding level.
    """

    method = "rq_kmeans_full"

    def __init__(
        self,
        n_levels: int = 3,
        codebook_size: int | Sequence[int] = 64,
        random_state: int = 42,
        max_iter: int = 100,
        n_init: int = 3,
        tol: float = 1e-4,
        device: str | None = None,
        backend: str = "torch",
        balanced: bool | Sequence[bool] = False,
        fit_batch_size: int | None = None,
    ):
        """Configure full residual KMeans, including shared or per-level balance settings.

        A scalar codebook size applies to every level. device=None selects CUDA when
        available, otherwise CPU.
        """
        codebook_size = self.normalize_codebook_size(codebook_size, n_levels)

        if isinstance(balanced, bool):
            balanced = [balanced] * n_levels
        else:
            balanced = list(balanced)
            if len(balanced) == 1:
                balanced = balanced * n_levels
            elif len(balanced) != n_levels:
                raise ValueError(
                    f"balanced length {len(balanced)} must be 1 or "
                    f"equal n_levels={n_levels}"
                )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if backend not in {"sklearn", "torch"}:
            raise ValueError("backend must be 'sklearn' or 'torch'")
        if any(balanced) and backend != "torch":
            raise ValueError("balanced codebooks require backend='torch'")

        logger.info(
            "RQKMeansFull parameters: n_levels=%s codebook_size=%s max_iter=%s "
            "n_init=%s tol=%s device=%s backend=%s balanced=%s",
            n_levels,
            codebook_size,
            max_iter,
            n_init,
            tol,
            device,
            backend,
            balanced,
        )

        self._n_levels = n_levels
        self.codebook_size = codebook_size
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.device = torch.device(device)
        self.backend = backend
        self.balanced = balanced
        self.fit_batch_size = fit_batch_size
        self.codebooks: list[np.ndarray] = []
        self.fit_diagnostics: list[dict[str, float | int | str | bool]] = []

        # Cache tensor codebooks on the inference device.
        self._codebooks_t: list[torch.Tensor] | None = None

    def fit(self, X: np.ndarray, verbose: int = 0) -> "RQKMeansFull":
        """Fit residual codebooks to X (rows, features) and return self."""
        # The first level can consume the read-only mmap directly.  Subsequent
        # residual levels allocate only when their residual is actually needed.
        residuals = X
        self.codebooks = []
        self._codebooks_t = None
        self.fit_diagnostics = []

        for l in tqdm(range(self.n_levels), desc="Fitting RQ-KMeansFull", leave=False):
            if self.backend == "sklearn":
                kmeans = KMeans(
                    n_clusters=self.codebook_size[l],
                    init="k-means++",
                    n_init=self.n_init,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    random_state=self.random_state + l,
                    verbose=verbose,
                )
            else:
                kmeans = TorchKMeans(
                    n_clusters=self.codebook_size[l],
                    init="k-means++",
                    n_init=self.n_init,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    random_state=self.random_state + l,
                    verbose=verbose,
                    device=str(self.device),
                    balanced=self.balanced[l],
                    assignment_batch_size=self.fit_batch_size,
                )
            labels = kmeans.fit_predict(residuals)
            self.codebooks.append(kmeans.cluster_centers_)
            iterations = int(kmeans.n_iter_)
            inertia = float(kmeans.inertia_)
            stop_reason = getattr(kmeans, "stop_reason_", None)
            if stop_reason is None:
                stop_reason = "converged" if iterations < self.max_iter else "max_iter"
            diagnostic: dict[str, float | int | str | bool] = {
                "level": l + 1,
                "backend": self.backend,
                "items": len(residuals),
                "codebook_size": self.codebook_size[l],
                "iterations": iterations,
                "max_iter": self.max_iter,
                "stop_reason": str(stop_reason),
                "inertia": inertia,
                "mean_squared_distance": inertia / len(residuals),
            }
            center_shift = getattr(kmeans, "center_shift_sq_", None)
            if center_shift is not None:
                diagnostic["center_shift_sq"] = float(center_shift)
            self.fit_diagnostics.append(diagnostic)
            logger.info(
                "fitted RQ-KMeansFull level %s/%s: iterations=%s/%s "
                "stop_reason=%s inertia=%.6g mean_squared_distance=%.6g",
                l + 1,
                self.n_levels,
                iterations,
                self.max_iter,
                stop_reason,
                inertia,
                inertia / len(residuals),
            )
            # The last residual is never consumed. Skipping it avoids two
            # additional full-size host arrays for large embedding catalogs.
            if l + 1 < self.n_levels:
                residuals = residuals - self.codebooks[l][labels]

        return self


__all__ = ["RQKMeansFull"]
