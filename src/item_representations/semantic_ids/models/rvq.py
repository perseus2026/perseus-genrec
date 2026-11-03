"""Layer-wise residual vector quantization of fixed item embeddings."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Literal, Self

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseSIDSGen
from .distributed import (
    check_loss,
    distributed_config,
    training_accelerator,
)
from .torch_kmeans import pairwise_squared_distances

logger = logging.getLogger(__name__)


def codebook_loss(residual, codebook, loss_scale, loss_reduction):
    ids = pairwise_squared_distances(residual.detach(), codebook.detach()).argmin(-1)
    error = (residual.detach() - codebook[ids]).square()
    return loss_scale * (error.sum() if loss_reduction == "sum" else error.mean())


class _TrainableCodebook(nn.Module):
    """Expose the current RVQ level's loss through DDP's forward hooks."""

    def __init__(self, centers, loss_scale, loss_reduction):
        super().__init__()
        self.centers = nn.Parameter(centers)
        self.loss_scale = loss_scale
        self.loss_reduction = loss_reduction

    def forward(self, residual):
        return codebook_loss(
            residual, self.centers, self.loss_scale, self.loss_reduction
        )


class RVQ(nn.Module, BaseSIDSGen):
    """Train one codebook at a time, optionally normalizing every residual.

    Fit uses finite shuffled epochs and retains the final partial batch. No
    autoencoder, commitment term or partial-training resume is involved.
    Under torchrun, whole per-device batches are sharded and padded by Accelerate.
    Reconstruction uses input-derived residual norms, not SID codes alone.
    """

    method = "rvq"

    def __init__(
        self,
        n_levels: int = 3,
        codebook_size: int | Sequence[int] = 256,
        *,
        normalize_residuals: bool = True,
        init_buffer_size: int = 3072,
        loss_reduction: Literal["sum", "mean"] = "sum",
        loss_scale: float = 1.0,
        random_state: int = 42,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self._n_levels = n_levels
        self.codebook_size = self.normalize_codebook_size(codebook_size, n_levels)
        if init_buffer_size < max(self.codebook_size):
            raise ValueError("init_buffer_size must be >= every codebook size")
        if loss_reduction not in {"sum", "mean"}:
            raise ValueError("loss_reduction must be sum or mean")
        if not math.isfinite(loss_scale) or loss_scale <= 0:
            raise ValueError("loss_scale must be finite and positive")
        self.normalize_residuals = normalize_residuals
        self.init_buffer_size = init_buffer_size
        self.loss_reduction = loss_reduction
        self.loss_scale = loss_scale
        self.random_state = random_state
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("RVQ supports a single CPU or CUDA device")
        self.codebooks = nn.ParameterList()
        self.fit_diagnostics: list[dict] = []
        self.config: dict = {}
        self._fitted = False

    @contextmanager
    def _fp32(self):
        # Matrix-product distances should not inherit mixed-precision autocast.
        previous = torch.backends.cuda.matmul.allow_tf32
        try:
            if self.device.type == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = False
            with torch.autocast(device_type=self.device.type, enabled=False):
                yield
        finally:
            if self.device.type == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = previous

    def _validate(self, embeddings: np.ndarray, *, fitting: bool = False) -> None:
        if embeddings.ndim != 2 or embeddings.shape[1] == 0:
            raise ValueError(
                "embeddings must be a two-dimensional nonempty-feature matrix"
            )
        if fitting:
            if len(embeddings) < max(self.codebook_size):
                raise ValueError(
                    "fit requires at least as many rows as codebook entries"
                )
        elif not self._fitted:
            raise RuntimeError("RVQ must be fitted before inference")
        elif embeddings.shape[1] != self.input_dim:
            raise ValueError("embedding dimension differs from fitted RVQ")
        for start in range(0, len(embeddings), 8192):
            batch = np.asarray(embeddings[start : start + 8192], dtype=np.float32)
            if not np.isfinite(batch).all():
                raise ValueError("embeddings must contain finite FP32 values")

    def _tensor(self, embeddings: np.ndarray) -> torch.Tensor:
        return torch.tensor(embeddings, dtype=torch.float32, device=self.device)

    def _normalize(self, residual: torch.Tensor) -> torch.Tensor:
        return F.normalize(residual, dim=-1) if self.normalize_residuals else residual

    def _residual(self, batch: torch.Tensor) -> torch.Tensor:
        """Get the active level's normalized input using frozen earlier levels."""
        with torch.no_grad():
            for codebook in self.codebooks:
                batch = self._normalize(batch)
                ids = pairwise_squared_distances(batch, codebook).argmin(-1)
                batch = batch - codebook[ids]
            return self._normalize(batch)

    def _initialize(
        self, points: torch.Tensor, size: int, generator: torch.Generator
    ) -> torch.Tensor:
        """Single-draw D² sampling; no local trials or Lloyd iterations."""
        with torch.no_grad():
            first = torch.randint(
                len(points), (1,), generator=generator, device=self.device
            )
            centers = [points[first].squeeze(0).clone()]
            nearest = (points - centers[0]).square().sum(-1)
            for _ in range(1, size):
                if nearest.sum() == 0:
                    index = torch.randint(
                        len(points), (1,), generator=generator, device=self.device
                    )
                else:
                    index = torch.multinomial(nearest, 1, generator=generator)
                center = points[index].squeeze(0).clone()
                centers.append(center)
                nearest = torch.minimum(nearest, (points - center).square().sum(-1))
            return torch.stack(centers)

    def _loss(self, residual: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        return codebook_loss(residual, codebook, self.loss_scale, self.loss_reduction)

    def fit(
        self,
        embeddings: np.ndarray,
        *,
        epochs_per_layer: int = 50,
        batch_size: int = 4096,
        learning_rate: float = 0.001,
        accelerator=None,
    ) -> Self:
        """Fit from scratch; initialization does not consume training updates."""
        self._validate(embeddings, fitting=True)
        if epochs_per_layer < 1 or batch_size < 1:
            raise ValueError("epochs_per_layer and batch_size must be positive")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        accelerator = accelerator or training_accelerator(self.device)
        self.device = accelerator.device
        self._fitted = False
        self.input_dim = embeddings.shape[1]
        self.codebooks = nn.ParameterList()
        self.fit_diagnostics = []
        shuffle = torch.Generator(device="cpu").manual_seed(self.random_state)
        # Reuse the loader while retaining the existing initialization/shuffle RNG order.
        order = torch.arange(len(embeddings))
        loader = DataLoader(
            TensorDataset(order),
            batch_size=batch_size,
            generator=torch.Generator().manual_seed(self.random_state),
        )
        loader = accelerator.prepare_data_loader(loader, device_placement=False)
        initialization = torch.Generator(device=self.device).manual_seed(
            self.random_state
        )
        buffer_rows = min(self.init_buffer_size, len(embeddings))
        with self._fp32():
            for level, size in enumerate(self.codebook_size):
                logger.info(
                    "RVQ level %d/%d: %d epochs, batch=%d, init_rows=%d",
                    level + 1,
                    self.n_levels,
                    epochs_per_layer,
                    batch_size,
                    buffer_rows,
                )
                indices = torch.randperm(len(embeddings), generator=shuffle)[
                    :buffer_rows
                ].numpy()
                if accelerator.is_main_process:
                    points = self._residual(self._tensor(embeddings[indices]))
                    centers = self._initialize(points, size, initialization)
                else:
                    centers = torch.empty(
                        size, self.input_dim, dtype=torch.float32, device=self.device
                    )
                level_model = _TrainableCodebook(
                    centers, self.loss_scale, self.loss_reduction
                )
                current = level_model.centers
                optimizer = torch.optim.Adam([current], lr=learning_rate)
                level_model, optimizer = accelerator.prepare(level_model, optimizer)
                updates = seen = 0
                last_epoch_loss = 0.0
                for _ in range(epochs_per_layer):
                    order.copy_(torch.randperm(len(embeddings), generator=shuffle))
                    epoch_loss = 0.0
                    for (indices,) in loader:
                        batch = self._tensor(embeddings[indices.numpy()])
                        residual = self._residual(batch)
                        optimizer.zero_grad(set_to_none=True)
                        loss = level_model(residual)
                        check_loss(loss, accelerator)
                        accelerator.backward(loss)
                        optimizer.step()
                        epoch_loss += loss.detach().item()
                        updates += 1
                        seen += len(batch)
                    epoch_loss = accelerator.reduce(
                        torch.tensor(
                            epoch_loss, dtype=torch.float64, device=self.device
                        ),
                        reduction="mean",
                    ).item()
                    last_epoch_loss = epoch_loss / len(loader)
                if not torch.isfinite(current).all():
                    raise FloatingPointError("Non-finite RVQ codebook")
                seen *= accelerator.num_processes
                current.requires_grad_(False)
                self.codebooks.append(current)
                self.fit_diagnostics.append(
                    {
                        "level": level + 1,
                        "completed_epochs": epochs_per_layer,
                        "optimizer_updates": updates,
                        "examples_seen": seen,
                        "initialization_rows": buffer_rows,
                        "last_epoch_mean_scaled_batch_loss": last_epoch_loss,
                    }
                )
                logger.info(
                    "RVQ level %d complete: updates=%d examples=%d loss=%g",
                    level + 1,
                    updates,
                    seen,
                    last_epoch_loss,
                )
        self.config = {
            "normalize_residuals": self.normalize_residuals,
            "init_buffer_size": self.init_buffer_size,
            "initializer": "kmeans++_single_draw",
            "loss_reduction": self.loss_reduction,
            "loss_scale": self.loss_scale,
            "random_state": self.random_state,
            "epochs_per_layer": epochs_per_layer,
            "training_batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": "adam",
            "precision": "float32",
            "tf32": False,
            "shuffle": True,
            "drop_last": False,
            "codebook_size": self.codebook_size,
            "input_dim": self.input_dim,
            **distributed_config(accelerator, batch_size),
        }
        self._fitted = True
        return self

    def _encode(self, embeddings: np.ndarray):
        self._validate(embeddings)
        with torch.no_grad(), self._fp32():
            residual = self._tensor(embeddings)
            reconstruction = torch.zeros_like(residual)
            scale = torch.ones((len(residual), 1), device=self.device)
            ids, squared_errors = [], []
            for codebook in self.codebooks:
                if self.normalize_residuals:
                    norm = residual.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    residual = residual / norm
                    scale = scale * norm
                index = pairwise_squared_distances(residual, codebook).argmin(-1)
                selected = codebook[index]
                reconstruction += scale * selected
                residual -= selected
                ids.append(index)
                squared_errors.append(residual.square().sum())
            return torch.stack(ids, -1), reconstruction, torch.stack(squared_errors)

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Return integer codes; batch boundaries do not affect residual normalization."""
        return self._encode(embeddings)[0].cpu().numpy()

    def reconstruct(self, embeddings: np.ndarray) -> np.ndarray:
        """Diagnostic reconstruction with input-derived norms, not SID-only decoding."""
        return self._encode(embeddings)[1].cpu().numpy()

    def residual_squared_errors(self, embeddings: np.ndarray) -> np.ndarray:
        """Per-level SSE after subtracting each centroid, before next normalization."""
        return self._encode(embeddings)[2].cpu().numpy()
