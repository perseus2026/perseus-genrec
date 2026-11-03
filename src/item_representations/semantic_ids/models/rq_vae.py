"""Residual-quantized autoencoder semantic-ID builder."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from itertools import pairwise
from typing import Any, Literal, Self

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from item_representations.semantic_ids.models.base import BaseSIDSGen
from item_representations.semantic_ids.models.torch_kmeans import TorchKMeans

from .distributed import check_loss, distributed_config, training_accelerator

logger = logging.getLogger(__name__)


class L2NormalizationLayer(nn.Module):
    """Normalize the final tensor dimension with a stable epsilon."""

    def __init__(self, dim: int = -1, eps: float = 1e-12) -> None:
        """Set the L2 normalization dimension and epsilon."""

        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """L2-normalize x along the configured dimension without changing its shape."""

        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)


class MLP(nn.Module):
    """Small configurable MLP used only by RQ-VAE encoders/decoders."""

    def __init__(
        self,
        dims: list[int],
        dropout: float = 0.0,
        normalize: bool = False,
    ) -> None:
        """Build bias-free linear layers with hidden ReLU/dropout and optional output L2."""

        super().__init__()
        self.input_dim = dims[0]
        layers: list[nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(pairwise(dims)):
            layers.append(nn.Linear(input_dim, output_dim, bias=False))
            if index != len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout:
                    layers.append(nn.Dropout(dropout))
        if normalize:
            layers.append(L2NormalizationLayer())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Validate x's last dimension and apply the MLP."""

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Invalid input dim: expected {self.input_dim}, found {x.shape[-1]}"
            )
        return self.mlp(x)


class Quantizer(nn.Module):
    """One trainable residual codebook."""

    def __init__(
        self,
        embedding_dim: int,
        size: int,
        *,
        normalize: bool,
        beta: float,
        distance: Literal["l2", "cosine"],
    ) -> None:
        """Create a trainable codebook with distance, normalization, and commitment settings."""

        super().__init__()
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.distance = distance
        self.embedding = nn.Embedding(size, embedding_dim)
        self.output = L2NormalizationLayer() if normalize else nn.Identity()
        nn.init.uniform_(self.embedding.weight)

    def vectors(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up codebook vectors and optionally L2-normalize them."""

        return self.output(self.embedding(indices))

    def quantize(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return nearest code indices and vectors using L2 or cosine distance."""

        codebook = self.output(self.embedding.weight)
        if self.distance == "l2":
            distances = torch.cdist(inputs, codebook).square_()
        elif self.distance == "cosine":
            distances = -(F.normalize(inputs) @ F.normalize(codebook).T)
        else:  # pragma: no cover - constructor typing is not runtime validation
            raise ValueError(f"unknown distance mode: {self.distance}")
        indices = distances.detach().argmin(dim=1)
        return indices, self.vectors(indices)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return code indices, vectors, and per-row VQ loss.

        Use a straight-through estimator during training.
        """

        indices, vectors = self.quantize(inputs)
        codebook_loss = (inputs.detach() - vectors).square().mean(dim=-1)
        query_loss = (inputs - vectors.detach()).square().mean(dim=-1)
        if self.training:
            vectors = inputs + (vectors - inputs).detach()
        return indices, vectors, codebook_loss + self.beta * query_loss


class NormalizedInputEncoder(nn.Module):
    """Apply saved BatchNorm statistics and L2 before the encoder backbone."""

    def __init__(self, backbone: nn.Module, input_dim: int):
        super().__init__()
        self.normalization = nn.Sequential(
            nn.BatchNorm1d(input_dim), L2NormalizationLayer()
        )
        self.backbone = backbone

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.normalization(inputs))


class RQVAE(nn.Module, BaseSIDSGen):
    """Train an encoder, residual codebooks and decoder end to end."""

    method = "rq_vae"

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (256, 128, 64),
        codebook_size: Sequence[int] = (64, 64, 64),
        *,
        codebook_normalize: bool = False,
        codebook_kmeans_init: bool = True,
        codebook_distance_mode: Literal["l2", "cosine"] = "l2",
        beta: float = 0.25,
        random_state: int = 42,
        latent_l2: bool = True,
        mlp_bias: bool = False,
        joint_normalization: bool = False,
        optimizer_name: Literal["adam", "adagrad"] = "adam",
        gradient_clip: float | None = 1.0,
        device: str | None = None,
    ) -> None:
        """Seed weights locally; use the default recipe unless options are enabled."""

        super().__init__()
        if not codebook_size:
            raise ValueError("codebook_size must not be empty")
        if any(size < 2 for size in codebook_size):
            raise ValueError("every codebook_size must be at least 2")
        if beta < 0:
            raise ValueError("beta must be non-negative")
        if codebook_distance_mode not in {"l2", "cosine"}:
            raise ValueError("codebook_distance_mode must be l2 or cosine")

        if optimizer_name not in {"adam", "adagrad"}:
            raise ValueError("optimizer_name must be adam or adagrad")
        if gradient_clip is not None and (
            not math.isfinite(gradient_clip) or gradient_clip <= 0
        ):
            raise ValueError("gradient_clip must be finite and positive or None")
        if any(
            type(flag) is not bool
            for flag in (latent_l2, mlp_bias, joint_normalization)
        ):
            raise ValueError("Normalization and bias flags must be boolean")
        self.random_state = random_state
        self.latent_l2 = latent_l2
        self.mlp_bias = mlp_bias
        self.joint_normalization = joint_normalization
        self.optimizer_name = optimizer_name
        self.gradient_clip = gradient_clip
        self.input_dim = input_dim
        self.codebook_size = list(codebook_size)
        self._n_levels = len(self.codebook_size)
        self.codebooks_kmeans_init = codebook_kmeans_init
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        dimensions = [input_dim, *hidden_dims]
        self.latent_dim = dimensions[-1]
        # Seed before initialization without consuming the caller's CPU/CUDA RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(random_state)
            self.encoder = MLP(dimensions, normalize=True)
            self.codebooks = nn.ModuleList(
                Quantizer(
                    self.latent_dim,
                    size,
                    normalize=level == 0 and codebook_normalize,
                    beta=beta,
                    distance=codebook_distance_mode,
                )
                for level, size in enumerate(self.codebook_size)
            )
            self.decoder = MLP(list(reversed(dimensions)))
        if not latent_l2:
            self.encoder.mlp[-1] = nn.Identity()
        if mlp_bias:
            # Adding zero biases after weight initialization preserves the fitted
            # Amazon recipe's exact initial weights and RNG draw order.
            for network in (self.encoder, self.decoder):
                for layer in network.modules():
                    if isinstance(layer, nn.Linear):
                        layer.bias = nn.Parameter(
                            layer.weight.new_zeros(layer.out_features)
                        )
        if joint_normalization:
            self.encoder = NormalizedInputEncoder(self.encoder, input_dim)
        self.to(self.device)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        return_loss: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return codes/reconstructions or quantization/reconstruction losses.

        Move inputs to the model device and quantize residual levels sequentially.
        """

        inputs = inputs.to(self.device)
        target = inputs
        if self.joint_normalization:
            # One normalization call, shared differentiable input/target. Eval
            # and codebook initialization also go through this saved BN module.
            target = self.encoder.normalization(inputs)
            latent = self.encoder.backbone(target)
        else:
            latent = self.encoder(inputs)
        residual = latent
        quantized = torch.zeros_like(latent)
        codes: list[torch.Tensor] = []
        quantization_loss = torch.zeros((), device=self.device)
        for codebook in self.codebooks:
            indices, vectors, level_loss = codebook(residual)
            residual = residual - vectors
            quantized = quantized + vectors
            codes.append(indices)
            quantization_loss = quantization_loss + level_loss.mean()
        reconstruction = self.decoder(quantized)
        if return_loss:
            reconstruction_loss = (reconstruction - target).square().mean()
            return quantization_loss, reconstruction_loss
        return torch.stack(codes, dim=1), reconstruction

    def fit(
        self,
        embeddings: np.ndarray,
        *,
        epochs: int = 50,
        batch_size: int = 4096,
        learning_rate: float = 1e-4,
        verbose: int = 0,
        clearml_logger: Any | None = None,
        kmeans_init_max_samples: int = 200_000,
        kmeans_init_batch_size: int | None = None,
        kmeans_assignment_batch_size: int | None = 65_536,
        random_state: int | None = None,
        accelerator=None,
        **_: object,
    ) -> Self:
        """Train the autoencoder and codebooks with the configured optimizer.

        Read only the current batch from the array/mmap into FP32 device memory.
        Optional KMeans initialization also holds sampled latent vectors on the device.
        Batch size is per process; distributed epochs are padded by Accelerate.
        """

        if epochs < 1 or batch_size < 1 or learning_rate <= 0:
            raise ValueError("epochs, batch_size and learning_rate must be positive")
        if kmeans_init_batch_size is not None and kmeans_init_batch_size < 1:
            raise ValueError("kmeans_init_batch_size must be positive")
        random_state = self.random_state if random_state is None else random_state
        accelerator = accelerator or training_accelerator(self.device)
        self.device = accelerator.device
        self.to(self.device)
        loader = DataLoader(
            TensorDataset(torch.arange(len(embeddings))),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(random_state),
            pin_memory=self.device.type == "cuda",
        )
        if self.codebooks_kmeans_init and accelerator.is_main_process:
            self._init_codebooks(
                embeddings,
                max_samples=kmeans_init_max_samples,
                batch_size=kmeans_init_batch_size or batch_size,
                assignment_batch_size=kmeans_assignment_batch_size,
                random_state=random_state,
                verbose=verbose,
            )
        if accelerator.num_processes > 1:
            # Convert after rank-zero initialization, before DDP broadcasts weights/buffers.
            self.encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
        optimizer_type = {"adam": torch.optim.Adam, "adagrad": torch.optim.Adagrad}[
            self.optimizer_name
        ]
        optimizer = optimizer_type(self.parameters(), lr=learning_rate)
        train_model, optimizer = accelerator.prepare(self, optimizer)
        loader = accelerator.prepare_data_loader(loader, device_placement=False)
        updates = 0
        examples_seen = 0
        for epoch in range(epochs):
            totals = {"quantization": 0.0, "reconstruction": 0.0}
            seen = 0
            self.train()
            for (indices,) in loader:
                batch = torch.tensor(
                    embeddings[indices.numpy()],
                    dtype=torch.float32,
                    device=self.device,
                )
                optimizer.zero_grad(set_to_none=True)
                quantization, reconstruction = train_model(
                    batch,
                    return_loss=True,
                )
                loss = quantization + reconstruction
                check_loss(loss, accelerator)
                accelerator.backward(loss)
                clip = self.gradient_clip
                if clip is not None:
                    accelerator.clip_grad_norm_(self.parameters(), clip)
                optimizer.step()
                updates += 1
                seen += len(batch)
                totals["quantization"] += quantization.item() * len(batch)
                totals["reconstruction"] += reconstruction.item() * len(batch)
            reduced = accelerator.reduce(
                torch.tensor(
                    [*totals.values(), seen],
                    dtype=torch.float64,
                    device=self.device,
                ),
                reduction="sum",
            ).tolist()
            totals = dict(zip(totals, reduced[:-1]))
            seen = int(reduced[-1])
            examples_seen += seen
            metrics = {name: value / seen for name, value in totals.items()}
            main_process = accelerator.is_main_process
            if verbose and main_process:
                logger.info("RQVAE epoch %s/%s: %s", epoch + 1, epochs, metrics)
            if clearml_logger is not None and main_process:
                for name, value in metrics.items():
                    clearml_logger.report_scalar(
                        title=f"RQ-VAE {name} loss",
                        series="train",
                        value=value,
                        iteration=epoch + 1,
                    )
        self.fit_diagnostics = [
            {
                "completed_epochs": epochs,
                "optimizer_updates": updates,
                "examples_seen": examples_seen,
            }
        ]
        self.config = {
            "random_state": self.random_state,
            "fit_random_state": random_state,
            "latent_l2": self.latent_l2,
            "mlp_bias": self.mlp_bias,
            "joint_normalization": self.joint_normalization,
            "optimizer_name": self.optimizer_name,
            "gradient_clip": self.gradient_clip,
            "epochs": epochs,
            "training_batch_size": batch_size,
            "learning_rate": learning_rate,
            **distributed_config(accelerator, batch_size),
        }
        if kmeans_init_batch_size is not None:
            self.config["kmeans_init_batch_size"] = kmeans_init_batch_size
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Return NumPy codes shaped (rows, levels) for the input embedding batch."""

        codes, _ = self._eval_forward(embeddings)
        return codes.cpu().numpy()

    def reconstruct(self, embeddings: np.ndarray) -> np.ndarray:
        """Return reconstructed embeddings as a NumPy array."""

        _, reconstruction = self._eval_forward(embeddings)
        return reconstruction.cpu().numpy()

    def _eval_forward(
        self,
        embeddings: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run eval/no_grad inference with the complete input batch on the device."""

        self.eval()
        inputs = torch.tensor(
            embeddings,
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            result = self.forward(inputs)
        return result

    @torch.no_grad()
    def _init_codebooks(
        self,
        embeddings: np.ndarray,
        *,
        max_samples: int,
        batch_size: int,
        assignment_batch_size: int | None,
        random_state: int,
        verbose: int,
    ) -> None:
        """Initialize residual codebooks with KMeans on a seeded latent-vector sample.

        Encode in batches, then hold all sampled latents on the device. Assignment
        batching bounds distance buffers, not the sampled latent matrix.
        """

        if max_samples < max(self.codebook_size):
            raise ValueError("kmeans_init_max_samples is smaller than a codebook")
        generator = torch.Generator().manual_seed(random_state)
        indices = torch.randperm(len(embeddings), generator=generator)[:max_samples]
        latent_parts = [
            self.encoder(
                torch.tensor(
                    embeddings[indices[start : start + batch_size].numpy()],
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            for start in range(0, len(indices), batch_size)
        ]
        residual = torch.cat(latent_parts)
        for level, codebook in enumerate(self.codebooks):
            kmeans = TorchKMeans(
                n_clusters=self.codebook_size[level],
                n_init=1,
                max_iter=100,
                random_state=random_state + level,
                verbose=verbose,
                device=str(self.device),
                assignment_batch_size=assignment_batch_size,
            )
            labels = torch.as_tensor(
                kmeans.fit_predict(residual),
                device=self.device,
            )
            centers = torch.as_tensor(
                kmeans.cluster_centers_,
                device=self.device,
            )
            codebook.embedding.weight.copy_(centers)
            residual = residual - codebook.vectors(labels)


__all__ = ["RQVAE"]
