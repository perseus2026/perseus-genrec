"""Model- and domain-independent data artifact commands."""

from pathlib import Path
from typing import Literal

from cyclopts import App

from item_representations.semantic_ids.mapping import (
    SIDMethod,
)
from item_representations.semantic_ids.mapping import (
    apply_sid_mapper as apply_sid_mapper_api,
)
from item_representations.semantic_ids.mapping import (
    extend_sid_mapping as extend_sid_mapping_api,
)
from item_representations.semantic_ids.mapping import (
    fit_sid_mapper as fit_sid_mapper_api,
)

app = App()


@app.command
def fit_sid_mapper(
    *,
    items: Path | None = None,
    embeddings: Path | None = None,
    output_dir: Path | None = None,
    workdir: Path = Path("workdir/"),
    method: SIDMethod = "rq_kmeans_full",
    codebook_size: tuple[int, ...] = (512, 512),
    balanced: tuple[int, ...] = (0,),
    random_state: int = 42,
    device: str | None = None,
    fit_backend: Literal["sklearn", "torch"] = "torch",
    fit_batch_size: int = 65_536,
    eval_batch_size: int = 25_000,
    max_iter: int = 100,
    n_init: int = 3,
    tolerance: float = 1e-4,
    hidden_dim_factors: tuple[float, ...] = (2.0, 4.0, 8.0),
    codebook_normalize: bool = False,
    codebook_kmeans_init: bool = True,
    codebook_distance_mode: Literal["l2", "cosine"] = "l2",
    beta: float = 0.25,
    epochs: int = 50,
    training_batch_size: int = 4096,
    kmeans_init_batch_size: int | None = None,
    learning_rate: float = 1e-4,
    overwrite: bool = False,
    max_gpu_memory_fraction: float | None = None,
    mapper_only: bool = False,
    epochs_per_layer: int = 50,
    normalize_residuals: bool = True,
    init_buffer_size: int = 3072,
    loss_reduction: Literal["sum", "mean"] = "sum",
    loss_scale: float = 1.0,
    latent_l2: bool = True,
    mlp_bias: bool = False,
    joint_normalization: bool = False,
    optimizer_name: Literal["adam", "adagrad"] = "adam",
    gradient_clip: float = 1.0,
) -> None:
    """Fit a selected SID builder and materialize a complete item mapping.

    With --workdir, read embeddings/train/{items.pq,embeddings.npy} and save
    to sid-fit/. Explicit paths override these defaults and remain relative to
    the current directory, as in commands without --workdir.

    RVQ/RQ-VAE use Accelerate, including ordinary single-process launches.
    For these methods, device selects CPU/CUDA; GPU selection belongs to the
    launcher or CUDA_VISIBLE_DEVICES, rather than the index in device=cuda:N.
    torchrun enables multiple processes; training-batch-size is per process.
    Distributed epochs are padded to equal batches; one process keeps the tail.
    """

    if any(value not in {0, 1} for value in balanced):
        raise ValueError("balanced values must be 0 or 1")

    fit_sid_mapper_api(
        items if items is not None else workdir / "embeddings/train/items.pq",
        embeddings
        if embeddings is not None
        else workdir / "embeddings/train/embeddings.npy",
        output_dir if output_dir is not None else workdir / "sid-fit",
        method=method,
        codebook_size=codebook_size,
        balanced=tuple(bool(value) for value in balanced),
        random_state=random_state,
        device=device,
        fit_backend=fit_backend,
        fit_batch_size=fit_batch_size,
        eval_batch_size=eval_batch_size,
        max_iter=max_iter,
        n_init=n_init,
        tolerance=tolerance,
        hidden_dim_factors=hidden_dim_factors,
        codebook_normalize=codebook_normalize,
        codebook_kmeans_init=codebook_kmeans_init,
        codebook_distance_mode=codebook_distance_mode,
        beta=beta,
        epochs=epochs,
        training_batch_size=training_batch_size,
        kmeans_init_batch_size=kmeans_init_batch_size,
        learning_rate=learning_rate,
        overwrite=overwrite,
        max_gpu_memory_fraction=max_gpu_memory_fraction,
        mapper_only=mapper_only,
        epochs_per_layer=epochs_per_layer,
        normalize_residuals=normalize_residuals,
        init_buffer_size=init_buffer_size,
        loss_reduction=loss_reduction,
        loss_scale=loss_scale,
        latent_l2=latent_l2,
        mlp_bias=mlp_bias,
        joint_normalization=joint_normalization,
        optimizer_name=optimizer_name,
        gradient_clip=None if gradient_clip == 0 else gradient_clip,
    )


@app.command
def apply_sid_mapper(
    *,
    checkpoint: Path | None = None,
    items: Path | None = None,
    embeddings: Path | None = None,
    output_dir: Path | None = None,
    workdir: Path = Path("workdir/"),
    eval_batch_size: int = 25_000,
    device: str | None = None,
    add_collision_level: bool = True,
    overwrite: bool = False,
) -> None:
    """Apply one frozen mapper to a complete catalog snapshot.

    With --workdir, load sid-fit/checkpoint.pkl, read
    embeddings/inference/{items.pq,embeddings.npy} and save to resources/sid/.
    Explicit paths override defaults and remain relative to the current directory.
    """

    apply_sid_mapper_api(
        checkpoint if checkpoint is not None else workdir / "sid-fit/checkpoint.pkl",
        items if items is not None else workdir / "embeddings/inference/items.pq",
        embeddings
        if embeddings is not None
        else workdir / "embeddings/inference/embeddings.npy",
        output_dir if output_dir is not None else workdir / "resources/sid",
        eval_batch_size=eval_batch_size,
        device=device,
        add_collision_level=add_collision_level,
        overwrite=overwrite,
    )


@app.command
def extend_sid_mapping(
    *,
    checkpoint: Path,
    existing_mapping: Path,
    contract: Path,
    items: Path,
    embeddings: Path,
    output_dir: Path,
    eval_batch_size: int = 25_000,
    audit_existing_embeddings: bool = False,
    overwrite: bool = False,
) -> None:
    """Append new items while preserving every existing SID."""

    extend_sid_mapping_api(
        checkpoint,
        existing_mapping,
        contract,
        items,
        embeddings,
        output_dir,
        eval_batch_size=eval_batch_size,
        audit_existing_embeddings=audit_existing_embeddings,
        overwrite=overwrite,
    )


__all__ = ["apply_sid_mapper", "extend_sid_mapping", "fit_sid_mapper"]
