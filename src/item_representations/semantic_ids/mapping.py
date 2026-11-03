"""Explicit fit/apply/extend lifecycle for semantic-ID mappings."""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import torch

from genrecsys.utils.files import file_sha256, publish_directory
from genrecsys.utils.progress import logged_stage
from item_representations.semantic_ids.models import (
    RQVAE,
    RVQ,
    BaseSIDSGen,
    RandomHashing,
    RQKMeans,
    RQKMeansFull,
)
from item_representations.semantic_ids.models.base import ResidualQuantizer
from item_representations.semantic_ids.models.distributed import training_accelerator

SIDMethod = Literal["random_hashing", "rq_kmeans", "rq_kmeans_full", "rq_vae", "rvq"]
logger = logging.getLogger(__name__)


@logged_stage
def fit_sid_mapper(
    items_path: str | Path,
    embeddings_path: str | Path,
    output_dir: str | Path,
    /,
    *,
    method: SIDMethod = "rq_kmeans_full",
    codebook_size: tuple[int, ...] = (512, 512),
    balanced: tuple[bool, ...] = (False,),
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
    add_collision_level: bool = True,
    overwrite: bool = False,
    max_gpu_memory_fraction: float | None = None,
    epochs_per_layer: int = 50,
    normalize_residuals: bool = True,
    init_buffer_size: int = 3072,
    loss_reduction: Literal["sum", "mean"] = "sum",
    loss_scale: float = 1.0,
    latent_l2: bool = True,
    mlp_bias: bool = False,
    joint_normalization: bool = False,
    optimizer_name: Literal["adam", "adagrad"] = "adam",
    gradient_clip: float | None = 1.0,
    mapper_only: bool = False,
) -> dict[str, Any]:
    """Fit a SID mapper; optionally defer all item-code application to preparation."""

    if method not in {"rvq", "rq_vae"} and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("Distributed SID fit supports only rvq and rq_vae")
    accelerator = training_accelerator(device) if method in {"rvq", "rq_vae"} else None
    if accelerator is not None:
        device = str(accelerator.device)
    items, embeddings = _load_aligned(items_path, embeddings_path)
    model = _fit_builder(
        method,
        embeddings,
        codebook_size=codebook_size,
        balanced=balanced,
        random_state=random_state,
        device=device,
        fit_backend=fit_backend,
        fit_batch_size=fit_batch_size,
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
        max_gpu_memory_fraction=max_gpu_memory_fraction,
        epochs_per_layer=epochs_per_layer,
        normalize_residuals=normalize_residuals,
        init_buffer_size=init_buffer_size,
        loss_reduction=loss_reduction,
        loss_scale=loss_scale,
        latent_l2=latent_l2,
        mlp_bias=mlp_bias,
        joint_normalization=joint_normalization,
        optimizer_name=optimizer_name,
        gradient_clip=gradient_clip,
        accelerator=accelerator,
    )
    if accelerator is not None and accelerator.num_processes > 1:
        torch.distributed.barrier()
        if not accelerator.is_main_process:
            return {"operation": "fit", "method": method, "published": False}
    # Only rank zero evaluates the full catalog, resolves collisions and publishes.
    base_codes, fit_report = _fit_quality_report(
        model,
        embeddings,
        batch_size=eval_batch_size,
    )
    mapping = (
        None
        if mapper_only
        else _mapping_frame(
            model, items, base_codes, add_collision_level=add_collision_level
        )
    )
    return _publish(
        output_dir,
        model=model,
        mapping=mapping,
        overwrite=overwrite,
        quality_report=fit_report,
        metadata={
            "operation": "fit",
            "method": model.method,
            "items": len(items),
            "embedding_shape": list(embeddings.shape),
            "codebook_size": list(codebook_size),
            "mapping_deferred": mapper_only,
            "collision_level": add_collision_level if not mapper_only else None,
            "device": device or "auto",
            "fit_batch_size": fit_batch_size,
            "fit_backend": fit_backend,
            "max_iter": max_iter,
            "n_init": n_init,
            "tolerance": tolerance,
            "random_state": random_state,
            **({model.method: model.config} if isinstance(model, (RVQ, RQVAE)) else {}),
            "items_sha256": file_sha256(items_path),
            "embeddings_sha256": file_sha256(embeddings_path),
        },
    )


@logged_stage
def apply_sid_mapper(
    checkpoint_path: str | Path,
    items_path: str | Path,
    embeddings_path: str | Path,
    output_dir: str | Path,
    /,
    *,
    eval_batch_size: int = 25_000,
    add_collision_level: bool = True,
    overwrite: bool = False,
    device: str | None = None,
) -> dict[str, Any]:
    """Apply a frozen SID mapper to one complete candidate snapshot."""

    items, embeddings = _load_aligned(items_path, embeddings_path)
    model = BaseSIDSGen.load(checkpoint_path)
    if device is not None:
        import torch

        model.device = torch.device(device)
        if isinstance(model, torch.nn.Module):
            model.to(model.device)
    mapping = _mapping_frame(
        model,
        items,
        _transform(model, embeddings, eval_batch_size),
        add_collision_level=add_collision_level,
    )
    return _publish(
        output_dir,
        model=None,
        checkpoint_path=Path(checkpoint_path),
        mapping=mapping,
        overwrite=overwrite,
        metadata={
            "operation": "apply",
            "method": model.method,
            "items": len(items),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "collision_level": add_collision_level,
        },
    )


def extend_sid_mapping(
    checkpoint_path: str | Path,
    existing_mapping_path: str | Path,
    contract_path: str | Path,
    items_path: str | Path,
    embeddings_path: str | Path,
    output_dir: str | Path,
    /,
    *,
    eval_batch_size: int = 25_000,
    audit_existing_embeddings: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Append new items without changing persisted SID rows.

    Infer new items only by default. audit_existing_embeddings additionally
    replays the full catalog through the checkpoint.
    """

    items, embeddings = _load_aligned(items_path, embeddings_path)
    existing_path = Path(existing_mapping_path)
    _validate_extension_contract(
        Path(contract_path),
        checkpoint_path=Path(checkpoint_path),
        mapping_path=existing_path,
    )
    existing = _normalise_mapping(pl.read_parquet(existing_path))
    model = BaseSIDSGen.load(checkpoint_path)
    new_mask = items["item_id"].is_in(existing["item_id"].implode()).not_()
    if new_mask.any():
        new_items = items.filter(new_mask)
        new_embeddings = embeddings[np.flatnonzero(new_mask.to_numpy())]
        new_base = _mapping_frame(
            model,
            new_items,
            _transform(model, new_embeddings, eval_batch_size),
            add_collision_level=False,
        )
        appended = _append_collision_level(existing, new_base)
        mapping = pl.concat([existing, appended]).sort("item_id")
    else:
        appended = existing.head(0)
        mapping = existing

    audit = {"enabled": audit_existing_embeddings, "checked": 0}
    if audit_existing_embeddings:
        audit = _audit_existing_prefixes(
            model,
            existing,
            items,
            embeddings,
            eval_batch_size,
        )
    return _publish(
        output_dir,
        model=None,
        checkpoint_path=Path(checkpoint_path),
        mapping=mapping,
        overwrite=overwrite,
        metadata={
            "operation": "extend",
            "method": model.method,
            "existing_items": len(existing),
            "appended_items": len(appended),
            "items": len(mapping),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "existing_mapping_sha256": file_sha256(existing_path),
            "existing_rows_unchanged": True,
            "full_catalog_audit": audit,
        },
    )


def _load_aligned(
    items_path: str | Path,
    embeddings_path: str | Path,
) -> tuple[pl.DataFrame, np.ndarray]:
    """Load item IDs and row-aligned NPY embeddings.

    Sorted input preserves mmap; sorting unsorted items copies the embedding matrix
    into host RAM.
    """

    items = pl.read_parquet(items_path)
    if "item_id" not in items.columns:
        raise ValueError("SID items must contain canonical item_id")
    items = items.select("item_id")
    if items["item_id"].null_count() or items["item_id"].n_unique() != len(items):
        raise ValueError("SID items must contain unique non-null item_id values")
    embeddings = np.load(embeddings_path, mmap_mode="r")
    if embeddings.ndim != 2 or len(embeddings) != len(items):
        raise ValueError(
            f"items/embeddings mismatch: rows={len(items)}, shape={embeddings.shape}"
        )
    if items["item_id"].is_sorted():
        # Stage 05 guarantees sorted, row-aligned output. Keep that matrix as a
        # mmap instead of materializing the whole catalog in host memory.
        return items, embeddings

    items = items.with_row_index("__source_row").sort("item_id")
    order = items["__source_row"].to_numpy()
    items = items.drop("__source_row")
    # Preserve the historical deterministic item_id ordering for callers that
    # provide an unsorted catalog. Production inputs never take this fallback.
    return items, np.asarray(embeddings[order])


def _transform(model: Any, embeddings: np.ndarray, batch_size: int) -> np.ndarray:
    """Apply a frozen SID builder in batches and collect the complete code matrix."""

    if batch_size < 1:
        raise ValueError("eval_batch_size must be positive")
    return np.vstack(
        [
            model.transform(embeddings[start : start + batch_size])
            for start in range(0, len(embeddings), batch_size)
        ]
    )


def _fit_quality_report(
    model: BaseSIDSGen,
    embeddings: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return codes and bounded diagnostics: reconstruction error and code usage."""
    if batch_size < 1:
        raise ValueError("eval_batch_size must be positive")
    started = time.perf_counter()
    counts = [np.zeros(size, dtype=np.int64) for size in model.codebook_size]
    code_parts = []
    squared_error = input_squared_norm = cosine_sum = 0.0
    rvq_squared_errors = np.zeros(model.n_levels, dtype=np.float64)
    for start in range(0, len(embeddings), batch_size):
        batch = np.asarray(embeddings[start : start + batch_size], dtype=np.float32)
        codes = np.asarray(model.transform(batch), dtype=np.int64)
        if codes.shape != (len(batch), model.n_levels):
            raise ValueError("SID code shape does not match evaluation batch/model")
        code_parts.append(codes)
        for level, level_counts in enumerate(counts):
            level_counts += np.bincount(codes[:, level], minlength=len(level_counts))
        if isinstance(model, ResidualQuantizer):
            reconstruction = np.zeros_like(batch)
            for level, codebook in enumerate(model.codebooks):
                reconstruction += np.asarray(codebook, dtype=np.float32)[
                    codes[:, level]
                ]
        else:
            reconstruction = np.asarray(model.reconstruct(batch), dtype=np.float32)
        if reconstruction.shape != batch.shape:
            raise ValueError("SID reconstruction shape does not match evaluation batch")
        difference = batch - reconstruction
        if isinstance(model, RVQ):
            rvq_squared_errors += model.residual_squared_errors(batch)
        squared_error += float(np.einsum("ij,ij->", difference, difference))
        input_squared_norm += float(np.einsum("ij,ij->", batch, batch))
        denominator = np.linalg.norm(batch, axis=1) * np.linalg.norm(
            reconstruction, axis=1
        )
        cosine_sum += float(
            np.divide(
                np.einsum("ij,ij->i", batch, reconstruction),
                denominator,
                out=np.zeros(len(batch), dtype=np.float32),
                where=denominator > 0,
            ).sum(dtype=np.float64)
        )
    report = {
        "version": 2,
        "method": model.method,
        "items": len(embeddings),
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "evaluation_batch_size": batch_size,
        "elapsed_seconds": time.perf_counter() - started,
        "reconstruction": {
            "rmse": math.sqrt(squared_error / embeddings.size),
            "normalized_rmse": math.sqrt(squared_error / input_squared_norm)
            if input_squared_norm
            else 0.0,
            "cosine_mean": cosine_sum / len(embeddings),
        },
        "levels": [
            {
                "level": level,
                "codebook_size": len(level_counts),
                "active_codes": int(np.count_nonzero(level_counts)),
                "max_code_share": float(level_counts.max() / level_counts.sum()),
            }
            for level, level_counts in enumerate(counts, 1)
        ],
        "fit": list(getattr(model, "fit_diagnostics", ())),
    }
    if isinstance(model, RVQ):
        report["rvq"] = model.config
        report["reconstruction"].update(
            kind="input_norm_assisted" if model.normalize_residuals else "codebook_sum",
            sid_only=not model.normalize_residuals,
            warning="Normalized RVQ reconstruction uses input-derived residual norms; do not rank against raw RQ-VAE reconstruction MSE."
            if model.normalize_residuals
            else None,
        )
        report["quantization"] = {
            "space": "levelwise_normalized_residual"
            if model.normalize_residuals
            else "raw_residual",
            "per_level_mse": (rvq_squared_errors / embeddings.size).tolist(),
            "final_residual_mse": float(rvq_squared_errors[-1] / embeddings.size),
        }
    if isinstance(model, RQVAE) and model.joint_normalization:
        report["reconstruction"].update(
            target="raw_input_diagnostic",
            training_target="batchnorm_l2_input",
            warning="Raw-input reconstruction error is not the normalized training objective.",
        )
    logger.info("SID fit quality: %s", report)
    return np.vstack(code_parts), report


def _fit_builder(
    method: SIDMethod,
    embeddings: np.ndarray,
    *,
    codebook_size: tuple[int, ...],
    balanced: tuple[bool, ...],
    random_state: int,
    device: str | None,
    fit_backend: Literal["sklearn", "torch"],
    fit_batch_size: int,
    max_iter: int,
    n_init: int,
    tolerance: float,
    hidden_dim_factors: tuple[float, ...],
    codebook_normalize: bool,
    codebook_kmeans_init: bool,
    codebook_distance_mode: Literal["l2", "cosine"],
    beta: float,
    epochs: int,
    training_batch_size: int,
    learning_rate: float,
    max_gpu_memory_fraction: float | None = None,
    epochs_per_layer: int = 50,
    normalize_residuals: bool = True,
    init_buffer_size: int = 3072,
    loss_reduction: Literal["sum", "mean"] = "sum",
    loss_scale: float = 1.0,
    latent_l2: bool = True,
    mlp_bias: bool = False,
    joint_normalization: bool = False,
    optimizer_name: Literal["adam", "adagrad"] = "adam",
    gradient_clip: float | None = 1.0,
    accelerator=None,
    kmeans_init_batch_size: int | None = None,
) -> BaseSIDSGen:
    """Build and fit the selected SID method.

    Full Torch KMeans may check the estimated GPU working set before fitting;
    it does not switch automatically to mini-batch fitting.
    """

    if not codebook_size:
        raise ValueError("codebook_size must not be empty")
    if max_gpu_memory_fraction is not None:
        import torch

        if not 0 < max_gpu_memory_fraction < 1:
            raise ValueError("max_gpu_memory_fraction must be between 0 and 1")
        target = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if (
            method == "rq_kmeans_full"
            and fit_backend == "torch"
            and target.type == "cuda"
        ):
            # Full Lloyd fitting keeps FP32 X on GPU; assignment batching alone
            # cannot make an arbitrarily large matrix fit. Include working copies.
            estimated = (
                3 * embeddings.size * 4 + 2 * fit_batch_size * max(codebook_size) * 4
            )
            free, _ = torch.cuda.mem_get_info(target)
            logger.info(
                "full KMeans GPU memory preflight: estimated_gib=%.2f "
                "free_gib=%.2f budget_gib=%.2f fraction=%.2f",
                estimated / 2**30,
                free / 2**30,
                free * max_gpu_memory_fraction / 2**30,
                max_gpu_memory_fraction,
            )
            if estimated > free * max_gpu_memory_fraction:
                raise MemoryError(
                    f"full KMeans estimated working set {estimated / 2**30:.2f} GiB exceeds "
                    f"GPU budget {free * max_gpu_memory_fraction / 2**30:.2f} GiB; "
                    "use the batched rq_kmeans method explicitly or a larger device"
                )
    if method == "random_hashing":
        model: BaseSIDSGen = RandomHashing(
            n_levels=len(codebook_size),
            codebook_size=codebook_size,
            random_state=random_state,
        )
    elif method == "rq_kmeans":
        model = RQKMeans(
            n_levels=len(codebook_size),
            codebook_size=codebook_size,
            random_state=random_state,
            batch_size=fit_batch_size,
            max_iter=max_iter,
            n_init=n_init,
            tol=tolerance,
            device=device,
        )
    elif method == "rq_kmeans_full":
        model = RQKMeansFull(
            n_levels=len(codebook_size),
            codebook_size=codebook_size,
            balanced=balanced,
            random_state=random_state,
            max_iter=max_iter,
            n_init=n_init,
            tol=tolerance,
            device=device,
            backend=fit_backend,
            fit_batch_size=fit_batch_size,
        )
    elif method == "rvq":
        if fit_backend != "torch":
            raise ValueError("RVQ supports only fit_backend=torch")
        model = RVQ(
            n_levels=len(codebook_size),
            codebook_size=codebook_size,
            normalize_residuals=normalize_residuals,
            init_buffer_size=init_buffer_size,
            loss_reduction=loss_reduction,
            loss_scale=loss_scale,
            random_state=random_state,
            device=device,
        )
        return model.fit(
            embeddings,
            epochs_per_layer=epochs_per_layer,
            batch_size=training_batch_size,
            learning_rate=learning_rate,
            accelerator=accelerator,
        )
    elif method == "rq_vae":
        if not hidden_dim_factors or any(factor <= 0 for factor in hidden_dim_factors):
            raise ValueError("hidden_dim_factors must contain positive values")
        model = RQVAE(
            input_dim=embeddings.shape[1],
            hidden_dims=tuple(
                max(1, int(embeddings.shape[1] / factor))
                for factor in hidden_dim_factors
            ),
            codebook_size=codebook_size,
            codebook_normalize=codebook_normalize,
            codebook_kmeans_init=codebook_kmeans_init,
            codebook_distance_mode=codebook_distance_mode,
            beta=beta,
            random_state=random_state,
            latent_l2=latent_l2,
            mlp_bias=mlp_bias,
            joint_normalization=joint_normalization,
            optimizer_name=optimizer_name,
            gradient_clip=gradient_clip,
            device=device,
        )
        model.fit(
            embeddings,
            epochs=epochs,
            batch_size=training_batch_size,
            learning_rate=learning_rate,
            random_state=random_state,
            kmeans_assignment_batch_size=fit_batch_size,
            kmeans_init_batch_size=kmeans_init_batch_size,
            verbose=1,
            accelerator=accelerator,
        )
        return model
    else:
        raise ValueError(f"unknown SID method: {method}")
    model.fit(embeddings, verbose=1)
    return model


def _mapping_frame(
    model: BaseSIDSGen,
    items: pl.DataFrame,
    codes: np.ndarray,
    *,
    add_collision_level: bool,
) -> pl.DataFrame:
    """Build an item-to-SID table with separate token ranges per level.

    Reserve token IDs 0/1. An optional collision suffix ensures unique codes in
    stable item ID order.
    """

    if codes.ndim != 2 or codes.shape != (len(items), model.n_levels):
        raise ValueError(
            "SID code shape does not match items/model: "
            f"codes={codes.shape}, items={len(items)}, levels={model.n_levels}"
        )
    sid_columns = [f"__sid_{level}" for level in range(model.n_levels)]
    frame = items.with_columns(
        pl.Series(column, codes[:, level]).cast(pl.Int64)
        + sum(model.codebook_size[:level])
        + 2
        for level, column in enumerate(sid_columns)
    )
    if add_collision_level:
        collision = "__sid_collision"
        frame = frame.with_columns(
            (
                pl.int_range(pl.len()).over(*sid_columns) + sum(model.codebook_size) + 2
            ).alias(collision)
        )
        sid_columns.append(collision)
    return frame.select(
        "item_id",
        pl.concat_list(sid_columns).alias("sid"),
    ).sort("item_id")


def _normalise_mapping(frame: pl.DataFrame) -> pl.DataFrame:
    """Validate unique items/codes and a common SID width, then sort the mapping."""

    if set(frame.columns) != {"item_id", "sid"}:
        frame = frame.select("item_id", "sid")
    if frame["item_id"].null_count() or frame["item_id"].n_unique() != len(frame):
        raise ValueError("existing item_to_sid has invalid item_id values")
    lengths = frame["sid"].list.len().unique().to_list()
    if len(lengths) != 1 or int(lengths[0]) < 2:
        raise ValueError("existing SIDs must have one width including collision level")
    if frame["sid"].n_unique() != len(frame):
        raise ValueError("existing SIDs are not unique")
    return frame.sort("item_id")


def _append_collision_level(
    existing: pl.DataFrame,
    new_base: pl.DataFrame,
) -> pl.DataFrame:
    """Append collision suffixes for new items without changing existing SID codes."""

    width = int(existing["sid"].list.len().first())
    prefix_width = width - 1
    if int(new_base["sid"].list.len().first()) != prefix_width:
        raise ValueError("checkpoint base SID width differs from existing mapping")
    existing_sids = existing["sid"].to_list()
    collision_offset = min(int(sid[-1]) for sid in existing_sids)
    maxima: dict[tuple[int, ...], int] = {}
    for sid in existing_sids:
        prefix = tuple(int(value) for value in sid[:-1])
        maxima[prefix] = max(maxima.get(prefix, -1), int(sid[-1]))
    assigned: list[list[int]] = []
    new_base = new_base.sort("item_id")
    for sid in new_base["sid"].to_list():
        prefix = tuple(int(value) for value in sid)
        collision = maxima.get(prefix, collision_offset - 1) + 1
        maxima[prefix] = collision
        assigned.append([*prefix, collision])
    return new_base.with_columns(pl.Series("sid", assigned))


def _audit_existing_prefixes(
    model: Any,
    existing: pl.DataFrame,
    items: pl.DataFrame,
    embeddings: np.ndarray,
    batch_size: int,
) -> dict[str, Any]:
    """Verify that the frozen builder reproduces every existing item's SID prefix.

    Requires existing item embeddings; indexed selection copies the audit matrix
    to host RAM.
    """

    positions = {item: index for index, item in enumerate(items["item_id"].to_list())}
    missing = [item for item in existing["item_id"].to_list() if item not in positions]
    if missing:
        examples = missing[:20]
        raise ValueError(
            f"full SID audit misses {len(missing)} existing embeddings; "
            f"examples={examples}"
        )
    indices = np.asarray([positions[item] for item in existing["item_id"].to_list()])
    base = _mapping_frame(
        model,
        existing.select("item_id"),
        _transform(model, embeddings[indices], batch_size),
        add_collision_level=False,
    )
    actual = dict(base.iter_rows())
    changed = [
        item
        for item, sid in existing.iter_rows()
        if tuple(sid[:-1]) != tuple(actual[item])
    ]
    if changed:
        raise ValueError(
            f"checkpoint does not reproduce {len(changed)} existing SID prefixes; "
            f"examples={changed[:20]}"
        )
    return {"enabled": True, "checked": len(existing), "changed": 0}


def _publish(
    output_dir: str | Path,
    *,
    model: Any | None,
    mapping: pl.DataFrame | None,
    overwrite: bool,
    metadata: dict[str, Any],
    checkpoint_path: Path | None = None,
    quality_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish a mapper, optional bijective mapping and its immutable hashes."""
    if mapping is not None:
        for column in ("item_id", "sid"):
            if mapping[column].null_count() or mapping[column].n_unique() != len(
                mapping
            ):
                raise ValueError(f"SID mapping must contain unique non-null {column}")
    with publish_directory(output_dir, overwrite=overwrite) as staging:
        target_checkpoint = staging / "checkpoint.pkl"
        if model is not None:
            model.save(str(target_checkpoint))
        else:
            shutil.copyfile(checkpoint_path, target_checkpoint)
        payload = {
            **metadata,
            "artifacts": {"checkpoint": "checkpoint.pkl"},
            "checkpoint_sha256": file_sha256(target_checkpoint),
        }
        if mapping is not None:
            mapping_path = staging / "item_to_sid.parquet"
            mapping.write_parquet(mapping_path)
            payload.update(
                mapping_rows=len(mapping),
                sid_width=int(mapping["sid"].list.len().first()),
                mapping_is_bijective=True,
                mapping_sha256=file_sha256(mapping_path),
            )
            payload["artifacts"]["mapping"] = mapping_path.name
        if quality_report is not None:
            report_path = staging / "sid_fit_report.json"
            report_path.write_text(json.dumps(quality_report, indent=2) + "\n")
            payload["quality_report"] = report_path.name
            payload["artifacts"]["quality_report"] = report_path.name
            payload["quality_report_sha256"] = file_sha256(report_path)
        (staging / "sid_mapper.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return payload


def _validate_extension_contract(
    path: Path,
    *,
    checkpoint_path: Path,
    mapping_path: Path,
) -> None:
    """Verify checkpoint and mapping hashes against the saved extension contract."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_checkpoint = payload.get("checkpoint_sha256")
    expected_mapping = payload.get("mapping_sha256")
    if not expected_checkpoint or not expected_mapping:
        raise ValueError("SID contract misses checkpoint/mapping hashes")
    if expected_checkpoint != file_sha256(checkpoint_path):
        raise ValueError("SID checkpoint hash differs from its persisted contract")
    if expected_mapping != file_sha256(mapping_path):
        raise ValueError(
            "existing SID mapping hash differs from its persisted contract"
        )


__all__ = [
    "SIDMethod",
    "apply_sid_mapper",
    "extend_sid_mapping",
    "fit_sid_mapper",
]
