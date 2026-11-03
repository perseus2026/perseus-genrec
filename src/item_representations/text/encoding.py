"""Bounded, row-aligned text encoding shared by dataset adapters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

logger = logging.getLogger(__name__)


class ComputeDtype(str, Enum):
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"


class EmbeddingDtype(str, Enum):
    FLOAT16 = "float16"
    FLOAT32 = "float32"


@dataclass
class EmbeddingStatistics:
    rows: int = 0
    non_finite_values: int = 0
    norm_min: float = float("inf")
    norm_max: float = float("-inf")
    norm_sum: float = 0.0

    def update(self, embeddings: np.ndarray) -> None:
        """Accumulate row counts, nonfinite values, and L2 norm statistics for one batch."""

        values = np.asarray(embeddings, dtype=np.float32)
        self.rows += len(values)
        self.non_finite_values += int(values.size - np.isfinite(values).sum())
        if len(values):
            norms = np.linalg.norm(values, axis=1)
            self.norm_min = min(self.norm_min, float(norms.min()))
            self.norm_max = max(self.norm_max, float(norms.max()))
            self.norm_sum += float(norms.sum(dtype=np.float64))

    def as_dict(self) -> dict[str, Any]:
        """Return accumulated embedding statistics as a serializable report."""

        return {
            "rows": self.rows,
            "non_finite_values": self.non_finite_values,
            "norm_min": self.norm_min if self.rows else None,
            "norm_mean": self.norm_sum / self.rows if self.rows else None,
            "norm_max": self.norm_max if self.rows else None,
        }


def encode_to_memmap(
    model: Any,
    texts: pl.Series,
    output_path: Path,
    *,
    prompt: str,
    batch_size: int,
    encode_chunk_size: int,
    device: str,
    output_dim: int,
    embedding_dtype: EmbeddingDtype,
    multi_process_pool: dict[str, Any] | None = None,
    oom_backoff: bool = False,
) -> tuple[np.memmap, EmbeddingStatistics]:
    """Encode bounded chunks while preserving SentenceTransformer row order."""
    total_rows = len(texts)
    if not total_rows:
        raise ValueError("There are no texts to encode")
    stats = EmbeddingStatistics()
    numpy_dtype = np.dtype(embedding_dtype.value)
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=numpy_dtype, shape=(total_rows, output_dim)
    )

    for start in range(0, total_rows, encode_chunk_size):
        values = texts.slice(start, encode_chunk_size).to_list()
        encode_kwargs: dict[str, Any] = {
            "prompt": prompt or None,
            "batch_size": batch_size,
            "show_progress_bar": total_rows <= encode_chunk_size,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
            "truncate_dim": output_dim,
        }
        sorted_indices: np.ndarray | None = None
        texts_to_encode = values
        if multi_process_pool is None:
            encode_kwargs["device"] = device
        else:
            # SentenceTransformer sorts inputs before single-device encoding.
            # Repeat that ordering before multi-process dispatch so worker batch
            # composition is identical, then scatter results back to input order.
            sorted_indices = np.argsort(
                [-model._input_length(text) for text in values]  # type: ignore[attr-defined]
            )
            if model._can_flatten_inputs():  # type: ignore[attr-defined]
                sorted_indices = model._interleave_sorted_indices(  # type: ignore[attr-defined]
                    sorted_indices
                )
            texts_to_encode = [values[index] for index in sorted_indices]
            encode_kwargs["pool"] = multi_process_pool
            encode_kwargs["chunk_size"] = batch_size

        while True:
            try:
                chunk = np.asarray(model.encode(texts_to_encode, **encode_kwargs))
                break
            except torch.cuda.OutOfMemoryError:
                if not oom_backoff or multi_process_pool is not None or batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                encode_kwargs["batch_size"] = batch_size
                logger.warning(
                    "CUDA OOM: retrying text chunk with batch_size=%s", batch_size
                )
                torch.cuda.empty_cache()
        if chunk.shape != (len(values), output_dim):
            raise RuntimeError(
                f"Expected ({len(values)}, {output_dim}) embeddings, got {chunk.shape}"
            )
        chunk = chunk.astype(numpy_dtype, copy=False)
        if sorted_indices is not None:
            restored = np.empty_like(chunk)
            restored[sorted_indices] = chunk
            chunk = restored
        if not np.isfinite(chunk).all():
            raise RuntimeError("Model returned non-finite embedding values")

        end = start + len(chunk)
        output[start:end] = chunk
        output.flush()
        stats.update(chunk)
        logger.info("Encoded %d/%d unique texts", end, total_rows)
        del values, texts_to_encode, chunk

    return output, stats


def expand_embeddings(
    source: np.ndarray,
    indices: np.ndarray,
    output_path: Path,
    *,
    embedding_dtype: EmbeddingDtype,
    chunk_size: int,
) -> tuple[np.memmap, EmbeddingStatistics]:
    """Expand unique-text embeddings into row-aligned item embeddings in an NPY mmap.

    Write bounded chunks and accumulate finite-value/norm statistics. Multiple items
    may share one source vector.
    """

    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.dtype(embedding_dtype.value),
        shape=(len(indices), source.shape[1]),
    )
    stats = EmbeddingStatistics()
    for start in range(0, len(indices), chunk_size):
        end = min(start + chunk_size, len(indices))
        chunk = np.asarray(source[indices[start:end]])
        if not np.isfinite(chunk).all():
            raise RuntimeError("Non-finite values found while expanding embeddings")
        output[start:end] = chunk
        output.flush()
        stats.update(chunk)
        logger.info("Materialized %d/%d catalog embeddings", end, len(indices))
    return output, stats


def validate_normalized_embeddings(
    stats: EmbeddingStatistics,
    artifact_name: str,
    *,
    tolerance: float = 0.02,
) -> None:
    """Reject nonfinite vectors or L2 norms outside the unit-norm tolerance."""

    if stats.non_finite_values:
        raise RuntimeError(
            f"{artifact_name} contains {stats.non_finite_values} non-finite values"
        )
    lower = 1.0 - tolerance
    upper = 1.0 + tolerance
    if stats.rows and (stats.norm_min < lower or stats.norm_max > upper):
        raise RuntimeError(
            f"{artifact_name} is not L2-normalized: norm range "
            f"[{stats.norm_min:.6f}, {stats.norm_max:.6f}], expected "
            f"[{lower:.2f}, {upper:.2f}]"
        )


def load_text_encoder(
    model_path: str | Path,
    *,
    compute_dtype: ComputeDtype,
    output_dim: int,
    max_seq_length: int,
    allow_cpu: bool = False,
) -> tuple[Any, Any, list[str]]:
    """Shared local-only SentenceTransformer loader, independent of catalog schema."""
    from sentence_transformers import SentenceTransformer

    devices = (
        [f"cuda:{index}" for index in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else ["cpu"]
    )
    if devices == ["cpu"] and not allow_cpu:
        raise RuntimeError(
            "No CUDA device found; pass --allow-cpu only for smoke tests"
        )
    dtype = {
        ComputeDtype.FLOAT16: torch.float16,
        ComputeDtype.BFLOAT16: torch.bfloat16,
        ComputeDtype.FLOAT32: torch.float32,
    }[compute_dtype]
    init_kwargs: dict[str, Any] = {}
    if len(devices) > 1:
        # Worker processes put one shared model copy on each GPU.  Keeping the
        # parent copy on CPU avoids wasting another full Qwen allocation on GPU 0.
        init_kwargs["device"] = "cpu"
    model = SentenceTransformer(
        str(model_path),
        local_files_only=True,
        model_kwargs={"dtype": dtype},
        truncate_dim=output_dim,
        **init_kwargs,
    )
    model.max_seq_length = max_seq_length
    return model, torch, devices
