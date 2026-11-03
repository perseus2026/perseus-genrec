"""Schema-configurable text encoding with an exact, copied train subset."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl

from genrecsys.utils.files import file_sha256, publish_directory
from item_representations.text.encoding import (
    ComputeDtype,
    EmbeddingDtype,
    encode_to_memmap,
    expand_embeddings,
    load_text_encoder,
    validate_normalized_embeddings,
)

logger = logging.getLogger(__name__)


def encode_catalog(
    *,
    catalog: Path,
    train_items: Path,
    model_path: Path,
    output_dir: Path,
    item_id_column: str = "item_id",
    text_column: str = "text",
    prompt: str = "",
    max_seq_length: int = 1024,
    output_dim: int = 2048,
    batch_size: int = 32,
    encode_chunk_size: int = 32768,
    compute_dtype: ComputeDtype = ComputeDtype.FLOAT16,
    output_dtype: EmbeddingDtype = EmbeddingDtype.FLOAT16,
    allow_cpu: bool = False,
) -> None:
    """Encode once from model weights; no pre-existing embedding store is required."""
    if min(max_seq_length, output_dim, batch_size, encode_chunk_size) < 1:
        raise ValueError("length, dimension and batch sizes must be positive")
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    items = pl.read_parquet(catalog)
    if "item_id" in items and item_id_column != "item_id":
        raise ValueError(
            "catalog already has an item_id column distinct from the configured ID"
        )
    items = items.with_columns(pl.col(item_id_column).alias("item_id")).sort("item_id")
    if (
        not len(items)
        or items["item_id"].null_count()
        or items["item_id"].n_unique() != len(items)
    ):
        raise ValueError("catalog must have unique nonempty item IDs")
    text = pl.col(text_column).cast(pl.String).str.strip_chars()
    items = items.with_columns(
        pl.when(text != "").then(text).otherwise(None).alias(text_column)
    ).with_row_index("embedding_row_index")
    train = (
        pl.read_parquet(train_items)
        .select(pl.col(item_id_column).alias("item_id"))
        .sort("item_id")
    )
    if (
        not len(train)
        or train["item_id"].null_count()
        or train["item_id"].n_unique() != len(train)
    ):
        raise ValueError("train catalog must have unique nonempty IDs")
    selected = train.join(
        items, on="item_id", how="left", maintain_order="left", validate="1:1"
    )
    if selected["embedding_row_index"].null_count():
        raise ValueError("train catalog is not covered by the inference catalog")
    with publish_directory(output_dir, overwrite=False) as output:
        logger.info("Fingerprinting local text encoder files: %s", model_path)
        model_hashes = {
            str(path.relative_to(model_path)): file_sha256(path)
            for path in sorted(model_path.rglob("*"))
            if path.is_file()
        }
        encoder, _, devices = load_text_encoder(
            model_path,
            compute_dtype=compute_dtype,
            max_seq_length=max_seq_length,
            output_dim=output_dim,
            allow_cpu=allow_cpu,
        )
        (output / "inference").mkdir()
        (output / "train").mkdir()
        pool = (
            encoder.start_multi_process_pool(target_devices=devices)
            if len(devices) > 1
            else None
        )
        try:
            vectors, stats = encode_to_memmap(
                encoder,
                items[text_column].fill_null(""),
                output / "inference/embeddings.npy",
                prompt=prompt,
                batch_size=batch_size,
                encode_chunk_size=encode_chunk_size,
                device=devices[0],
                output_dim=output_dim,
                embedding_dtype=output_dtype,
                oom_backoff=True,
                multi_process_pool=pool,
            )
        finally:
            if pool is not None:
                encoder.stop_multi_process_pool(pool)
        validate_normalized_embeddings(stats, "catalog embeddings")
        train_vectors, _ = expand_embeddings(
            vectors,
            selected["embedding_row_index"].to_numpy(),
            output / "train/embeddings.npy",
            embedding_dtype=output_dtype,
            chunk_size=4096,
        )
        train_vectors.flush()
        del vectors, train_vectors
        items.write_parquet(output / "inference/items.pq")
        selected.drop("embedding_row_index").with_row_index(
            "embedding_row_index"
        ).write_parquet(output / "train/items.pq")
        (output / "metadata.json").write_text(
            json.dumps(
                {
                    "catalog_sha256": file_sha256(catalog),
                    "train_items_sha256": file_sha256(train_items),
                    "model_path": str(model_path),
                    "model_files_sha256": model_hashes,
                    "prompt": prompt,
                    "max_seq_length": max_seq_length,
                    "output_dim": output_dim,
                    "compute_dtype": compute_dtype.value,
                    "output_dtype": output_dtype.value,
                    "catalog_items": len(items),
                    "train_items": len(train),
                    "statistics": stats.as_dict(),
                    "embedding_source": "fresh_model_encoding",
                    "train_vectors": "exact_subset_of_inference_matrix",
                },
                indent=2,
            )
            + "\n"
        )
        logger.info(
            "Encoded catalog=%s, train subset=%s; artifacts=%s",
            len(items),
            len(train),
            output_dir,
        )
