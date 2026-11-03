"""Validate and export embeddings computed by GRID; never run a text encoder."""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml

from genrecsys.utils.files import file_sha256, publish_directory
from item_representations.text.encoding import EmbeddingStatistics
from model_datasets.amazon2014_grid.prepare import source_protocol


def export_grid_embeddings(
    *,
    source_dir: Path,
    stage01_dir: Path,
    output_dir: Path,
) -> None:
    """Check author pickle/PT outputs and publish full-catalog FP32 representations.

    source_dir is the validated Stage 02 import. GRID output files are
    trusted artifacts of this task. Row i must denote original GRID item ID i.
    No re-encoding, text edits, normalization or dimension reduction is applied.
    """
    source_report = json.loads((source_dir / "report.json").read_text())
    category = source_report["category"]
    protocol = source_protocol(category)
    if source_report["protocol"] != protocol:
        raise ValueError("Expected a validated GRID Amazon source")
    input_report_path = stage01_dir / "runtime/source-inputs.json"
    if category != "beauty" or input_report_path.exists():
        inputs = json.loads(input_report_path.read_text())
        if inputs.get("category") != category or inputs.get(
            "source_report_sha256"
        ) != file_sha256(source_dir / "report.json"):
            raise ValueError("GRID embeddings category/source provenance mismatch")
    catalog_path = source_dir / "catalog/items.pq"
    catalog = pl.read_parquet(catalog_path).sort("grid_item_id")
    count = len(catalog)
    if (
        count != source_report["items"]
        or catalog["grid_item_id"].to_list() != list(range(count))
        or catalog["item_id"].to_list() != [str(i) for i in range(count)]
    ):
        raise ValueError("Expected the full contiguous GRID item catalog")
    grid_output_dir = stage01_dir / "grid"
    tensor_path = grid_output_dir / "pickle/merged_predictions_tensor.pt"
    rows_path = grid_output_dir / "pickle/merged_predictions.pkl"
    embeddings = torch.load(tensor_path, map_location="cpu", weights_only=True)
    if not isinstance(embeddings, torch.Tensor) or embeddings.shape != (count, 2048):
        raise ValueError(f"Expected a [{count}, 2048] GRID embedding tensor")
    if embeddings.dtype != torch.float32 or not torch.isfinite(embeddings).all():
        raise ValueError("Expected finite FP32 GRID embeddings")
    with rows_path.open("rb") as stream:
        rows = pickle.load(stream)
    ids = [row["item_id"] for row in rows]
    if len(ids) != count or set(ids) != set(range(count)):
        raise ValueError("GRID prediction IDs must cover the catalog exactly once")
    for row in rows:
        vector = torch.as_tensor(row["embedding"], dtype=torch.float32)
        if not torch.equal(vector, embeddings[row["item_id"]]):
            raise ValueError(f"GRID embedding/ID mismatch: {row['item_id']}")
    hydra_dir = grid_output_dir / ".hydra"
    config = yaml.safe_load((hydra_dir / "config.yaml").read_text())
    model_path = config["embedding_model"]
    provenance_files = [
        *sorted(hydra_dir.glob("*.yaml")),
        stage01_dir / "runtime/grid-requirements.txt",
        stage01_dir / "runtime/grid-packages.txt",
    ]
    provenance_hashes = {
        str(path.relative_to(stage01_dir)): file_sha256(path)
        for path in provenance_files
    }

    statistics = EmbeddingStatistics()
    statistics.update(embeddings.numpy())
    with publish_directory(output_dir, overwrite=False) as output:
        items = catalog.with_row_index("embedding_row_index")
        inference = output / "inference"
        inference.mkdir()
        items.write_parquet(inference / "items.pq")
        np.save(inference / "embeddings.npy", embeddings.numpy())
        # Author quantizer fitting uses the full catalog, including holdout-only items.
        shutil.copytree(inference, output / "train")
        metadata = {
            "stage": "02_grid_embedding_export",
            "framework_dataset_ready": False,
            "source_protocol": protocol,
            "category": category,
            "sid_fit_diagnostics": source_report.get("sid_fit_diagnostics"),
            "source_report_sha256": file_sha256(source_dir / "report.json"),
            "catalog_sha256": file_sha256(catalog_path),
            "embedding_source": "GRID src.inference experiment=sem_embeds_inference_flat",
            "archive_embeddings_reused": False,
            "model": "google/flan-t5-xl",
            "model_path": model_path,
            "model_snapshot": Path(model_path).name,
            "model_files_sha256": None,
            "stage01_files_sha256": provenance_hashes,
            "grid_predictions_sha256": file_sha256(rows_path),
            "grid_tensor_sha256": file_sha256(tensor_path),
            "prompt": "",
            "pooling": "mean_nonpadding_including_special_tokens",
            "normalize_embeddings": False,
            "max_seq_length": 128,
            "output_dim": 2048,
            "compute_dtype": "float32",
            "output_dtype": "float32",
            "catalog_items": count,
            "train_items": count,
            "quantizer_fit_scope": "full author catalog",
            "row_mapping": "embedding_row_index equals original numeric GRID item ID",
            "train_vectors": "exact_copy_of_full_inference_matrix",
            "statistics": statistics.as_dict(),
        }
        (output / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
        )
