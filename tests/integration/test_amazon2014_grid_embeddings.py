"""CPU artifact checks with synthetic GRID outputs; no LLM inference or download."""

import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
import yaml
from test_amazon2014_grid_import import _source

from model_datasets.amazon2014_grid.basis import build_basis
from model_datasets.amazon2014_grid.grid_embeddings import export_grid_embeddings
from model_datasets.amazon2014_grid.prepare import import_beauty


def _inputs(root: Path):
    raw = _source(root / "raw")
    source = root / "source"
    import_beauty(source_dir=raw, output_dir=source)
    stage01 = root / "stage01"
    grid_output = stage01 / "grid"
    (grid_output / "pickle").mkdir(parents=True)
    # Non-unit norms expose accidental normalization; IDs are deliberately shuffled.
    vectors = torch.arange(27 * 2048, dtype=torch.float32).reshape(27, 2048)
    rows = [
        {"item_id": item, "embedding": vectors[item].tolist()}
        for item in reversed(range(27))
    ]
    torch.save(vectors, grid_output / "pickle/merged_predictions_tensor.pt")
    with (grid_output / "pickle/merged_predictions.pkl").open("wb") as stream:
        pickle.dump(rows, stream)
    (grid_output / ".hydra").mkdir()
    (grid_output / ".hydra/config.yaml").write_text(
        yaml.safe_dump({"embedding_model": "/worker/cache/snapshots/test-snapshot"})
    )
    (stage01 / "runtime").mkdir()
    (stage01 / "runtime/grid-requirements.txt").write_text("torch==2.6.0+cu124\n")
    (stage01 / "runtime/grid-packages.txt").write_text("torch==2.6.0+cu124\n")
    return (
        {
            "source_dir": source,
            "stage01_dir": stage01,
            "output_dir": root / "representations",
        },
        vectors,
        rows,
    )


def test_export_preserves_vectors_ids_texts_and_full_fit_scope(tmp_path: Path):
    args, vectors, _ = _inputs(tmp_path)
    export_grid_embeddings(**args)
    output = args["output_dir"]
    for scope in ("train", "inference"):
        exported = np.load(output / scope / "embeddings.npy")
        assert exported.dtype == np.float32
        np.testing.assert_array_equal(exported, vectors.numpy())
        items = pl.read_parquet(output / scope / "items.pq")
        assert items["embedding_row_index"].to_list() == list(range(27))
        assert items["item_id"].to_list() == [str(i) for i in range(27)]
        assert items["text"][0].endswith("; ")
        assert {"25", "26"} <= set(items["item_id"])
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["normalize_embeddings"] is False
    assert metadata["archive_embeddings_reused"] is False
    assert metadata["framework_dataset_ready"] is False
    assert metadata["statistics"]["norm_min"] > 1
    assert metadata["stage"] == "02_grid_embedding_export"
    assert metadata["stage01_files_sha256"]["runtime/grid-requirements.txt"]
    assert metadata["model_snapshot"] == "test-snapshot"
    assert metadata["model_files_sha256"] is None
    with pytest.raises(FileExistsError):
        export_grid_embeddings(**args)


@pytest.mark.parametrize(
    "problem,error",
    [
        ("missing_id", "cover the catalog"),
        ("duplicate_id", "cover the catalog"),
        ("swapped_vectors", "embedding/ID mismatch"),
        ("nonfinite", "finite FP32"),
        ("dtype", "finite FP32"),
        ("shape", "2048"),
    ],
)
def test_export_rejects_incomplete_or_inconsistent_grid_result(
    tmp_path: Path, problem: str, error: str
):
    args, vectors, rows = _inputs(tmp_path)
    if problem == "missing_id":
        rows.pop()
    elif problem == "duplicate_id":
        rows[0]["item_id"] = rows[1]["item_id"]
    elif problem == "swapped_vectors":
        vectors = vectors.flip(0)
    elif problem == "nonfinite":
        vectors[0, 0] = float("nan")
    elif problem == "dtype":
        vectors = vectors.half()
    elif problem == "shape":
        vectors = vectors[:, :768]
    root = args["stage01_dir"] / "grid/pickle"
    torch.save(vectors, root / "merged_predictions_tensor.pt")
    with (root / "merged_predictions.pkl").open("wb") as stream:
        pickle.dump(rows, stream)
    with pytest.raises(ValueError, match=error):
        export_grid_embeddings(**args)
    assert not args["output_dir"].exists()


@pytest.mark.parametrize("invalid_vectors", [False, True])
def test_stage02_raw_basis_and_embedding_export_are_published_together(
    tmp_path: Path, invalid_vectors: bool
):
    args, vectors, _ = _inputs(tmp_path)
    output = tmp_path / "basis-output"
    if invalid_vectors:
        vectors[0, 0] = float("nan")
        torch.save(
            vectors, args["stage01_dir"] / "grid/pickle/merged_predictions_tensor.pt"
        )
    kwargs = {
        "source_dir": tmp_path / "raw",
        "stage01_dir": args["stage01_dir"],
        "output_dir": output,
        "augmentation": "none",
        "max_history": 20,
        "pack_eventhub": True,
    }
    if invalid_vectors:
        with pytest.raises(ValueError, match="finite FP32"):
            build_basis(**kwargs)
        assert not output.exists()
        return
    build_basis(**kwargs)
    np.testing.assert_array_equal(
        np.load(output / "representations/inference/embeddings.npy"), vectors.numpy()
    )
    assert (output / "basis/train/samples.pq").is_file()
    assert (output / "source/report.json").is_file()
    assert (output / "eventhub.tar").is_file()
    report = json.loads((output / "report.json").read_text())
    assert report["representations"] == {
        "dense_embeddings": True,
        "semantic_ids": False,
    }
    assert report["framework_dataset_ready"] is False
