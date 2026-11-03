"""Fit, apply and append-only SID mapper lifecycle test."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from item_representations.semantic_ids.mapping import (
    _load_aligned,
    apply_sid_mapper,
    extend_sid_mapping,
    fit_sid_mapper,
)
from item_representations.semantic_ids.models.rq_kmeans import RQKMeans
from item_representations.semantic_ids.models.rq_kmeans_full import RQKMeansFull


def _write_catalog(
    root: Path,
    name: str,
    item_ids: list[int],
    embeddings: np.ndarray,
) -> tuple[Path, Path]:
    items_path = root / f"{name}-items.pq"
    embeddings_path = root / f"{name}-embeddings.npy"
    pl.DataFrame({"item_id": item_ids}).write_parquet(items_path)
    np.save(embeddings_path, embeddings.astype(np.float32))
    return items_path, embeddings_path


def test_aligned_catalog_keeps_embedding_mmap_for_sorted_stage05_contract(
    tmp_path: Path,
) -> None:
    items_path, embeddings_path = _write_catalog(
        tmp_path,
        "mmap",
        [10, 20, 30],
        np.asarray([[1.0], [2.0], [3.0]], dtype=np.float16),
    )

    items, embeddings = _load_aligned(items_path, embeddings_path)

    assert items["item_id"].to_list() == [10, 20, 30]
    assert isinstance(embeddings, np.memmap)
    assert embeddings[:, 0].tolist() == [1.0, 2.0, 3.0]


def test_rq_kmeans_fit_reads_only_bounded_slices() -> None:
    class SliceOnlyEmbeddings:
        ndim = 2

        def __init__(self, values: np.ndarray, max_rows: int) -> None:
            self.values = values
            self.max_rows = max_rows
            self.maximum_read = 0

        def __len__(self) -> int:
            return len(self.values)

        def __getitem__(self, rows: slice) -> np.ndarray:
            assert isinstance(rows, slice)
            selected = self.values[rows]
            self.maximum_read = max(self.maximum_read, len(selected))
            assert len(selected) <= self.max_rows
            return selected

    embeddings = SliceOnlyEmbeddings(
        np.random.default_rng(42).normal(size=(24, 4)).astype(np.float32),
        max_rows=6,
    )
    model = RQKMeans(
        n_levels=2,
        codebook_size=(2, 2),
        random_state=42,
        batch_size=6,
        max_iter=2,
        n_init=1,
        device="cpu",
    )

    model.fit(embeddings)

    assert embeddings.maximum_read == 6


def test_full_kmeans_memory_guard_precedes_gpu_fit(tmp_path, monkeypatch):
    items, embeddings = _write_catalog(
        tmp_path, "guard", [1, 2], np.eye(2, dtype=np.float32)
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1, 1))

    def unexpected_fit(*args, **kwargs):
        raise AssertionError("must reject before attempting the GPU allocation")

    monkeypatch.setattr(RQKMeansFull, "fit", unexpected_fit)
    with pytest.raises(MemoryError, match="GPU budget"):
        fit_sid_mapper(
            items,
            embeddings,
            tmp_path / "fit",
            device="cuda",
            codebook_size=(2,),
            max_gpu_memory_fraction=0.8,
        )
    assert not (tmp_path / "fit").exists()


def test_minibatch_initialization_handles_shuffled_short_tail():
    model = RQKMeans(
        n_levels=1,
        codebook_size=2,
        batch_size=2,
        random_state=42,
        max_iter=2,
        n_init=1,
        device="cpu",
    )
    model.fit(np.array([[0, 1], [1, 0], [1, 1]], dtype=np.float32))
    assert model.codebooks[0].shape == (2, 2)


def test_mapper_only_does_not_apply_codes(tmp_path, monkeypatch):
    from item_representations.semantic_ids import mapping

    items, embeddings = _write_catalog(tmp_path, "fit", [1, 2, 3], np.eye(3))

    def unexpected_transform(*args, **kwargs):
        raise AssertionError("mapper-only must defer SID application to Stage 08")

    monkeypatch.setattr(mapping, "_transform", unexpected_transform)
    fit_sid_mapper(
        items,
        embeddings,
        tmp_path / "mapper",
        method="rq_kmeans",
        codebook_size=(2,),
        fit_batch_size=2,
        max_iter=1,
        n_init=1,
        device="cpu",
        mapper_only=True,
    )
    assert (tmp_path / "mapper/checkpoint.pkl").is_file()
    assert not (tmp_path / "mapper/item_to_sid.parquet").exists()
    report = json.loads((tmp_path / "mapper/sid_fit_report.json").read_text())
    metadata = json.loads((tmp_path / "mapper/sid_mapper.json").read_text())
    assert report["items"] == 3
    assert report["levels"][0]["active_codes"] == 2
    assert report["reconstruction"]["rmse"] >= 0
    assert metadata["quality_report"] == "sid_fit_report.json"


def test_sid_quality_report_uses_bounded_batches_without_detailed_distributions():
    from item_representations.semantic_ids.mapping import _fit_quality_report

    batch_sizes = []

    class Mapper:
        method = "fake"
        n_levels = 1
        codebook_size = (2,)

        def transform(self, batch):
            batch_sizes.append(len(batch))
            return (batch[:, 0] < 0).astype(np.int64).reshape(-1, 1)

        def reconstruct(self, batch):
            assert len(batch) <= 2
            return batch.copy()

    embeddings = np.array([[1, 0], [-1, 0], [1, 0]], dtype=np.float32)
    codes, report = _fit_quality_report(Mapper(), embeddings, batch_size=2)
    assert batch_sizes == [2, 1]
    np.testing.assert_array_equal(codes[:, 0], [0, 1, 0])
    assert report["version"] == 2
    assert report["reconstruction"] == {
        "rmse": 0.0,
        "normalized_rmse": 0.0,
        "cosine_mean": 1.0,
    }
    assert report["levels"] == [
        {"level": 1, "codebook_size": 2, "active_codes": 2, "max_code_share": 2 / 3}
    ]


def test_sid_mapper_fit_apply_and_stable_extension(tmp_path: Path) -> None:
    initial_items, initial_embeddings = _write_catalog(
        tmp_path,
        "initial",
        [30, 10, 20, 40],
        np.asarray(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.9, 0.1],
                [0.1, 0.9],
            ]
        ),
    )
    fitted = tmp_path / "fitted"
    fit_sid_mapper(
        initial_items,
        initial_embeddings,
        fitted,
        codebook_size=(2, 2),
    )

    fitted_mapping = pl.read_parquet(fitted / "item_to_sid.parquet")
    metadata = json.loads((fitted / "sid_mapper.json").read_text())
    assert metadata["quality_report"] == "sid_fit_report.json"
    assert (fitted / metadata["quality_report"]).is_file()
    assert fitted_mapping.columns == ["item_id", "sid"]
    assert fitted_mapping["item_id"].to_list() == [10, 20, 30, 40]
    assert fitted_mapping["sid"].n_unique() == len(fitted_mapping)

    applied = tmp_path / "applied"
    apply_sid_mapper(
        fitted / "checkpoint.pkl",
        initial_items,
        initial_embeddings,
        applied,
    )
    assert pl.read_parquet(applied / "item_to_sid.parquet").equals(fitted_mapping)

    extended_items, extended_embeddings = _write_catalog(
        tmp_path,
        "extended",
        [50, 30, 10, 20, 40],
        np.asarray(
            [
                [0.5, 0.5],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.9, 0.1],
                [0.1, 0.9],
            ]
        ),
    )
    extended = tmp_path / "extended-output"
    report = extend_sid_mapping(
        fitted / "checkpoint.pkl",
        fitted / "item_to_sid.parquet",
        fitted / "sid_mapper.json",
        extended_items,
        extended_embeddings,
        extended,
    )
    extended_mapping = pl.read_parquet(extended / "item_to_sid.parquet")
    old_after_extension = extended_mapping.join(
        fitted_mapping,
        on="item_id",
        how="inner",
        suffix="_before",
    )

    assert old_after_extension.filter(pl.col("sid") != pl.col("sid_before")).is_empty()
    assert extended_mapping["sid"].n_unique() == len(extended_mapping)
    assert report["existing_rows_unchanged"] is True
    assert report["appended_items"] == 1
    assert report["full_catalog_audit"] == {"enabled": False, "checked": 0}


@pytest.mark.parametrize("method", ["random_hashing", "rq_kmeans", "rq_vae"])
def test_experimental_sid_builders_fit_and_apply_stably(
    tmp_path: Path,
    method: str,
) -> None:
    items, embeddings = _write_catalog(
        tmp_path,
        method,
        [10, 20, 30, 40, 50, 60],
        np.asarray(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
                [0.7, 0.3],
                [0.3, 0.7],
            ]
        ),
    )
    fitted = tmp_path / f"{method}-fitted"
    report = fit_sid_mapper(
        items,
        embeddings,
        fitted,
        method=method,
        codebook_size=(2, 2),
        device="cpu",
        fit_batch_size=6,
        max_iter=2,
        n_init=1,
        hidden_dim_factors=(1.0,),
        codebook_kmeans_init=False,
        epochs=1,
        training_batch_size=3,
    )
    assert report["method"] == method

    applied = tmp_path / f"{method}-applied"
    apply_sid_mapper(
        fitted / "checkpoint.pkl",
        items,
        embeddings,
        applied,
    )
    assert pl.read_parquet(applied / "item_to_sid.parquet").equals(
        pl.read_parquet(fitted / "item_to_sid.parquet")
    )
