"""Native RVQ fit/apply lifecycle and mathematical contracts on CPU."""

import json

import numpy as np
import polars as pl
import pytest
import torch

from item_representations.semantic_ids.cli import app
from item_representations.semantic_ids.mapping import (
    apply_sid_mapper,
    extend_sid_mapping,
    fit_sid_mapper,
)
from item_representations.semantic_ids.models import RVQ, BaseSIDSGen
from item_representations.semantic_ids.models.torch_kmeans import (
    pairwise_squared_distances,
)


def vectors():
    return np.random.default_rng(71).normal(size=(17, 5)).astype(np.float32)


def builder(**kwargs):
    return RVQ(codebook_size=(3, 3, 3), init_buffer_size=8, device="cpu", **kwargs)


@pytest.mark.parametrize(
    "reduction,scale", [("sum", 1.0), ("sum", 0.125), ("mean", 2.0)]
)
def test_configured_loss_and_gradient(reduction, scale):
    model = builder(loss_reduction=reduction, loss_scale=scale)
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    c = torch.tensor([[0.8, 0.0], [0.0, 0.7]], requires_grad=True)
    loss = model._loss(x, c)
    expected = scale * (
        (x - c).square().sum() if reduction == "sum" else (x - c).square().mean()
    )
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(
        torch.autograd.grad(loss, c)[0], torch.autograd.grad(expected, c)[0]
    )


def test_seed_partial_batches_freezing_and_refit():
    x = vectors()
    rng_before = torch.random.get_rng_state().clone()
    model = builder(loss_scale=0.125).fit(x, epochs_per_layer=2, batch_size=7)
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before)
    assert all(not c.requires_grad for c in model.codebooks)
    for row in model.fit_diagnostics:
        assert row["completed_epochs"] == 2
        assert row["optimizer_updates"] == 6
        assert row["examples_seen"] == 34
        assert row["initialization_rows"] == 8
    # Training later levels must leave the first codebook exactly unchanged.
    first_only = RVQ(
        n_levels=1, codebook_size=3, init_buffer_size=8, device="cpu", loss_scale=0.125
    ).fit(x, epochs_per_layer=2, batch_size=7)
    torch.testing.assert_close(
        model.codebooks[0], first_only.codebooks[0], rtol=0, atol=0
    )
    before = [c.clone() for c in model.codebooks]
    model.fit(x, epochs_per_layer=2, batch_size=7)
    for a, b in zip(before, model.codebooks, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("normalize", [True, False])
def test_transform_and_norm_assisted_reconstruction(normalize):
    x = vectors()
    model = builder(normalize_residuals=normalize).fit(
        x, epochs_per_layer=2, batch_size=7
    )
    residual = torch.tensor(x)
    norms, centers, codes = [], [], []
    for codebook in model.codebooks:
        norm = (
            residual.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            if normalize
            else torch.ones((len(x), 1))
        )
        residual = residual / norm
        index = ((residual[:, None] - codebook[None]) ** 2).sum(-1).argmin(-1)
        selected = codebook[index]
        norms.append(norm)
        centers.append(selected)
        codes.append(index)
        residual -= selected
    reconstruction = torch.zeros_like(residual)
    for norm, center in reversed(list(zip(norms, centers, strict=True))):
        reconstruction = norm * (center + reconstruction)
    np.testing.assert_array_equal(model.transform(x), torch.stack(codes, -1).numpy())
    np.testing.assert_allclose(
        model.reconstruct(x), reconstruction.numpy(), rtol=2e-5, atol=2e-6
    )
    if normalize:
        np.testing.assert_array_equal(model.transform(x), model.transform(x * 5))
    np.testing.assert_array_equal(
        model.transform(x), np.concatenate([model.transform(row[None]) for row in x])
    )
    assert model.transform(x[:0]).shape == (0, 3)


def write_catalog(root, name, ids, x):
    items, embeddings = root / f"{name}.pq", root / f"{name}.npy"
    pl.DataFrame({"item_id": ids}).write_parquet(items)
    np.save(embeddings, x)
    return items, embeddings


def run_cli(tokens):
    with pytest.raises(SystemExit) as result:
        app(tokens)
    assert result.value.code == 0


def test_cli_fit_load_apply_extend_and_metadata(tmp_path):
    x = vectors()
    ids = list(reversed(range(len(x))))
    items, embeddings = write_catalog(tmp_path, "train", ids, x)
    output = tmp_path / "fit"
    run_cli(
        [
            "fit-sid-mapper",
            "--items",
            str(items),
            "--embeddings",
            str(embeddings),
            "--output-dir",
            str(output),
            "--method",
            "rvq",
            "--device",
            "cpu",
            "--codebook-size",
            "3",
            "--codebook-size",
            "3",
            "--codebook-size",
            "3",
            "--epochs-per-layer",
            "2",
            "--training-batch-size",
            "7",
            "--init-buffer-size",
            "8",
            "--learning-rate",
            "0.001",
            "--loss-reduction",
            "sum",
            "--loss-scale",
            "0.125",
            "--eval-batch-size",
            "4",
        ]
    )
    meta = json.loads((output / "sid_mapper.json").read_text())
    report = json.loads((output / "sid_fit_report.json").read_text())
    assert meta["rvq"]["loss_scale"] == 0.125
    assert meta["rvq"]["epochs_per_layer"] == 2
    assert report["reconstruction"]["kind"] == "input_norm_assisted"
    assert report["reconstruction"]["sid_only"] is False
    assert len(report["quantization"]["per_level_mse"]) == 3
    loaded = BaseSIDSGen.load(output / "checkpoint.pkl")
    assert isinstance(loaded, RVQ)
    mapping = pl.read_parquet(output / "item_to_sid.parquet")
    assert mapping["item_id"].to_list() == sorted(ids)
    assert mapping["sid"].n_unique() == len(x)
    assert mapping["sid"].list.len().unique().to_list() == [4]
    applied = tmp_path / "applied"
    apply_sid_mapper(
        output / "checkpoint.pkl", items, embeddings, applied, eval_batch_size=3
    )
    assert pl.read_parquet(applied / "item_to_sid.parquet").equals(mapping)
    extended_items, extended_x = write_catalog(
        tmp_path, "extended", ids + [20], np.concatenate([x, x[:1]])
    )
    extended = tmp_path / "extended"
    result = extend_sid_mapping(
        output / "checkpoint.pkl",
        output / "item_to_sid.parquet",
        output / "sid_mapper.json",
        extended_items,
        extended_x,
        extended,
        eval_batch_size=5,
    )
    final = pl.read_parquet(extended / "item_to_sid.parquet")
    assert result["existing_rows_unchanged"] and result["appended_items"] == 1
    assert final["sid"].n_unique() == len(x) + 1
    assert final.filter(pl.col("item_id") != 20).equals(mapping)


def test_degenerate_catalog_and_mapper_only(tmp_path):
    x = np.zeros((5, 3), dtype=np.float32)
    items, embeddings = write_catalog(tmp_path, "zero", list(range(5)), x)
    output = tmp_path / "fit"
    fit_sid_mapper(
        items,
        embeddings,
        output,
        method="rvq",
        codebook_size=(3, 3),
        epochs_per_layer=1,
        training_batch_size=3,
        init_buffer_size=7,
        device="cpu",
        mapper_only=True,
    )
    model = BaseSIDSGen.load(output / "checkpoint.pkl")
    assert not (output / "item_to_sid.parquet").exists()
    assert all(r["initialization_rows"] == 5 for r in model.fit_diagnostics)
    np.testing.assert_array_equal(model.transform(x), np.zeros((5, 2)))
    np.testing.assert_array_equal(model.reconstruct(x), x)


@pytest.mark.parametrize(
    "options",
    [
        {"loss_scale": 0},
        {"loss_scale": float("nan")},
        {"loss_reduction": "none"},
        {"init_buffer_size": 2},
    ],
)
def test_invalid_configuration(options):
    with pytest.raises(ValueError):
        RVQ(codebook_size=3, **options)


def test_invalid_input_and_unfitted_model():
    model = builder()
    with pytest.raises(RuntimeError, match="fitted"):
        model.transform(vectors())
    with pytest.raises(ValueError, match="rows"):
        model.fit(vectors()[:2])
    bad = vectors()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.fit(bad)
    model.fit(vectors(), epochs_per_layer=1)
    with pytest.raises(ValueError, match="dimension"):
        model.transform(np.ones((4, 6)))


def test_distance_ties_and_near_ties_have_bounded_rounding_error():
    x = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    c = torch.tensor([[1.0, 0.0001], [1.0, 0.0], [1.0, 0.0]])
    efficient = pairwise_squared_distances(x, c)
    direct = (x[:, None].double() - c[None].double()).square().sum(-1)
    torch.testing.assert_close(efficient.double(), direct, atol=1e-7, rtol=1e-7)
    # Near-zero distances can round to ties; do not promise identical SID bits.
    ids = efficient.argmin(-1)
    assert torch.all(direct[torch.arange(len(x)), ids] <= direct.min(-1).values + 1e-7)
    assert pairwise_squared_distances(x[:1], c[1:]).argmin(-1).item() == 0


def test_checkpoint_cuda_tag_loads_on_cpu_without_cuda(tmp_path, monkeypatch):
    model = builder().fit(vectors(), epochs_per_layer=1)
    expected = model.transform(vectors())
    model.device = torch.device("cuda:0")
    path = tmp_path / "mapper.pkl"
    model.save(path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    restored = BaseSIDSGen.load(path)
    assert restored.device.type == "cpu"
    np.testing.assert_array_equal(restored.transform(vectors()), expected)
