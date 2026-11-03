"""Compare one-process Accelerate fits with plain PyTorch optimizer/loader execution."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from item_representations.semantic_ids import mapping
from item_representations.semantic_ids.models import RQVAE, RVQ
from item_representations.semantic_ids.models.distributed import training_accelerator


def plain_training(device):
    """Run actual PyTorch operations without wrapping, sharding or reduction."""
    return SimpleNamespace(
        device=torch.device(device),
        num_processes=1,
        is_main_process=True,
        prepare=lambda *objects: objects,
        prepare_data_loader=lambda loader, **kwargs: loader,
        backward=torch.Tensor.backward,
        clip_grad_norm_=torch.nn.utils.clip_grad_norm_,
        reduce=lambda tensor, **kwargs: tensor,
    )


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("batch_size", [8, 32])
@pytest.mark.parametrize("recipe", ["rvq-sum", "rvq-mean", "rq-adam", "rq-adagrad-bn"])
def test_accelerate_matches_plain_fit(seed, batch_size, recipe):
    compare_fits(seed, batch_size, recipe, "cpu")


@pytest.mark.parametrize("seed", [42, 43])
@pytest.mark.parametrize("batch_size", [8, 32])
def test_rvq_loader_preserves_initialization_and_epoch_order(seed, batch_size):
    x = np.random.default_rng(71).normal(size=(19, 8)).astype("float32")
    model = RVQ(
        n_levels=2,
        codebook_size=(3, 3),
        init_buffer_size=8,
        random_state=seed,
        device="cpu",
    )
    with patch.object(model, "_tensor", wraps=model._tensor) as read_batch:
        model.fit(x, epochs_per_layer=3, batch_size=batch_size)
    # The original loop drew one permutation for initialization, then one per epoch.
    shuffle = torch.Generator().manual_seed(seed)
    expected = []
    for _ in range(2):
        expected.append(x[torch.randperm(len(x), generator=shuffle).numpy()[:8]])
        for _ in range(3):
            order = torch.randperm(len(x), generator=shuffle).numpy()
            expected.extend(
                x[order[start : start + batch_size]]
                for start in range(0, len(x), batch_size)
            )
    assert read_batch.call_count == len(expected)
    for call, batch in zip(read_batch.call_args_list, expected):
        np.testing.assert_array_equal(call.args[0], batch)


def compare_fits(seed, batch_size, recipe, device):
    x = np.random.default_rng(71).normal(size=(19, 8)).astype("float32")
    is_rvq = recipe.startswith("rvq")

    def fit(accelerator=None):
        if is_rvq:
            model = RVQ(
                n_levels=2,
                codebook_size=(3, 3),
                init_buffer_size=8,
                loss_reduction=recipe.split("-")[1],
                loss_scale=0.125,
                random_state=seed,
                device=device,
            )
            model.fit(
                x, epochs_per_layer=3, batch_size=batch_size, accelerator=accelerator
            )
        else:
            model = RQVAE(
                8,
                hidden_dims=(4, 2),
                codebook_size=(3, 3),
                random_state=seed,
                device=device,
                joint_normalization=recipe.endswith("bn"),
                optimizer_name="adagrad" if recipe.endswith("bn") else "adam",
                gradient_clip=None if recipe.endswith("bn") else 1.0,
            )
            model.fit(x, epochs=3, batch_size=batch_size, accelerator=accelerator)
        return model

    reference = fit(plain_training(device))
    accelerator = training_accelerator(device)
    with (
        patch.object(accelerator, "prepare", wraps=accelerator.prepare) as prepare,
        patch.object(accelerator, "backward", wraps=accelerator.backward) as backward,
        patch.object(
            accelerator, "prepare_data_loader", wraps=accelerator.prepare_data_loader
        ) as prepare_loader,
    ):
        actual = fit(accelerator)
    assert prepare.call_count == (2 if is_rvq else 1)
    assert prepare_loader.call_count == 1
    assert backward.call_count == sum(
        r["optimizer_updates"] for r in actual.fit_diagnostics
    )
    assert actual.config == reference.config
    assert actual.config["distributed"] == {
        "world_size": 1,
        "batch_size_per_device": batch_size,
        "nominal_global_batch_size": batch_size,
        "even_batches": True,
        "gradient_reduction": "mean",
        "precision": "float32",
    }
    tolerance = 0 if device == "cpu" else 2e-5
    for actual_row, reference_row in zip(
        actual.fit_diagnostics, reference.fit_diagnostics
    ):
        assert actual_row == pytest.approx(reference_row, rel=tolerance, abs=tolerance)
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(
            value, actual.state_dict()[key], rtol=tolerance, atol=tolerance
        )
    assert all(p.device == torch.device(device) for p in actual.parameters())
    np.testing.assert_array_equal(actual.transform(x), reference.transform(x))
    assert not torch.distributed.is_initialized()
    if recipe.endswith("bn"):
        assert type(actual.encoder.normalization[0]) is torch.nn.BatchNorm1d


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_single_cuda_matches_plain_fit():
    # Accelerate caches its selected device: isolate this from the CPU test process.
    env = {k: v for k, v in os.environ.items() if not k.startswith("ACCELERATE_")}
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-7000:]


@pytest.mark.parametrize("method", ["random_hashing", "rq_kmeans", "rq_kmeans_full"])
def test_non_gradient_mappers_do_not_initialize_accelerator(
    tmp_path, monkeypatch, method
):
    import polars as pl

    def unexpected(*args, **kwargs):
        raise AssertionError("KMeans/hash fitting must not initialize Accelerate")

    monkeypatch.setattr(mapping, "training_accelerator", unexpected)
    np.save(
        tmp_path / "x.npy",
        np.random.default_rng(7).normal(size=(12, 4)).astype("float32"),
    )
    pl.DataFrame({"item_id": [f"{i:02}" for i in range(12)]}).write_parquet(
        tmp_path / "items.pq"
    )
    metadata = mapping.fit_sid_mapper(
        tmp_path / "items.pq",
        tmp_path / "x.npy",
        tmp_path / "fit",
        method=method,
        codebook_size=(3, 3),
        device="cpu",
        max_iter=2,
        n_init=1,
    )
    assert metadata["mapping_rows"] == 12


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES / the launcher selects the physical GPU.
    device = "cuda:0"
    for seed in (42, 43):
        for batch_size in (8, 32):
            for recipe in ("rvq-sum", "rvq-mean", "rq-adam", "rq-adagrad-bn"):
                compare_fits(seed, batch_size, recipe, device)
