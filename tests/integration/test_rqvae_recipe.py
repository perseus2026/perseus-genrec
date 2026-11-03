"""RQ-VAE default recipe, reproducible fitting and normalized recipe."""

import numpy as np
import polars as pl
import pytest
import torch
from torch import nn

from item_representations.semantic_ids.cli import app
from item_representations.semantic_ids.models import RQVAE, BaseSIDSGen
from item_representations.semantic_ids.models.rq_vae import MLP, Quantizer


def make(seed=43, normalized=False):
    return RQVAE(
        8,
        hidden_dims=(6, 4),
        codebook_size=(3, 3, 3),
        device="cpu",
        random_state=seed,
        latent_l2=not normalized,
        mlp_bias=normalized,
        joint_normalization=normalized,
        optimizer_name="adagrad" if normalized else "adam",
        gradient_clip=None if normalized else 1.0,
    )


@pytest.mark.parametrize("normalized", [False, True])
def test_initialization_and_shuffle_are_seeded_without_ambient_rng(normalized):
    x = np.random.default_rng(31).normal(size=(24, 8)).astype("float32")
    states = []
    with torch.random.fork_rng(devices=[]):
        for ambient in (10, 20):
            torch.manual_seed(ambient)
            before = torch.get_rng_state().clone()
            model = make(normalized=normalized)
            torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
            torch.rand(ambient)
            before = torch.get_rng_state().clone()
            model.fit(x, epochs=2, batch_size=16)
            torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
            states.append(model.state_dict())
    for name in states[0]:
        torch.testing.assert_close(states[0][name], states[1][name], rtol=0, atol=0)
    assert not torch.equal(
        make(seed=42).encoder.mlp[0].weight, make(seed=43).encoder.mlp[0].weight
    )


def test_defaults_match_pre_port_architecture_and_initialization():
    # Before the seed fix these exact modules consumed the ambient CPU RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(43)
        encoder = MLP([8, 6, 4], normalize=True)
        codebooks = nn.ModuleList(
            Quantizer(4, 3, normalize=False, beta=0.25, distance="l2") for _ in range(3)
        )
        decoder = MLP([4, 6, 8])
    model = make()
    for actual, expected in [
        (model.encoder, encoder),
        (model.codebooks, codebooks),
        (model.decoder, decoder),
    ]:
        for k, v in expected.state_dict().items():
            torch.testing.assert_close(actual.state_dict()[k], v, rtol=0, atol=0)
    assert not model.joint_normalization and not model.mlp_bias and model.latent_l2
    assert model.optimizer_name == "adam" and model.gradient_clip == 1.0


def test_normalization_shared_target_bn_once_and_inference_independent_of_batch():
    model = make(normalized=True)
    x = torch.randn(24, 8)
    bn = model.encoder.normalization[0]
    losses = model(x, return_loss=True)
    assert bn.num_batches_tracked.item() == 1
    sum(losses).backward()
    assert (
        bn.weight.grad is not None
        and model.encoder.backbone.mlp[0].bias.grad is not None
    )
    model.eval()
    with torch.no_grad():
        target = model.encoder.normalization(x)
        _, reconstruction = model(x)
        _, loss = model(x, return_loss=True)
        torch.testing.assert_close(loss, (reconstruction - target).square().mean())
    np.testing.assert_array_equal(
        model.transform(x.numpy()),
        np.concatenate([model.transform(row[None].numpy()) for row in x]),
    )


def test_normalized_cli_and_reload(tmp_path):
    x = np.random.default_rng(2).normal(size=(24, 8)).astype("float32")
    np.save(tmp_path / "x.npy", x)
    pl.DataFrame({"item_id": [str(i) for i in range(24)]}).write_parquet(
        tmp_path / "items.pq"
    )
    out = tmp_path / "fit"
    with pytest.raises(SystemExit) as result:
        app(
            [
                "fit-sid-mapper",
                f"--items={tmp_path / 'items.pq'}",
                f"--embeddings={tmp_path / 'x.npy'}",
                f"--output-dir={out}",
                "--method=rq_vae",
                "--device=cpu",
                "--codebook-size=3",
                "--codebook-size=3",
                "--codebook-size=3",
                "--hidden-dim-factors=2",
                "--hidden-dim-factors=4",
                "--epochs=2",
                "--training-batch-size=16",
                "--random-state=43",
                "--no-latent-l2",
                "--mlp-bias",
                "--joint-normalization",
                "--optimizer-name=adagrad",
                "--gradient-clip=0",
            ]
        )
    assert result.value.code == 0
    model = BaseSIDSGen.load(out / "checkpoint.pkl")
    assert model.optimizer_name == "adagrad" and model.gradient_clip is None
    assert model.joint_normalization and not model.latent_l2
    assert model.fit_diagnostics[0]["optimizer_updates"] == 4
    np.testing.assert_array_equal(
        model.transform(x),
        np.concatenate([model.transform(x[:10]), model.transform(x[10:])]),
    )


@pytest.mark.parametrize("count,batch_size", [(17, 16), (8, 1)])
def test_bn_singleton_tail_rejected_without_dropping_items(count, batch_size):
    with pytest.raises(ValueError, match="Expected more than 1 value per channel"):
        make(normalized=True).fit(
            np.random.default_rng(4).normal(size=(count, 8)).astype("float32"),
            epochs=1,
            batch_size=batch_size,
            kmeans_init_batch_size=32,
        )


def test_initialization_batch_is_independent_from_training_batch():
    x = np.random.default_rng(41).normal(size=(24, 8)).astype("float32")
    full = make(normalized=True).fit(x, epochs=1, batch_size=32)
    explicit = make(normalized=True).fit(
        x, epochs=1, batch_size=32, kmeans_init_batch_size=32
    )
    for key, value in full.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)
    half = make(normalized=True).fit(
        x, epochs=1, batch_size=12, kmeans_init_batch_size=32
    )
    # One initialization forward, then two training forwards. It would be four
    # BN updates if initialization were accidentally split into local batches.
    assert half.encoder.normalization[0].num_batches_tracked.item() == 3
    assert half.config["kmeans_init_batch_size"] == 32
