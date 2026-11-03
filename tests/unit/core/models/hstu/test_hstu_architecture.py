"""Perseus attention, point-in-time readout and full/sampled CE equivalence."""

from copy import deepcopy

import pytest
import torch

from genrecsys.core.models.hstu.architecture import HSTU, HSTUBlock
from genrecsys.core.models.training import _to_device


def description(reduction="mean_targets_per_sample", gradient_checkpointing=False):
    return {
        "dim": 8,
        "embedding_dim": 4,
        "num_heads": 2,
        "num_layers": 2,
        "dropout": 0.0,
        "catalog_chunk_size": 2,
        "max_len": 6,
        "num_time_buckets": 16,
        "weight_initialization": "original",
        "feature_fusion": "sum",
        "gradient_checkpointing": gradient_checkpointing,
        "reduction": reduction,
        "loss_type": "cross_entropy",
        "num_negatives": None,
        "context_features": [],
        "target_encoder": "item",
        "features": [
            {
                "name": "item",
                "encoder": "item",
                "type": "id",
                "vocab_size": 7,
                "embedding_dim": 4,
            },
            {
                "name": "event",
                "encoder": "event",
                "type": "id",
                "vocab_size": 4,
                "embedding_dim": 4,
            },
        ],
    }


def batch():
    return {
        "history": {
            "features": {
                "item": torch.tensor(
                    [[[0], [1], [2]], [[2], [0], [3]], [[0], [0], [0]]]
                ),
                "event": torch.tensor(
                    [[[0], [1], [1]], [[1], [2], [1]], [[0], [0], [0]]]
                ),
            },
            "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1], [0, 0, 0]]),
            "timestamps": torch.tensor(
                [[0.0, -2.0, -1.0], [-3.0, -2.0, -1.0], [0.0, 0.0, 0.0]],
                dtype=torch.float64,
            ),
        },
        "target": torch.tensor([[0, 2], [1, -100], [2, 1]]),
    }


def dense_loss(model, data):
    query = model.encode(data)
    scores = query @ model.item_embeddings(model.candidate_tokens).T
    valid = data["target"].ne(-100)
    losses = -scores.log_softmax(-1).gather(-1, data["target"].clamp_min(0))
    losses = losses.masked_fill(~valid, 0)
    if model.description["reduction"] == "mean_targets_global":
        return losses.sum() / valid.sum()
    return (losses.sum(-1) / valid.sum(-1)).mean()


@pytest.mark.parametrize("fusion", ["sum", "concat_projection"])
def test_context_conditions_readout_without_extra_tokens_and_receives_gradients(fusion):
    settings = description()
    settings.update(
        feature_fusion=fusion,
        event_encoder_type="concat_projection",
        additive_features=["event"],
    )
    if fusion == "concat_projection":
        settings["features"][1]["embedding_dim"] = settings["dim"]
    settings["context_features"] = [
        {"name": "region", "encoder": "region", "type": "id", "vocab_size": 5},
        {"name": "socdem", "encoder": "socdem", "type": "id", "vocab_size": 4},
    ]
    model = HSTU(settings)
    model.candidate_tokens = torch.tensor([5, 1, 3])
    data = batch()
    data["context"] = {
        "features": {
            "region": torch.tensor([[[1]], [[2]], [[0]]]),
            "socdem": torch.tensor([[[1]], [[1]], [[0]]]),
        }
    }
    widths = []
    model.layers[0].register_forward_pre_hook(
        lambda _module, args: widths.append(args[0].shape[1])
    )
    original = model.encode(data)
    changed = deepcopy(data)
    changed["context"]["features"]["region"][0] = 3
    assert not torch.allclose(original[0], model.encode(changed)[0])
    assert widths == [data["history"]["attention_mask"].shape[1] + 1] * 2
    loss, _ = model(data, labels=data["target"])
    loss.backward()
    assert model.embeddings["region"].weight.grad[1:].abs().sum() > 0
    assert model.embeddings["socdem"].weight.grad[1:].abs().sum() > 0
    zero_context = deepcopy(data)
    for value in zero_context["context"]["features"].values():
        value.zero_()
    prior = deepcopy(model)
    prior.description["context_features"] = []
    torch.testing.assert_close(model.encode(zero_context), prior.encode(data))


@pytest.mark.parametrize(
    "reduction", ["mean_targets_per_sample", "mean_targets_global"]
)
@pytest.mark.parametrize("checkpointing", [False, True])
def test_chunked_ce_matches_dense_loss_and_every_parameter_gradient(
    reduction, checkpointing
):
    actual = HSTU(description(reduction, checkpointing))
    actual.candidate_tokens = torch.tensor([5, 1, 3])
    expected = deepcopy(actual)
    data = batch()
    loss, _ = actual(data, labels=data["target"])
    reference = dense_loss(expected, data)
    torch.testing.assert_close(loss, reference)
    loss.backward()
    reference.backward()
    for (name, parameter), (_, other) in zip(
        actual.named_parameters(), expected.named_parameters()
    ):
        torch.testing.assert_close(parameter.grad, other.grad, msg=name)


@pytest.mark.parametrize(
    "reduction", ["mean_targets_per_sample", "mean_targets_global"]
)
def test_weighted_microbatches_match_one_global_batch(reduction):
    actual = HSTU(description(reduction))
    actual.candidate_tokens = torch.tensor([5, 1, 3])
    expected = deepcopy(actual)
    data = batch()
    dense_loss(expected, data).backward()
    total = data["target"].ne(-100).sum() if reduction.endswith("global") else 3
    for rows in (slice(0, 1), slice(1, 3)):
        part = {
            "target": data["target"][rows],
            "history": {
                "attention_mask": data["history"]["attention_mask"][rows],
                "timestamps": data["history"]["timestamps"][rows],
                "features": {
                    name: value[rows]
                    for name, value in data["history"]["features"].items()
                },
            },
        }
        loss, _ = actual(part, labels=part["target"])
        (loss * actual.loss_normalizer / total).backward()
    for (name, parameter), (_, other) in zip(
        actual.named_parameters(), expected.named_parameters()
    ):
        torch.testing.assert_close(parameter.grad, other.grad, msg=name)


def test_catalog_chunking_preserves_exact_topk_and_candidate_order():
    model = HSTU(description()).eval()
    candidates = torch.tensor([6, 3, 1, 5, 2])
    data = batch()
    actual, indices = model.topk(data, candidates, k=4)
    scores = model.encode(data) @ model.item_embeddings(candidates).T
    expected, expected_indices = scores.topk(4, -1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(indices[:2], expected_indices[:2])


def test_block_matches_perseus_silu_length_normalization_and_gating():
    model = HSTUBlock(8, 2, 0.0).eval()
    x = torch.randn(2, 4, 8)
    future = torch.ones(4, 4, dtype=torch.bool).triu(1)[None, None]
    bias = torch.randn(1, 1, 4, 4)
    scale = torch.full((2, 1, 1, 1), 4.0)
    actual = model(x, future, bias, scale)
    u, v, q, k = [
        value.reshape(2, 4, 2, 4).transpose(1, 2)
        for value in torch.nn.functional.silu(model.uvqk(model.input_norm(x))).chunk(
            4, -1
        )
    ]
    weights = (
        torch.nn.functional.silu(q @ k.transpose(-2, -1) + bias) / scale
    ).masked_fill(future, 0)
    values = (weights @ v).transpose(1, 2).reshape(2, 4, 8)
    expected = x + model.output(
        model.output_norm(values) * u.transpose(1, 2).reshape(2, 4, 8)
    )
    torch.testing.assert_close(actual, expected)
    changed = x.clone()
    changed[:, 2:] = torch.randn(2, 2, 8) * 10
    torch.testing.assert_close(
        model(changed, future, bias, scale)[:, :2], actual[:, :2]
    )


def test_left_padding_itemless_context_and_empty_history():
    model = HSTU(description()).eval()
    data = batch()
    query = model.encode(data)
    assert torch.isfinite(query[2]).all()
    first = {
        "history": {
            "attention_mask": data["history"]["attention_mask"][:1, 1:],
            "timestamps": data["history"]["timestamps"][:1, 1:],
            "features": {
                name: value[:1, 1:]
                for name, value in data["history"]["features"].items()
            },
        }
    }
    torch.testing.assert_close(model.encode(first)[0], query[0])
    without_auxiliary = deepcopy(data)
    without_auxiliary["history"]["attention_mask"][1, 1] = 0
    assert not torch.allclose(model.encode(without_auxiliary)[1], query[1])


def test_features_are_summed_at_the_embedding_width():
    model = HSTU(description())
    features = batch()["history"]["features"]
    expected = model.embeddings["item"](
        features["item"].squeeze(-1)
    ) + model.embeddings["event"](features["event"].squeeze(-1))
    torch.testing.assert_close(model._event_embeddings(features), expected)


def test_bidirectional_history_readout_isolation_and_signed_bias():
    model = HSTU(description())
    with torch.no_grad():
        model.position_bias.weight[:, 0] = torch.arange(len(model.position_bias.weight))
        model.time_bias.weight.zero_()
    valid = torch.tensor([[True, True, True]])
    times = torch.tensor([[-2.0, -1.0, 0.0]], dtype=torch.float64)
    mask, bias, scale = model._attention(valid, times, torch.float32)
    assert not mask[0, 0, 0, 1] and not mask[0, 0, 1, 0]
    assert mask[0, 0, 0, 2] and mask[0, 0, 1, 2]
    assert not mask[0, 0, 2].any()
    assert bias[0, 0, 0, 1] != bias[0, 0, 1, 0]
    assert scale.item() == 3


def test_time_bias_changes_query_without_changing_item_order():
    model = HSTU(description()).eval()
    with torch.no_grad():
        model.time_bias.weight[:, 0] = torch.arange(len(model.time_bias.weight)) / 10
    data = batch()
    old = model.encode(data)
    data["history"]["timestamps"] *= 10
    new = model.encode(data)
    assert not torch.allclose(old[:2], new[:2])


@pytest.mark.parametrize(
    "reduction", ["mean_targets_per_sample", "mean_targets_global"]
)
def test_sampled_ce_keeps_all_positives_and_matches_restricted_dense_gradients(
    monkeypatch, reduction
):
    settings = description(reduction)
    settings.update(loss_type="sampled_cross_entropy", num_negatives=3)
    actual = HSTU(settings)
    actual.candidate_tokens = torch.tensor([5, 1, 3, 2, 4])
    expected = deepcopy(actual)
    data = batch()
    data["target"] = torch.tensor([[1, 4], [3, -100], [4, 1]])
    monkeypatch.setattr(
        torch, "randint", lambda high, size, device: torch.full(size, 2, device=device)
    )
    loss, _ = actual(data, labels=data["target"])
    indices = torch.tensor([1, 2, 3, 4])
    expected.candidate_tokens = expected.candidate_tokens[indices]
    reference_data = deepcopy(data)
    valid = reference_data["target"].ne(-100)
    reference_data["target"][valid] = torch.searchsorted(
        indices, reference_data["target"][valid]
    )
    reference = dense_loss(expected, reference_data)
    torch.testing.assert_close(loss, reference)
    loss.backward()
    reference.backward()
    for (name, parameter), (_, other) in zip(
        actual.named_parameters(), expected.named_parameters()
    ):
        torch.testing.assert_close(parameter.grad, other.grad, msg=name)


@pytest.mark.parametrize("loss_type", ["cross_entropy", "sampled_cross_entropy"])
def test_bf16_checkpointed_loss_and_backward_remain_finite(loss_type):
    settings = description(gradient_checkpointing=True)
    settings.update(
        loss_type=loss_type,
        num_negatives=2,
    )
    model = HSTU(settings)
    model.candidate_tokens = torch.tensor([5, 1, 3])
    data = batch()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, _ = model(data, labels=data["target"])
    loss.backward()
    assert torch.isfinite(loss)
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_training_device_transfer_preserves_autograd():
    # cpu:0 forces a real copy, exercising the path a CUDA transfer would take.
    values = torch.randn(2, 3, requires_grad=True)
    moved = _to_device({"nested": {"values": values}}, torch.device("cpu:0"))["nested"][
        "values"
    ]
    assert moved is not values
    torch.nn.Linear(3, 1)(moved).sum().backward()
    assert values.grad is not None


@pytest.mark.parametrize("initialization", ["original", "perseus"])
def test_concat_fusion_consumes_features_and_backpropagates(initialization):
    settings = description()
    settings.update(
        feature_fusion="concat_projection",
        weight_initialization=initialization,
        event_encoder_type="concat_projection",
        additive_features=["event"],
        loss_type="sampled_cross_entropy",
        num_negatives=3,
    )
    # Additive embeddings have model width; the tied item table keeps embedding width.
    settings["features"][1]["embedding_dim"] = settings["dim"]
    model = HSTU(settings)
    model.candidate_tokens = torch.tensor([5, 1, 3])
    data = batch()
    query = model.encode(data)
    changed = deepcopy(data)
    changed["history"]["features"]["event"][0, -1] = 2
    assert not torch.allclose(query[0], model.encode(changed)[0])
    assert model.item_embeddings.weight.shape == (7, 4)
    loss, _ = model(data, labels=data["target"])
    loss.backward()
    assert torch.isfinite(loss)
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            assert torch.count_nonzero(module.weight[0]) == 0
