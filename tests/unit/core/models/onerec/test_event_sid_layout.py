"""SID layout reuses event feature transforms and the ordinary compression path."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_sid_sequence import batch, recipe
from torch import nn

from genrecsys.core.models.onerec.architecture.onerec import OneRec
from genrecsys.core.models.onerec.training import initialize_feature_embeddings
from genrecsys.core.models.sequence import EventEncoder


def test_explicit_concat_projection_really_creates_a_projection():
    config = recipe()
    config["inputs"][1]["event_encoder_type"] = "concat_projection"
    model = OneRec(config)
    encoder = model.event_encoders["history"]
    assert isinstance(encoder, EventEncoder)
    assert isinstance(encoder.projection, nn.Sequential)
    assert encoder.projection[0].in_features == config["d_model"]
    values = batch()["inputs"]["history"]["features"]
    expected = encoder.projection(model.model.shared(values["item"])).flatten(1, 2)
    assert torch.equal(encoder(values, model.model.shared), expected)


def feature_description(projected=False):
    history = recipe()["inputs"][1]
    history["features"][0]["optional"] = True
    history["features"] += [
        {
            "name": "category",
            "type": "id",
            "encoder": "category",
            "vocab_size": 5,
            "embedding_dim": 16,
            "frozen_embedding": True,
        },
        {"name": "amount", "type": "numeric", "encoder": "amount"},
        {"name": "age", "type": "ple", "encoder": "age", "thresholds": [0.0, 1.0, 3.0]},
    ]
    history["additive_features"] = ["category", "amount", "age"]
    if projected:
        history["event_encoder_type"] = "concat_projection"
        history["additive_features"] = ["category"]
        history["features"][-1]["embedding_dim"] = 3
    return history


@pytest.mark.parametrize("projected", [False, True])
def test_feature_transforms_broadcast_per_event_and_optional_sid_keeps_other_features(
    projected, tmp_path
):
    description = feature_description(projected)
    encoder = EventEncoder(description, 16, 0.0)
    sid = nn.Embedding(24, 16)
    values = {
        "item": torch.tensor([[[0, 0, 0, 0], [2, 6, 10, 14]]]),
        "category": torch.tensor([[[2], [3]]]),
        "amount": torch.tensor([[[2.0], [4.0]]]),
        "age": torch.tensor([[[0.5], [2.0]]]),
    }
    # Exercise the existing frozen-feature loader, not just the requires_grad flag.
    matrix = np.arange(80, dtype=np.float32).reshape(5, 16) / 80
    path = tmp_path / "category.npy"
    np.save(path, matrix)
    holder = nn.Module()
    holder.event_encoders = nn.ModuleDict({"history": encoder})
    checkpoint = SimpleNamespace(
        config=SimpleNamespace(
            encoders={
                "category": SimpleNamespace(
                    type="id", embedder={"artifact": "category"}
                )
            }
        ),
        files={"category": path},
    )
    initialize_feature_embeddings(holder, checkpoint)
    assert not encoder.embeddings["category"].weight.requires_grad
    assert torch.equal(encoder.embeddings["category"].weight, torch.from_numpy(matrix))
    result = encoder(values, sid).view(1, 2, 4, 16)
    # Known PLE values for thresholds [0,1,3]: [0.5,0], [1,0.5].
    ple = torch.tensor([[[0.5, 0.0], [1.0, 0.5]]])
    sid_values = sid(values["item"]).clone()
    sid_values[:, 0] = 0
    category = torch.from_numpy(matrix)[values["category"].squeeze(-1)]
    if projected:
        parts = [
            sid_values,
            values["amount"].unsqueeze(2).expand(-1, -1, 4, -1),
            encoder.feature_projections["age"](ple).unsqueeze(2).expand(-1, -1, 4, -1),
        ]
        expected = encoder.projection(torch.cat(parts, dim=-1)) + category.unsqueeze(2)
        assert encoder.projection[0].in_features == 16 + 1 + 3
    else:
        extra = category + encoder.additive_projections["amount"](values["amount"])
        extra = extra + encoder.additive_projections["age"](ple)
        expected = sid_values + extra.unsqueeze(2)
    torch.testing.assert_close(result, expected)
    assert result[:, 0].abs().sum() > 0  # Missing SID is not a missing event.
    result.square().sum().backward()
    assert sid.weight.grad[0].count_nonzero() == 0
    assert sid.weight.grad[2].abs().sum() > 0
    assert encoder.embeddings["category"].weight.grad is None
    projections = (
        encoder.feature_projections if projected else encoder.additive_projections
    )
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in projections.parameters()
    )


@pytest.mark.parametrize("projected", [False, True])
def test_sid_qformer_expands_event_masks_and_recency_and_handles_empty_history(
    projected,
):
    config = recipe()
    config.update(
        position_embeddings=["daily_recency", "qformer_recency"],
        daily_recency_vocab_size=8,
    )
    history = feature_description(projected)
    history["max_len"] = 2  # expanded length 8 must NOT index this 3-row recency table
    history["compression"] = {
        "num_queries": 2,
        "num_heads": 2,
        "num_layers": 1,
        "dim_feedforward": 32,
    }
    config["inputs"][1] = history
    model = OneRec(config).eval()
    values = batch()
    values["inputs"]["user"]["daily_recency"] = torch.zeros(2, 1, dtype=torch.long)
    h = values["inputs"]["history"]
    h["daily_recency"] = torch.tensor([[0, 3], [2, 1]])
    h["features"].update(
        category=torch.tensor([[[0], [2]], [[3], [2]]]),
        amount=torch.ones(2, 2, 1),
        age=torch.ones(2, 2, 1),
    )
    # First row is empty; second row's first event has missing optional SID.
    h["attention_mask"][0] = 0
    h["features"]["item"][1, 0] = 0
    captured = []
    hook = model.compressors["history"].register_forward_pre_hook(
        lambda module, args: captured.append(args)
    )
    tokens, mask = model.build_encoder_inputs(values)
    hook.remove()
    assert model.max_encoder_positions == 3
    assert tokens.shape == (2, 3, 16)
    assert mask.tolist() == [[1, 1, 1], [1, 1, 1]]
    expected = model.event_encoders["history"](h["features"], model.model.shared)
    expected = expected + model.daily_recency_position(
        h["daily_recency"]
    ).repeat_interleave(4, dim=1)
    expected = expected + model.input_positions["history"](
        torch.tensor([[0, 0], [2, 1]])
    ).repeat_interleave(4, dim=1)
    torch.testing.assert_close(captured[0][0], expected)
    assert captured[0][1].tolist() == [[1, 0, 0, 0, 0, 0, 0, 0], [1] * 8]
    torch.testing.assert_close(tokens[0, 1:], model.empty_tokens["history"])
    loss, logits = model(values, torch.tensor([[2, 6, 10, 14], [3, 7, 11, 15]]))
    assert torch.isfinite(logits).all()
    loss.backward()
    assert model.empty_tokens["history"].grad.abs().sum() > 0
    assert model.compressors["history"].query_tokens.grad.abs().sum() > 0


def test_mixed_layouts_and_padding_do_not_leak_into_compressed_history():
    config = recipe()
    extra = copy.deepcopy(config["inputs"][1])
    extra.update(
        name="summary",
        event_encoder="summary",
        token_layout="event",
        event_encoder_type="concat_projection",
    )
    extra["compression"] = {
        "num_queries": 2,
        "num_heads": 2,
        "num_layers": 1,
        "dim_feedforward": 32,
    }
    config["inputs"].append(extra)
    model = OneRec(config).eval()
    values = batch()
    values["inputs"]["summary"] = copy.deepcopy(values["inputs"]["history"])
    before, mask = model.build_encoder_inputs(values)
    assert before.shape == (2, 11, 16)
    assert mask[0].tolist() == [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    values["inputs"]["summary"]["features"]["item"][0, 0] = torch.tensor([4, 8, 12, 16])
    after, _ = model.build_encoder_inputs(values)
    torch.testing.assert_close(before, after)


def test_sid_compression_ignores_masked_coordinates():
    config = recipe()
    config["inputs"][1]["compression"] = {
        "num_queries": 2,
        "num_heads": 2,
        "num_layers": 1,
        "dim_feedforward": 32,
    }
    model = OneRec(config).eval()
    values = batch()
    before, _ = model.build_encoder_inputs(values)
    values["inputs"]["history"]["features"]["item"][0, 0] = torch.tensor([4, 8, 12, 16])
    after, _ = model.build_encoder_inputs(values)
    torch.testing.assert_close(before, after)


@pytest.mark.parametrize("width", [1, 3, 4])
def test_sid_prefix_controls_tokens_and_masks_without_projection(width):
    config = recipe()
    config["inputs"][1]["features"][0]["prefix_length"] = width
    model = OneRec(config)
    values = batch()
    tokens, mask = model.build_encoder_inputs(values)
    expected = model.model.shared(
        values["inputs"]["history"]["features"]["item"][..., :width]
    ).flatten(1, 2)
    assert torch.equal(tokens[:, 1:], expected)
    assert mask[0].tolist() == [1] + [0] * width + [1] * width
    assert model.max_encoder_positions == 1 + 20 * width
    assert not list(model.event_encoders["history"].parameters())
