"""SID input layout and independently enabled encoder/decoder relative positions."""

import pytest
import torch

from genrecsys.core.models.onerec.architecture.onerec import OneRec


def recipe(encoder_relative=True, decoder_relative=False):
    return {
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "d_model": 16,
        "d_ff": 32,
        "num_heads": 2,
        "d_kv": 8,
        "dropout_rate": 0.0,
        "vocab_size": 24,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "code_len": 4,
        "input_type_embeddings": False,
        "position_embeddings": ["relative"] if encoder_relative else [],
        "decoder_relative_attention": decoder_relative,
        "inputs": [
            {
                "name": "user",
                "type": "context",
                "features": [
                    {
                        "name": "user_bin",
                        "type": "id",
                        "encoder": "user_bin",
                        "vocab_size": 8,
                    }
                ],
                "event_encoder": "context:user",
                "event_encoder_type": "identity",
                "additive_features": [],
                "max_len": 1,
                "compression": None,
            },
            {
                "name": "history",
                "type": "sequence",
                "features": [
                    {
                        "name": "item",
                        "type": "semantic_id",
                        "encoder": "item",
                        "prefix_length": 4,
                    }
                ],
                "event_encoder": "history",
                "event_encoder_type": "identity",
                "token_layout": "sid_sequence",
                "additive_features": [],
                "max_len": 20,
                "compression": None,
            },
        ],
    }


def batch():
    return {
        "inputs": {
            "user": {
                "attention_mask": torch.ones(2, 1, dtype=torch.long),
                "features": {"user_bin": torch.tensor([[[2]], [[3]]])},
            },
            "history": {
                "attention_mask": torch.tensor([[0, 1], [1, 1]]),
                "features": {
                    "item": torch.tensor(
                        [
                            [[0, 0, 0, 0], [2, 6, 10, 14]],
                            [[3, 7, 11, 15], [4, 8, 12, 16]],
                        ]
                    )
                },
            },
        }
    }


def test_sid_coordinates_and_padding_are_preserved_without_additions():
    model = OneRec(recipe())
    values = batch()
    tokens, mask = model.build_encoder_inputs(values)
    assert tokens.shape == (2, 9, 16)
    assert mask.tolist() == [[1, 0, 0, 0, 0, 1, 1, 1, 1], [1] * 9]
    assert model.max_encoder_positions == 81
    torch.testing.assert_close(
        tokens[:, 0],
        model.event_encoders["context:user"].embeddings["user_bin"](
            torch.tensor([2, 3])
        ),
    )
    torch.testing.assert_close(
        tokens[:, 1:],
        model.model.shared(values["inputs"]["history"]["features"]["item"]).flatten(
            1, 2
        ),
    )
    assert model.input_types is None
    assert not list(model.event_encoders["history"].parameters())


@pytest.mark.parametrize("encoder_relative", [False, True])
@pytest.mark.parametrize("decoder_relative", [False, True])
def test_relative_bias_has_independent_flags_gradients_and_generation(
    encoder_relative, decoder_relative
):
    torch.manual_seed(42)
    model = OneRec(recipe(encoder_relative, decoder_relative)).eval()
    for stack, enabled in [
        (model.model.encoder, encoder_relative),
        (model.model.decoder, decoder_relative),
    ]:
        bias = stack.block[0].layer[0].SelfAttention
        assert bias.has_relative_attention_bias is enabled
        assert hasattr(bias, "relative_attention_bias") is enabled
    labels = torch.tensor([[2, 6, 10, 14], [3, 7, 11, 15]])
    loss, _ = model(batch(), labels)
    loss.backward()
    if decoder_relative:
        grad = (
            model.model.decoder.block[0]
            .layer[0]
            .SelfAttention.relative_attention_bias.weight.grad
        )
        assert grad is not None and grad.abs().sum() > 0
    codes, scores = model.generate_codes_with_scores(
        batch(), num_beams=2, code_len=4, allowed_codes=((2, 6, 10, 14), (3, 7, 11, 15))
    )
    assert codes.shape == (4, 5)
    assert torch.isfinite(scores).all()
    restored = OneRec(recipe(encoder_relative, decoder_relative)).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(batch(), labels)[1], model(batch(), labels)[1])


def test_additive_event_and_stream_types_can_be_enabled_separately():
    cfg = recipe()
    cfg["input_type_embeddings"] = True
    history = cfg["inputs"][1]
    history["features"].append(
        {"name": "event", "type": "id", "encoder": "event", "vocab_size": 3}
    )
    history["additive_features"] = ["event"]
    model = OneRec(cfg)
    values = batch()
    values["inputs"]["history"]["features"]["event"] = torch.ones(
        2, 2, 1, dtype=torch.long
    )
    tokens, _ = model.build_encoder_inputs(values)
    sid = model.model.shared(values["inputs"]["history"]["features"]["item"])
    event = model.event_encoders["history"].embeddings["event"](
        torch.ones(2, 2, dtype=torch.long)
    )
    expected = (sid + event.unsqueeze(2)).flatten(1, 2) + model.input_types.weight[1]
    torch.testing.assert_close(tokens[:, 1:], expected)
    # User receives only its stream type, never the history event embedding.
    expected_user = (
        model.event_encoders["context:user"].embeddings["user_bin"](
            torch.tensor([2, 3])
        )
        + model.input_types.weight[0]
    )
    torch.testing.assert_close(tokens[:, 0], expected_user)


def test_generative_only_filters_future_items_before_filling_ranking():
    from genrecsys.core.models.onerec.inference import _predict_batch

    class Generator:
        code_len = 1

        def generate_codes_with_scores(self, **kwargs):
            return torch.tensor([[0, 2], [0, 3]]), torch.tensor([-0.1, -0.2])

    result = _predict_batch(
        Generator(),
        {"user_id": [0], "reference_time": torch.tensor([100.0])},
        ((2,), (3,)),
        {(2,): ["future", "available"], (3,): ["unknown"]},
        item_created_at={"future": 101.0, "available": 99.0},
        k=2,
        beam_size=2,
    )
    assert [value["item"] for value in result[0]] == ["available", "unknown"]
