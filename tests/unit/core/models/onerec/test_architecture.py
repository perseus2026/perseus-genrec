"""Dynamic input, generation and independent-target regression tests."""

import copy
from types import MethodType

import pytest
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from genrecsys.core.config import Training
from genrecsys.core.models.onerec.architecture.onerec import OneRec
from genrecsys.core.models.onerec.collate import onerec_collate_fn
from genrecsys.core.models.position import qformer_recency_position_ids
from genrecsys.core.models.semantic_generation import CodebookPrefixLogitsProcessor
from genrecsys.core.models.sequence import EventEncoder
from genrecsys.core.models.training import run_training

SID = {10: [2, 5, 8], 20: [3, 6, 9], 30: [4, 7, 10]}


def _dataset():
    rows = []
    for user, histories, target in (
        (1, ([10, 20], [20], [10, 30]), 10),
        (2, ([30], [10], []), 20),
    ):
        inputs = {}
        for name, items in zip(("recent", "liked", "archive"), histories):
            inputs[name] = {
                "padding_length": 1,
                "features": {
                    "item": {
                        "values": [SID[item] for item in items],
                        "width": 3,
                        "dtype": "long",
                    },
                    "kind": {
                        "values": [[1] for item in items],
                        "width": 1,
                        "dtype": "long",
                    },
                },
            }
        rows.append({"user_id": user, "inputs": inputs, "target": SID[target]})
    return rows


def _onerec_config():
    inputs = []
    for name in ("recent", "liked", "archive"):
        inputs.append(
            {
                "name": name,
                "type": "sequence",
                "features": [
                    {
                        "name": "item",
                        "type": "semantic_id",
                        "encoder": "item",
                        "prefix_length": 3,
                    },
                    {"name": "kind", "type": "id", "encoder": "kind", "vocab_size": 4},
                ],
                "event_encoder": "history",
                "event_encoder_type": "concat_projection",
                "additive_features": ["kind"],
                "max_len": 5,
                "compression": {
                    "num_queries": 2,
                    "num_heads": 4,
                    "dim_feedforward": 32,
                    "num_layers": 1,
                }
                if name == "archive"
                else None,
            }
        )
    return {
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "d_model": 16,
        "d_ff": 32,
        "num_heads": 4,
        "d_kv": 4,
        "dropout_rate": 0.0,
        "vocab_size": 32,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "code_len": 3,
        "inputs": inputs,
    }


def _independent_target_dataset(targets: list[list[int]]) -> list[dict]:
    rows = _dataset()
    for row, item_ids in zip(rows, targets):
        row.pop("target")
        row["independent_targets"] = [SID[item_id] for item_id in item_ids]
    return rows


def test_all_targets_independent_collates_each_target_as_a_separate_sid():
    dataset = _independent_target_dataset([[10, 20], [20, 30, 10]])

    batch = onerec_collate_fn([dataset[0], dataset[1]])

    assert batch["target"].shape == (2, 3, 3)
    assert batch["multi_target_mask"].tolist() == [
        [True, True, False],
        [True, True, True],
    ]
    assert batch["target"][0, 0].tolist() == [2, 5, 8]
    assert batch["target"][0, 1].tolist() == [3, 6, 9]
    assert batch["target"][0, 2].tolist() == [-100, -100, -100]


def test_unknown_backbone_is_rejected():
    with pytest.raises(ValueError, match="backbone must be one of"):
        OneRec({**_onerec_config(), "backbone": "unknown"})


def test_backbone_is_dense_gated_t5():
    from transformers import T5ForConditionalGeneration

    model = OneRec(_onerec_config())
    assert isinstance(model.model, T5ForConditionalGeneration)
    assert model.model.config.feed_forward_proj == "gated-gelu"


def test_original_switch_backbone_is_available_in_dense_mode():
    from transformers import SwitchTransformersForConditionalGeneration

    model = OneRec({**_onerec_config(), "backbone": "switch_dense"})
    assert isinstance(model.model, SwitchTransformersForConditionalGeneration)
    assert model.model.config.num_sparse_encoder_layers == 0
    assert model.model.config.num_sparse_decoder_layers == 0
    assert model.model.config.num_experts == 1
    assert model.model.config.router_aux_loss_coef == 0.0
    assert model.model.config.router_z_loss_coef == 0.0


@pytest.mark.parametrize("backbone", ["t5", "switch_dense"])
def test_non_reentrant_backbone_checkpointing_preserves_loss_and_gradients(backbone):
    """Cover separate encoder/decoder calls, dropout and QFormer on a tiny CPU model."""
    config = {**_onerec_config(), "backbone": backbone, "dropout_rate": 0.1}
    plain = OneRec(config).train()
    checked = copy.deepcopy(plain)
    checked.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
            "preserve_rng_state": True,
        }
    )
    batch = onerec_collate_fn(_dataset())
    losses = []
    for model in (plain, checked):
        torch.manual_seed(42)
        loss, _ = model(batch, labels=batch["target"])
        loss.backward()
        losses.append(loss)
    torch.testing.assert_close(*losses)
    assert checked.model.is_gradient_checkpointing
    for (_, plain_parameter), (_, checked_parameter) in zip(
        plain.named_parameters(), checked.named_parameters(), strict=True
    ):
        if plain_parameter.grad is None:
            assert checked_parameter.grad is None
        else:
            torch.testing.assert_close(plain_parameter.grad, checked_parameter.grad)


def test_all_targets_independent_uses_one_encoder_pass_and_backpropagates():
    dataset = _independent_target_dataset([[10, 20], [20, 30, 10]])
    batch = onerec_collate_fn([dataset[0], dataset[1]])
    config = {
        **_onerec_config(),
        "multi_target_decoder_chunk_size": 1,
    }
    model = OneRec(config)
    model.train()
    encoder_calls = 0

    def count_encoder_calls(_module, _inputs, _output):
        nonlocal encoder_calls
        encoder_calls += 1

    hook = model.model.get_encoder().register_forward_hook(count_encoder_calls)
    try:
        loss, logits = model(batch, labels=batch["target"])
        loss.backward()
    finally:
        hook.remove()

    assert torch.isfinite(loss)
    assert logits.shape == (2, 0, 32)
    assert encoder_calls == 1
    assert model.model.decoder.block[0].layer[0].SelfAttention.q.weight.grad is not None
    assert model.model.encoder.block[0].layer[0].SelfAttention.q.weight.grad is not None


def test_all_targets_independent_loss_averages_targets_within_each_user():
    single = _independent_target_dataset([[10], [20]])
    duplicated = _independent_target_dataset([[10, 10], [20]])
    single_batch = onerec_collate_fn([single[0], single[1]])
    duplicated_batch = onerec_collate_fn([duplicated[0], duplicated[1]])
    config = {
        **_onerec_config(),
        "multi_target_decoder_chunk_size": 1,
    }
    model = OneRec(config)
    model.eval()

    with torch.no_grad():
        single_loss, _ = model(single_batch, labels=single_batch["target"])
        duplicated_loss, _ = model(
            duplicated_batch,
            labels=duplicated_batch["target"],
        )

    torch.testing.assert_close(duplicated_loss, single_loss)


def test_all_targets_global_loss_weights_every_target_equally():
    dataset = _independent_target_dataset([[10, 20], [20]])
    batch = onerec_collate_fn([dataset[0], dataset[1]])
    model = OneRec(
        {
            **_onerec_config(),
            "multi_target_loss_reduction": "mean_tokens_then_targets_global",
        }
    ).eval()
    observed_weights = []

    def capture_weights(
        self,
        encoder_hidden,
        attention_mask,
        user_indices,
        labels,
        pair_weights,
    ):
        del self, attention_mask, user_indices, labels
        observed_weights.append(pair_weights.detach().clone())
        return pair_weights.sum() + encoder_hidden.sum() * 0

    model._independent_target_chunk_loss = MethodType(capture_weights, model)
    with torch.no_grad():
        loss, _ = model(batch, labels=batch["target"])

    torch.testing.assert_close(loss, torch.tensor(1.0))
    torch.testing.assert_close(model.loss_normalizer, torch.tensor(3.0))
    torch.testing.assert_close(
        torch.cat(observed_weights),
        torch.full((3,), 1.0 / 3.0),
    )


def test_all_targets_global_loss_compensates_for_ddp_average(monkeypatch):
    dataset = _independent_target_dataset([[10, 20], [20]])
    batch = onerec_collate_fn([dataset[0], dataset[1]])
    model = OneRec(
        {
            **_onerec_config(),
            "multi_target_loss_reduction": "mean_tokens_then_targets_global",
        }
    ).eval()
    observed_weights = []

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda count, op: count.add_(1),
    )

    def capture_weights(
        self,
        encoder_hidden,
        attention_mask,
        user_indices,
        labels,
        pair_weights,
    ):
        del self, attention_mask, user_indices, labels
        observed_weights.append(pair_weights.detach().clone())
        return pair_weights.sum() + encoder_hidden.sum() * 0

    model._independent_target_chunk_loss = MethodType(capture_weights, model)
    with torch.no_grad():
        loss, _ = model(batch, labels=batch["target"])

    torch.testing.assert_close(loss, torch.tensor(1.5))
    torch.testing.assert_close(model.loss_normalizer, torch.tensor(4.0))
    torch.testing.assert_close(
        torch.cat(observed_weights),
        torch.full((3,), 0.5),
    )


@pytest.mark.parametrize("optimizer_type", [torch.optim.SGD, torch.optim.Adam])
def test_global_target_updates_match_across_microbatches_and_partial_windows(
    optimizer_type, fp32_runtime
):
    """Same seven users form updates of four and three, despite unequal targets."""
    base = _independent_target_dataset([[10], [20, 30, 10]])
    rows = [copy.deepcopy(base[index % 2]) for index in range(7)]
    model_config = {
        **_onerec_config(),
        "multi_target_decoder_chunk_size": 2,
        "multi_target_loss_reduction": "mean_tokens_then_targets_global",
    }
    # Match T-ECD's fixed source positions: dynamic padding would itself move
    # absolute positions when a microbatch contains shorter histories.
    for row in rows:
        for description in model_config["inputs"]:
            row["inputs"][description["name"]]["padding_length"] = description[
                "max_len"
            ]
    initial = OneRec(model_config).state_dict()
    runs = []
    for microbatch, accumulation in ((4, 1), (2, 2), (1, 4)):
        accelerator = Accelerator(
            mixed_precision="no", gradient_accumulation_steps=accumulation
        )
        loader = accelerator.prepare_data_loader(
            DataLoader(rows, batch_size=microbatch, collate_fn=onerec_collate_fn)
        )
        model = OneRec(model_config)
        model.load_state_dict(initial)
        optimizer = optimizer_type(model.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        gradients = []

        def capture_gradients(optimizer, _args, _kwargs, *, gradients=gradients):
            gradients.append(
                torch.cat(
                    [
                        parameter.grad.detach().flatten().clone()
                        for group in optimizer.param_groups
                        for parameter in group["params"]
                        if parameter.grad is not None
                    ]
                )
            )

        optimizer.register_step_pre_hook(capture_gradients)
        model, optimizer = accelerator.prepare(model, optimizer)
        state, history, _ = run_training(
            model,
            optimizer,
            scheduler,
            loader,
            Training(
                num_epochs=2,
                gradient_accumulation_steps=accumulation,
                gradient_clip_norm=0.1,
            ),
            accelerator,
            None,
            dict,
            to_device=lambda batch, device: batch,
        )
        assert len(gradients) == 4
        runs.append((state, gradients, [epoch["train_loss"] for epoch in history]))

    reference_state, reference_gradients, reference_losses = runs[0]
    for state, gradients, losses in runs[1:]:
        for gradient, reference in zip(gradients, reference_gradients, strict=True):
            torch.testing.assert_close(gradient, reference, atol=2e-7, rtol=2e-5)
        for name in state:
            # Adam can amplify FP32 noise in nearly zero gradients through eps.
            torch.testing.assert_close(
                state[name], reference_state[name], atol=1e-5, rtol=2e-5
            )
        assert losses == pytest.approx(reference_losses, abs=2e-6)


@pytest.mark.parametrize("backbone", ["t5", "switch_dense"])
def test_single_target_training_uses_decoder_loss_and_backpropagates(backbone):
    batch = onerec_collate_fn(_dataset())
    model = OneRec({**_onerec_config(), "backbone": backbone}).train()

    loss, logits = model(batch, labels=batch["target"])
    expected_loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        batch["target"].reshape(-1),
        ignore_index=-100,
    )

    torch.testing.assert_close(loss, expected_loss)
    assert logits.shape == (2, 3, 32)
    loss.backward()
    for stack in (model.model.encoder, model.model.decoder):
        gradient = stack.block[0].layer[0].SelfAttention.q.weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


def test_codebook_prefix_logits_processor_masks_by_current_prefix() -> None:
    processor = CodebookPrefixLogitsProcessor(
        allowed_codes=((2, 5), (3, 7)),
        code_len=2,
    )
    min_score = torch.finfo(torch.float32).min

    level0 = processor(
        torch.tensor([[0], [0]]),
        torch.zeros(2, 10),
    )
    assert level0[0, 2].item() == 0.0
    assert level0[0, 3].item() == 0.0
    assert level0[0, 5].item() == min_score

    level1 = processor(
        torch.tensor([[0, 2], [0, 3], [0, 4]]),
        torch.zeros(3, 10),
    )
    assert level1[0, 5].item() == 0.0
    assert level1[0, 7].item() == min_score
    assert level1[1, 7].item() == 0.0
    assert torch.all(level1[2] == min_score)


def test_codebook_prefix_logits_processor_keeps_high_fanout_sparse() -> None:
    # Mirrors the 256x1 production shape where a collision/disambiguation
    # token can give one first-level prefix tens of thousands of children.
    # The compact table must not contain a padded parents x max_children
    # tensor, and masking must preserve only the exact child sets.
    high_fanout_children = tuple(range(10, 2010))
    processor = CodebookPrefixLogitsProcessor(
        allowed_codes=(
            *((2, token) for token in high_fanout_children),
            (3, 7),
        ),
        code_len=2,
    )
    scores = torch.arange(3 * 4096, dtype=torch.float32).reshape(3, 4096)
    original = scores.clone()
    result = processor(
        torch.tensor([[0, 2], [0, 3], [0, 4]]),
        scores,
    )
    min_score = torch.finfo(torch.float32).min

    assert result.data_ptr() == scores.data_ptr()
    assert result[0, 10].item() == original[0, 10].item()
    assert result[0, 2009].item() == original[0, 2009].item()
    assert result[0, 7].item() == min_score
    assert result[1, 7].item() == original[1, 7].item()
    assert result[1, 10].item() == min_score
    assert torch.all(result[2] == min_score)

    tables = processor._level_tables(torch.device("cpu"), 4096)
    assert all(len(table) == 3 for table in tables)
    assert sum(table[2].numel() for table in tables) == 2003


@pytest.mark.parametrize("levels", [2, 3, 4])
@pytest.mark.parametrize("codebook_size", [128, 256, 512])
def test_codebook_prefix_masks_yambda_sweep_with_large_collision_vocab(
    levels, codebook_size
) -> None:
    vocab_size = 34114  # Observed in the failed l4-k128 validation job.
    prefix = tuple(2 + level * codebook_size for level in range(levels))
    alternative = (*prefix[:-1], prefix[-1] + codebook_size - 1)
    allowed_codes = (
        (*alternative, vocab_size - 2),
        (*prefix, vocab_size - 1),
        (*prefix, vocab_size - 3),
        (*alternative, vocab_size - 2),
    )
    processor = CodebookPrefixLogitsProcessor(allowed_codes, code_len=levels + 1)
    min_score = torch.finfo(torch.float32).min

    for level in range(levels + 1):
        prefixes = sorted({code[:level] for code in allowed_codes})
        if level:
            prefixes.append((1,) * level)  # No allowed SID has this prefix.
        scores = torch.arange(len(prefixes) * vocab_size, dtype=torch.float32).reshape(
            len(prefixes), vocab_size
        )
        expected = torch.full_like(scores, min_score)
        for row, current_prefix in enumerate(prefixes):
            children = sorted(
                {
                    code[level]
                    for code in allowed_codes
                    if code[:level] == current_prefix
                }
            )
            expected[row, children] = scores[row, children]

        result = processor(
            torch.tensor([[99, *current_prefix] for current_prefix in prefixes]),
            scores,
        )
        torch.testing.assert_close(result, expected)


def test_codebook_prefix_rejects_overflow_in_parent_keys() -> None:
    processor = CodebookPrefixLogitsProcessor(((2, 3, 4, 5, 6),), code_len=5)
    with pytest.raises(ValueError, match="prefix encoding exceeds int64"):
        processor._level_tables(torch.device("cpu"), 65536)


def test_qformer_recency_position_ids_ignore_left_padding():
    mask = torch.tensor(
        [
            [0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ]
    )

    position_ids = qformer_recency_position_ids(mask)

    torch.testing.assert_close(
        position_ids,
        torch.tensor(
            [
                [0, 0, 3, 2, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
            ]
        ),
    )


def test_onerec_implicit_and_explicit_default_initialization_match():
    config = _onerec_config()
    torch.manual_seed(123)
    implicit_default = OneRec(config)
    torch.manual_seed(123)
    explicit_default = OneRec({**config, "weight_initialization": "default"})

    for name, value in implicit_default.state_dict().items():
        torch.testing.assert_close(value, explicit_default.state_dict()[name])


def test_onerec_rejects_unknown_weight_initialization():
    with pytest.raises(ValueError, match="Unknown weight initialization"):
        OneRec(
            {
                **_onerec_config(),
                "weight_initialization": "magic",
            }
        )


def test_fixed_padding_preserves_feature_rows():
    rows = _dataset()
    for row in rows:
        row["inputs"]["recent"]["padding_length"] = 4
    batch = onerec_collate_fn(rows)
    assert batch["inputs"]["recent"]["features"]["item"].shape == (2, 4, 3)
    assert batch["inputs"]["recent"]["attention_mask"].tolist() == [
        [0, 0, 1, 1],
        [0, 0, 0, 1],
    ]
    assert batch["inputs"]["archive"]["attention_mask"].tolist() == [[1, 1], [0, 0]]


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize(
    "positions",
    [[], ["absolute"], ["qformer_recency"], ["absolute", "qformer_recency"]],
)
def test_optional_compression_positions_and_cold_users(compressed, positions):
    rows, config = _dataset(), _onerec_config()
    config["position_embeddings"] = positions
    if not compressed:
        config["inputs"][2]["compression"] = None
    for row in rows:
        for value in row["inputs"].values():
            for feature in value["features"].values():
                feature["values"] = []
    batch = onerec_collate_fn(rows)
    model = OneRec(config)
    loss, logits = model(batch, labels=batch["target"])
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits).all()
    assert (model.position_embedding is not None) == ("absolute" in positions)
    assert bool(model.input_positions) == (
        compressed and "qformer_recency" in positions
    )


def test_inputs_are_ordered_shared_and_not_named_model_ports():
    model = OneRec(_onerec_config())
    embeddings, mask = model.build_encoder_inputs(onerec_collate_fn(_dataset()))
    assert embeddings.shape == (2, 5, 16)
    assert mask.shape == (2, 5)
    assert list(model.event_encoders) == ["history"]
    assert list(model.compressors) == ["archive"]


def test_numeric_and_context_inputs_use_the_same_projection_path():
    config, rows = _onerec_config(), _dataset()
    for name, kind, compress in (
        ("payments", "sequence", True),
        ("audience", "context", False),
    ):
        config["inputs"].append(
            {
                "name": name,
                "type": kind,
                "event_encoder": name,
                "event_encoder_type": "concat_projection",
                "additive_features": [],
                "max_len": 2 if compress else 1,
                "features": [
                    {
                        "name": "brand",
                        "type": "id",
                        "encoder": "brand",
                        "vocab_size": 8,
                    },
                    {"name": "amount", "type": "numeric", "encoder": "amount"},
                ],
                "compression": {
                    "num_queries": 2,
                    "num_heads": 4,
                    "num_layers": 1,
                    "dim_feedforward": 32,
                }
                if compress
                else None,
            }
        )
        for row in rows:
            row["inputs"][name] = {
                "padding_length": 1,
                "features": {
                    "brand": {"values": [[2]], "width": 1, "dtype": "long"},
                    "amount": {"values": [[0.5]], "width": 1, "dtype": "float32"},
                },
            }
    model = OneRec(config)
    batch = onerec_collate_fn(rows)
    loss, _ = model(batch, labels=batch["target"])
    loss.backward()
    assert torch.isfinite(loss)
    assert model.build_encoder_inputs(batch)[0].shape == (2, 8, 16)
    assert batch["inputs"]["payments"]["features"]["amount"].dtype == torch.float32


def test_ple_expands_scalars_before_the_learned_event_projection():
    encoder = EventEncoder(
        {
            "features": [
                {
                    "name": "price",
                    "type": "ple",
                    "encoder": "price",
                    "thresholds": [0.0, 10.0, 20.0],
                }
            ],
            "additive_features": [],
            "event_encoder_type": "concat_projection",
        },
        d_model=4,
        dropout=0.0,
    )

    encoded = encoder.ple_encoders["price"](torch.tensor([[[5.0], [15.0], [25.0]]]))

    torch.testing.assert_close(
        encoded,
        torch.tensor([[[0.5, 0.0], [1.0, 0.5], [1.0, 1.0]]]),
    )
    assert encoder.projection[0].in_features == 2
    assert encoder(
        {"price": torch.tensor([[[5.0], [15.0], [25.0]]])},
        torch.nn.Embedding(2, 4),
    ).shape == (1, 3, 4)


def test_history_sid_prefix_does_not_use_collision_token():
    config = _onerec_config()
    for value in config["inputs"]:
        value["features"][0]["prefix_length"] = 2
    model = OneRec(config)
    encoder = model.event_encoders["history"]
    values = {"item": torch.tensor([[[2, 5, 8]]]), "kind": torch.tensor([[[1]]])}
    first = encoder(values, model.model.get_input_embeddings())
    values["item"][0, 0, 2] = 10
    torch.testing.assert_close(
        encoder(values, model.model.get_input_embeddings()), first
    )


def test_xavier_uniform_initializes_projection_and_norms():
    model = OneRec({**_onerec_config(), "weight_initialization": "xavier_uniform"})
    projection = model.event_encoders["history"].projection[0]
    assert torch.count_nonzero(projection.weight) > 0
    assert torch.count_nonzero(projection.bias) == 0
    torch.testing.assert_close(
        model.model.encoder.final_layer_norm.weight,
        torch.ones_like(model.model.encoder.final_layer_norm.weight),
    )


@pytest.mark.parametrize("backbone", ["t5", "switch_dense"])
def test_relative_position_bias_is_disabled(backbone):
    model = OneRec({**_onerec_config(), "backbone": backbone})
    assert not any("relative_attention_bias" in key for key in model.state_dict())


def test_absolute_positions_cover_short_like_and_compressed_lifelong_tokens():
    model = OneRec({**_onerec_config(), "position_embeddings": ["absolute"]})
    batch = onerec_collate_fn(_dataset())
    with torch.no_grad():
        model.position_embedding.weight.zero_()
        baseline, _ = model.build_encoder_inputs(batch)
        model.position_embedding.weight.copy_(
            torch.arange(model.max_encoder_positions)[:, None].expand(-1, model.d_model)
        )
        positioned, _ = model.build_encoder_inputs(batch)
    expected = torch.arange(positioned.shape[1], dtype=positioned.dtype)
    torch.testing.assert_close(
        positioned - baseline, expected[None, :, None].expand_as(positioned)
    )


def test_decoder_sid_level_embeddings_train_and_generate():
    model = OneRec({**_onerec_config(), "decoder_level_embeddings": True})
    batch = onerec_collate_fn(_dataset())
    loss, _ = model(batch, labels=batch["target"])
    loss.backward()
    assert model.decoder_level_embedding.weight.grad is not None
    codes, _ = model.eval().generate_codes_with_scores(
        batch,
        num_beams=2,
        code_len=3,
        allowed_tokens_by_level=((2, 3, 4), (5, 6, 7), (8, 9, 10)),
    )
    assert codes.shape == (4, 4)


@pytest.mark.parametrize("code_len", [2, 3])
def test_generation_returns_sequence_log_probabilities(code_len):
    model = OneRec({**_onerec_config(), "code_len": code_len}).eval()
    batch = onerec_collate_fn(_dataset())
    allowed_levels = ((2, 3, 4), (5, 6, 7), (8, 9, 10))[:code_len]
    codes, scores = model.generate_codes_with_scores(
        batch, num_beams=2, code_len=code_len, allowed_tokens_by_level=allowed_levels
    )
    assert codes.shape == (4, code_len + 1)
    assert scores.shape == (4,)
    assert torch.isfinite(scores).all()
    assert torch.all(scores <= 0)
    for level, allowed in enumerate(allowed_levels):
        assert set(codes[:, level + 1].tolist()) <= set(allowed)


def test_generation_respects_five_token_sids_with_large_collision_vocab():
    code_len = 5
    model = OneRec(
        {**_onerec_config(), "code_len": code_len, "vocab_size": 34114}
    ).eval()
    allowed_codes = ((2, 5, 8, 11, 34113), (3, 6, 9, 12, 34112))

    codes, scores = model.generate_codes_with_scores(
        onerec_collate_fn(_dataset()),
        num_beams=2,
        code_len=code_len,
        allowed_codes=allowed_codes,
    )

    assert codes.shape == (4, code_len + 1)
    assert set(map(tuple, codes[:, 1:].tolist())) == set(allowed_codes)
    assert scores.shape == (4,)
    assert torch.isfinite(scores).all()
    assert torch.all(scores <= 0)
