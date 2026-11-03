"""Validation of the public Amazon recipe and model option combinations."""

import copy

import pytest

from genrecsys.core import Config


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("d_kv", 0, ValueError),
        ("d_kv", True, TypeError),
        ("input_type_embeddings", "false", TypeError),
    ],
)
def test_invalid_options_rejected(field, value, error):
    config = Config.load("configs/amazon2014-grid/beauty.yaml")
    config.model.params[field] = value
    with pytest.raises(error):
        config.validate()


def test_context_identity_and_sid_sequence_validate_feature_contracts():
    base = Config.load("configs/amazon2014-grid/beauty.yaml")
    config = copy.deepcopy(base)
    config.inputs["user"].features = ["user_bin", "item"]
    with pytest.raises(ValueError, match="identity context"):
        config.validate()
    config = copy.deepcopy(base)
    config.model.params["decoder"]["relative_attention"] = "yes"
    with pytest.raises(TypeError, match="relative_attention"):
        config.validate()
    config = copy.deepcopy(base)
    config.inputs["history"].optional_features = ["item"]
    config.validate()
    config = copy.deepcopy(base)
    config.model.params["decoder"]["relative_attention"] = True
    config.validate()


@pytest.mark.parametrize("layout", ["unknown", False, None])
def test_unknown_event_token_layout_rejected(layout):
    config = Config.load("configs/amazon2014-grid/beauty.yaml")
    config.event_encoders["history"].params["token_layout"] = layout
    with pytest.raises(ValueError, match="token_layout"):
        config.validate()


@pytest.mark.parametrize("encoder_type", ["identity", "concat_projection"])
def test_sid_layout_supports_features_and_qformer(encoder_type):
    from genrecsys.core.config import Compression, Encoder, Feature, LocatedIn

    config = Config.load("configs/amazon2014-grid/beauty.yaml")
    config.encoders["amount"] = Encoder(
        type="numeric", preprocessor={"normalization": "none"}
    )
    config.features["amount"] = Feature(
        encoder="amount", located_in=LocatedIn(event=True)
    )
    history = config.inputs["history"]
    history.features = ["item", "event", "amount"]
    history.optional_features = ["item"]
    history.compression = Compression(
        type="qformer",
        params={
            "num_queries": 2,
            "num_heads": 2,
            "num_layers": 1,
            "dim_feedforward": 32,
        },
    )
    encoder = config.event_encoders["history"]
    encoder.type = encoder_type
    encoder.params["additive_features"] = (
        ["event", "amount"] if encoder_type == "identity" else ["event"]
    )
    config.model.params["encoder"]["position_embeddings"] = [
        "relative",
        "qformer_recency",
    ]
    config.validate()


def test_sid_identity_rejects_nonadditive_features_and_wrong_additive_width():
    config = Config.load("configs/amazon2014-grid/beauty.yaml")
    config.inputs["history"].features = ["item", "event"]
    with pytest.raises(ValueError, match="non-SID features to be additive"):
        config.validate()
    config.event_encoders["history"].params["additive_features"] = ["event"]
    config.encoders["category"] = copy.deepcopy(config.encoders["event"])
    config.features["category"] = copy.deepcopy(config.features["event"])
    config.features["category"].encoder = "category"
    config.inputs["history"].features = ["item", "category"]
    config.event_encoders["history"].params["additive_features"] = ["category"]
    config.encoders["category"].embedder["dim"] = 7
    with pytest.raises(ValueError, match="embedding dim"):
        config.validate()


def test_sid_layout_requires_one_sid_and_consistent_optional_semantics():
    config = Config.load("configs/amazon2014-grid/beauty.yaml")
    config.features["other_item"] = copy.deepcopy(config.features["item"])
    config.inputs["history"].features = ["item", "other_item"]
    with pytest.raises(ValueError, match="exactly one SID"):
        config.validate()
    config.inputs["history"].features = ["event"]
    with pytest.raises(ValueError, match="exactly one SID"):
        config.validate()
    config.inputs["history"].features = ["item"]
    config.inputs["summary"] = copy.deepcopy(config.inputs["history"])
    config.inputs["summary"].optional_features = ["item"]
    with pytest.raises(ValueError, match="same optional features"):
        config.validate()
