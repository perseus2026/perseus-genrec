"""Configuration contracts for the Synthetic OneRec profile."""

from pathlib import Path

import polars as pl
import pytest

from genrecsys.core.config import Config


def test_multiple_parents_compose_data_and_model_with_a_shared_base(tmp_path):
    from omegaconf import OmegaConf

    from genrecsys.core.config import _load_recipe

    values = {
        "base.yaml": {"training": {"num_epochs": 30}},
        "data.yaml": {
            "extends": "base.yaml",
            "inputs": {"lifelong": {"max_events_per_sequence": 512}},
        },
        "model.yaml": {
            "extends": "base.yaml",
            "model": {"type": "hstu", "params": {"dim": 256}},
            "inference": {"predict_kwargs": {"k": 100}},
        },
        "recipe.yaml": {
            "extends": ["data.yaml", "model.yaml"],
            "training": {"num_epochs": 20},
        },
    }
    for name, value in values.items():
        OmegaConf.save(OmegaConf.create(value), tmp_path / name)
    actual = OmegaConf.to_container(
        _load_recipe(tmp_path / "recipe.yaml", seen=set()), resolve=True
    )
    assert actual == {
        "training": {"num_epochs": 20},
        "inputs": {"lifelong": {"max_events_per_sequence": 512}},
        "model": {"type": "hstu", "params": {"dim": 256}},
        "inference": {"predict_kwargs": {"k": 100}},
    }


def test_multiple_parents_reject_a_real_inheritance_cycle(tmp_path):
    from omegaconf import OmegaConf

    from genrecsys.core.config import _load_recipe

    OmegaConf.save(OmegaConf.create({"extends": ["b.yaml"]}), tmp_path / "a.yaml")
    OmegaConf.save(OmegaConf.create({"extends": "a.yaml"}), tmp_path / "b.yaml")
    with pytest.raises(ValueError, match="cyclic config extends"):
        _load_recipe(tmp_path / "a.yaml", seen=set())


@pytest.mark.parametrize("predict", [{"k": 10}, None])
def test_switching_model_replaces_architecture_dictionaries(tmp_path, predict):
    from omegaconf import OmegaConf

    from genrecsys.core.config import _load_recipe

    parent = {
        "model": {"type": "first", "params": {"decoder_layers": 4}},
        "inference": {"predict_kwargs": {"k": 100, "beam_size": 200}},
        "inputs": {"history": {"max_events_per_sequence": 512}},
    }
    child = {
        "extends": "parent.yaml",
        "model": {"type": "second", "params": {"dim": 64}},
    }
    if predict is not None:
        child["inference"] = {"predict_kwargs": predict}
    OmegaConf.save(OmegaConf.create(parent), tmp_path / "parent.yaml")
    OmegaConf.save(OmegaConf.create(child), tmp_path / "child.yaml")
    actual = OmegaConf.to_container(
        _load_recipe(tmp_path / "child.yaml", seen=set()), resolve=True
    )
    assert actual["model"] == child["model"]
    assert actual["inference"]["predict_kwargs"] == (predict or {})
    assert actual["inputs"] == parent["inputs"]


def test_same_model_overlay_keeps_ordinary_nested_merging(tmp_path):
    from omegaconf import OmegaConf

    from genrecsys.core.config import _load_recipe

    parent = {
        "model": {"type": "same", "params": {"dim": 16, "encoder": {"layers": 2}}},
        "inference": {"predict_kwargs": {"k": 100, "beam_size": 200}},
    }
    child = {"extends": "parent.yaml", "model": {"type": "same", "params": {"dim": 32}}}
    OmegaConf.save(OmegaConf.create(parent), tmp_path / "parent.yaml")
    OmegaConf.save(OmegaConf.create(child), tmp_path / "child.yaml")
    actual = OmegaConf.to_container(
        _load_recipe(tmp_path / "child.yaml", seen=set()), resolve=True
    )
    assert actual["model"]["params"] == {"dim": 32, "encoder": {"layers": 2}}
    assert actual["inference"] == parent["inference"]


def test_temporal_features_have_configurable_names():
    from genrecsys.core.config import Encoder, Feature, LocatedIn, Transform
    from genrecsys.core.feature_engineering import prepare_events

    config = Config.load("tests/fixtures/framework/sequence.yaml")
    config.encoders["time"] = Encoder(
        type="id", preprocessor={"categories": list(range(7))}
    )
    for name, derive in (("day_of_week", "weekday"), ("day_period", "time_of_day")):
        config.features[name] = Feature(
            located_in=LocatedIn(event=True),
            encoder="time",
            transform=Transform(type=derive),
        )
    for value in config.inputs.values():
        value.features = (*value.features, "day_of_week", "day_period")
    config.validate()
    events = pl.DataFrame(
        {
            "timestamp": ["2026-09-24T13:00:00"],
            "event": ["select"],
            "interaction_subtype": ["details_opened"],
        }
    ).with_columns(pl.col("timestamp").str.to_datetime())
    derived = prepare_events(events, config)
    assert derived.select("day_of_week", "day_period").row(0) == (3, 2)


def test_config_round_trip_is_idempotent(tmp_path: Path) -> None:
    config = Config.load("tests/fixtures/framework/multi-history.yaml")
    path = tmp_path / "config.yaml"
    config.save(path)
    restored = Config.load(path)
    restored.validate()
    assert restored == config


def test_data_validation_does_not_load_a_model_plugin(tmp_path, monkeypatch):
    from genrecsys.core import models

    config = Config.load_data("tests/fixtures/framework/sequence.yaml")
    config.model.type = "not_installed"
    path = tmp_path / "config.yaml"
    config.save(path)

    def reject_model_lookup(name):
        raise AssertionError(f"model lookup: {name}")

    monkeypatch.setattr(models, "get", reject_model_lookup)
    loaded = Config.load_data(path)
    loaded.validate_data()
    with pytest.raises(AssertionError, match="model lookup: not_installed"):
        loaded.validate()


@pytest.mark.parametrize(
    "change", ["compression", "heads", "projection", "aggregation"]
)
def test_preparation_does_not_validate_model_input_architecture(change):
    from genrecsys.core.config import Aggregation
    from genrecsys.core.data_contract import DataContract

    config = Config.load("tests/fixtures/framework/multi-history.yaml")
    original = DataContract.from_config(config)
    if change == "compression":
        config.inputs["lifelong"].compression.type = "not_a_qformer"
    elif change == "heads":
        config.inputs["lifelong"].compression.params["num_heads"] = 0
    elif change == "projection":
        config.inputs["short"].event_encoder = "not_installed"
    else:
        config.inputs["user"].aggregation = Aggregation(type="not_installed")
    config.validate_data()
    original.require_compatible(config)
    with pytest.raises(ValueError):
        config.validate()


@pytest.mark.parametrize("metric", ["Recall@100", "NDCG@100"])
def test_single_metric_config_is_valid(metric, tmp_path):
    config = Config.load("tests/fixtures/framework/sequence.yaml")
    config.task.metrics = {metric: config.task.metrics[metric]}
    config.training.early_stopping.metric = metric
    config.validate()
    path = tmp_path / "config.yaml"
    config.save(path)
    assert set(Config.load(path).task.metrics) == {metric}


def test_early_stopping_still_requires_a_declared_metric():
    config = Config.load("tests/fixtures/framework/sequence.yaml")
    config.task.metrics = {"NDCG@100": config.task.metrics["NDCG@100"]}
    with pytest.raises(ValueError, match="early-stopping metric"):
        config.validate()


def test_onerec_rejects_unknown_model_parameters() -> None:
    config = Config.load("tests/fixtures/framework/multi-history.yaml")
    config.model.params["legacy_experiment"] = True

    with pytest.raises(ValueError, match="unknown fields.*legacy_experiment"):
        config.validate()


def test_all_targets_accepts_global_target_reduction() -> None:
    config = Config.load("tests/fixtures/framework/multi-history.yaml")
    config.task.objective.loss["reduction"] = "mean_tokens_then_targets_global"

    config.validate()


def test_config_rejects_unknown_event_store_backend() -> None:
    config = Config.load("tests/fixtures/framework/multi-history.yaml")
    config.inference.event_store_backend = "unknown"

    with pytest.raises(ValueError, match="event_store_backend must be disk or ram"):
        config.validate()


@pytest.mark.parametrize("profile", ["multi-history", "sequence"])
@pytest.mark.parametrize("backbone", ["t5", "switch_dense"])
def test_active_profiles_accept_both_backbones(profile, backbone):
    config = Config.load(
        {
            "multi-history": "tests/fixtures/framework/multi-history.yaml",
            "sequence": "tests/fixtures/framework/sequence.yaml",
        }[profile]
    )
    config.model.params["backbone"] = backbone
    config.validate()


def test_unknown_backbone_is_rejected():
    config = Config.load("tests/fixtures/framework/multi-history.yaml")
    config.model.params["backbone"] = "unknown"
    with pytest.raises(ValueError, match="backbone must be one of"):
        config.validate()


def test_sequence_omits_default_initialization_but_resolves_it_for_the_model():
    from genrecsys.core.models.onerec.config import model_parameters

    config = Config.load("tests/fixtures/framework/sequence.yaml")
    assert config.model.params.pop("weight_initialization") == "default"
    implicit = model_parameters(config)
    config.model.params["weight_initialization"] = "default"
    config.validate()
    assert implicit == model_parameters(config)


def test_event_with_optional_subtype_keeps_stream_identity():
    from genrecsys.core.feature_engineering import prepare_events

    config = Config.load("tests/fixtures/framework/sequence.yaml")
    events = pl.DataFrame(
        {
            "event": ["select", "feedback", "revisit"],
            "interaction_subtype": ["opened", "like", None],
        }
    )

    assert prepare_events(events, config)["interaction_type"].to_list() == [
        "select:opened",
        "feedback:like",
        "revisit",
    ]


@pytest.mark.parametrize("layout", ["dynamic", "fixed"])
def test_onerec_accepts_both_source_position_layouts(layout):
    config = Config.load("tests/fixtures/framework/multi-history.yaml")
    config.model.params["encoder"]["source_position_layout"] = layout
    config.validate()


def test_onerec_rejects_unknown_source_position_layout():
    config = Config.load("tests/fixtures/framework/multi-history.yaml")
    config.model.params["encoder"]["source_position_layout"] = "unknown"
    with pytest.raises(
        ValueError, match="source_position_layout must be dynamic or fixed"
    ):
        config.validate()


@pytest.mark.parametrize("profile", ["multi-history", "sequence"])
def test_inputs_can_be_renamed_and_qformer_is_optional(profile):
    config = Config.load(
        {
            "multi-history": "tests/fixtures/framework/multi-history.yaml",
            "sequence": "tests/fixtures/framework/sequence.yaml",
        }[profile]
    )
    config.inputs = {
        f"source_{index}": value for index, value in enumerate(config.inputs.values())
    }
    for value in config.inputs.values():
        value.compression = None
    config.validate()
    assert not hasattr(config.model, "input_bindings")
    assert config.training.mixed_precision == "no"


@pytest.mark.parametrize(
    "positions",
    [[], ["absolute"], ["qformer_recency"], ["absolute", "qformer_recency"]],
)
def test_position_embeddings_are_declared_as_a_list(positions):
    config = Config.load("tests/fixtures/framework/sequence.yaml")
    config.model.params["encoder"]["position_embeddings"] = positions
    config.validate()
