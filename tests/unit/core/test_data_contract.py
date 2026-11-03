from copy import deepcopy

import polars as pl
import pytest
import torch

from genrecsys.core.config import Config
from genrecsys.core.data_contract import DataContract
from genrecsys.core.encoders.id import Observer, Preprocessor
from genrecsys.core.encoders.numeric import Observer as NumericObserver
from genrecsys.core.encoders.numeric import Preprocessor as NumericPreprocessor
from genrecsys.core.encoders.ple import Observer as PleObserver
from genrecsys.core.encoders.ple import Preprocessor as PlePreprocessor
from genrecsys.core.encoders.ple import encode as ple_encode
from genrecsys.core.feature_transforms import expression


def test_unknown_id_uses_reserved_zero_inside_embedding_bounds() -> None:
    observer = Observer()
    observer.observe(pl.Series("merchant", ["known", "known"]))
    preprocessor = Preprocessor.fit(observer, top_k=1)

    assert preprocessor.static_transform(
        pl.Series("merchant", ["missing"])
    ).to_list() == [0]
    assert preprocessor.num_embeddings == 2


def test_namespace_transform_keeps_service_identities_disjoint() -> None:
    values = pl.DataFrame(
        {"event": ["service-a", "service-b"], "brand": [123, 123]}
    ).select(
        expression("namespace", {"fields": ["event", "brand"], "separator": ":"}).alias(
            "token"
        )
    )

    assert values["token"].to_list() == ["service-a:123", "service-b:123"]


def test_numeric_encoder_standardizes_amount_and_handles_missing_values() -> None:
    observer = NumericObserver()
    observer.observe(pl.Series("amount", [10.0, 20.0, None]))
    preprocessor = NumericPreprocessor.fit(observer, clip=3.0)

    transformed = preprocessor.static_transform(pl.Series("amount", [10.0, 20.0, None]))

    assert transformed.to_list() == pytest.approx([-1.0, 1.0, 0.0])


def test_numeric_nonfinite_values_do_not_poison_model_inputs() -> None:
    observer = NumericObserver()
    observer.observe(pl.Series("amount", [10.0, 20.0, float("inf"), float("nan")]))
    preprocessor = NumericPreprocessor.fit(observer)
    assert preprocessor.static_transform(
        pl.Series("amount", [float("inf"), -float("inf"), float("nan"), None])
    ).to_list() == [0.0, 0.0, 0.0, 0.0]


def test_ple_matches_perseus_quantiles_and_piecewise_encoding(tmp_path) -> None:
    observer = PleObserver()
    observer.observe(pl.Series("price", [0.0, 10.0, 20.0]))
    observer.observe(pl.Series("price", [30.0, 40.0, None, float("nan")]))
    preprocessor = PlePreprocessor.fit(observer, max_bins=4)

    assert preprocessor.thresholds == pytest.approx([0.0, 10.0, 20.0, 30.0, 40.0])
    assert preprocessor.num_bins == 4
    transformed = ple_encode(
        torch.tensor([[[5.0]], [[25.0]], [[float("nan")]]]),
        torch.tensor(preprocessor.thresholds),
    )
    torch.testing.assert_close(
        transformed,
        torch.tensor(
            [
                [[0.5, 0.0, 0.0, 0.0]],
                [[1.0, 1.0, 0.5, 0.0]],
                [[0.0, 0.0, 0.0, 0.0]],
            ]
        ),
    )

    preprocessor.save(tmp_path)
    assert PlePreprocessor.load(tmp_path).thresholds == preprocessor.thresholds


def test_ple_quantiles_preserve_duplicate_train_frequencies() -> None:
    observer = PleObserver()
    observer.observe(pl.Series("price", [0.0, 0.0, 10.0, 20.0]))

    assert PlePreprocessor.fit(observer, max_bins=2).thresholds == [0.0, 5.0, 20.0]


def test_explicit_training_recipe_reuses_prepared_without_parsing_old_model(
    tmp_path, monkeypatch
):
    from omegaconf.errors import ConfigKeyError

    from genrecsys.core.checkpoint import Prepared

    recipe = Config.load("tests/fixtures/framework/multi-history.yaml")
    prepared = Prepared(recipe, {}, DataContract.from_config(recipe))
    prepared.save(tmp_path)
    # Simulate a stored recipe whose model-only fields no longer exist.
    stored = (tmp_path / "config.yaml").read_text()
    stored = stored.replace("model:\n", "model:\n  obsolete_binding: {}\n")
    (tmp_path / "config.yaml").write_text(stored)
    with pytest.raises(ConfigKeyError, match="obsolete_binding"):
        Config.load_data(tmp_path / "config.yaml")
    # Keep data contract unchanged; loading preprocessors is verified by lifecycle tests.
    from genrecsys.core.encoders import registry

    monkeypatch.setattr(registry["id"].preprocessor, "load", lambda path: None)
    monkeypatch.setattr(registry["semantic_id"].preprocessor, "load", lambda path: None)
    loaded = Prepared.load(tmp_path, config=recipe)
    assert loaded.config is recipe
    assert loaded.data_contract.sha256 == DataContract.from_config(recipe).sha256


@pytest.mark.parametrize("profile", ["multi-history", "sequence"])
def test_training_only_changes_reuse_prepared_contract(profile) -> None:
    config = Config.load(
        {
            "multi-history": "tests/fixtures/framework/multi-history.yaml",
            "sequence": "tests/fixtures/framework/sequence.yaml",
        }[profile]
    )
    contract = DataContract.from_config(config)
    experiment = deepcopy(config)
    experiment.training.optimizer.params["lr"] = 1e-4
    experiment.model.params["dropout"] = 0.2

    experiment.validate()
    contract.require_compatible(experiment)


def test_365_day_history_requires_repreparation_of_a_356_day_contract():
    config = Config.load("tests/fixtures/framework/sequence.yaml")
    previous = deepcopy(config)
    previous.max_duration_per_sequence = "356d"
    with pytest.raises(ValueError, match="Stage 08.*max_duration_per_sequence"):
        DataContract.from_config(previous).require_compatible(config)


@pytest.mark.parametrize("profile", ["multi-history", "sequence"])
@pytest.mark.parametrize("maximum", [256, 501])
def test_preparation_history_limit_change_requires_repreparation(
    profile, maximum
) -> None:
    config = Config.load(
        {
            "multi-history": "tests/fixtures/framework/multi-history.yaml",
            "sequence": "tests/fixtures/framework/sequence.yaml",
        }[profile]
    )
    contract = DataContract.from_config(config)
    experiment = deepcopy(config)
    experiment.inputs["lifelong"].max_events_per_sequence = maximum
    experiment.validate()
    with pytest.raises(ValueError, match="Stage 08.*histories"):
        contract.require_compatible(experiment)


@pytest.mark.parametrize(
    "change", ["rename", "features", "order", "duplicate", "compression"]
)
def test_model_input_layout_does_not_change_data_contract(change):
    config = Config.load("tests/fixtures/framework/sequence.yaml")
    contract = DataContract.from_config(config)
    experiment = deepcopy(config)
    if change == "rename":
        experiment.inputs["recent"] = experiment.inputs.pop("short")
    elif change == "features":
        for value in experiment.inputs.values():
            value.features = tuple(reversed(value.features))
    elif change == "order":
        experiment.inputs = dict(reversed(list(experiment.inputs.items())))
    elif change == "duplicate":
        experiment.inputs["another_short"] = deepcopy(experiment.inputs["short"])
    else:
        experiment.inputs["lifelong"].compression.params["num_queries"] = 64
    experiment.validate()
    contract.require_compatible(experiment)
    assert "inputs" not in contract.recipe
    assert DataContract.from_config(experiment).sha256 == contract.sha256


def test_default_event_priority_preserves_legacy_contract_shape() -> None:
    config = Config.load("tests/fixtures/framework/multi-history.yaml")
    current = DataContract.from_config(config)
    assert all("priority" not in event for event in current.recipe["events"].values())

    config.events[next(iter(config.events))].priority = 1
    prioritized = DataContract.from_config(config)
    assert prioritized.sha256 != current.sha256
    assert any(
        event.get("priority") == 1 for event in prioritized.recipe["events"].values()
    )


def test_priority_event_cap_is_part_of_materialized_history_contract() -> None:
    config = Config.load("configs/tecd-small/onerec-retail-cross-global.yaml")

    contract = DataContract.from_config(config)

    lifelong = next(
        history
        for history in contract.recipe["histories"]
        if history.get("max_priority_events") == 512
    )
    assert lifelong["max_events"] == 2048


def test_contract_recipe_is_detached_from_mutable_training_config(tmp_path):
    config = Config.load("tests/fixtures/framework/sequence.yaml")
    contract = DataContract.from_config(config)
    config.encoders["item"].preprocessor["artifact"] = "another_sid"
    assert (
        contract.recipe["encoders"]["item"]["preprocessor"]["artifact"] == "item_to_sid"
    )
    contract.save(tmp_path / "contract.json")
    assert DataContract.load(tmp_path / "contract.json") == contract


def test_prepared_requires_saved_contract_without_legacy_fallback(tmp_path):
    from genrecsys.core.checkpoint import Prepared

    Config.load_data("tests/fixtures/framework/sequence.yaml").save(
        tmp_path / "config.yaml"
    )
    with pytest.raises(FileNotFoundError, match="data_contract.json"):
        Prepared.load(tmp_path)


def test_fit_preserves_the_original_prepared_state_and_contract(tmp_path, monkeypatch):
    import importlib
    from types import SimpleNamespace

    from genrecsys.core.checkpoint import Prepared

    config = Config.load("tests/fixtures/framework/sequence.yaml")
    contract = DataContract.from_config(config)
    prepared = Prepared(config, {}, contract)
    prepared.config.inputs["lifelong"].max_events_per_sequence = 256
    state = SimpleNamespace(save=lambda path: None)
    fit_api = importlib.import_module("genrecsys.api.fit_model")
    implementation = SimpleNamespace(
        validate_prepared=None, fit_model=lambda checkpoint, dataset: state
    )
    monkeypatch.setattr(fit_api, "get", lambda name: implementation)
    trained = fit_api.fit_model(prepared, object())
    assert trained.prepared is prepared
    assert trained.prepared.data_contract is contract
    trained.save(tmp_path)
    assert DataContract.load(tmp_path / "data_contract.json") == contract


def test_id_preprocessor_roundtrip_has_no_unused_unknown_flag(tmp_path):
    import json

    observer = Observer()
    observer.observe(pl.Series("merchant", ["known"]))
    preprocessor = Preprocessor.fit(observer)
    preprocessor.save(tmp_path)
    assert "has_unk" not in json.loads((tmp_path / "kwargs.json").read_text())
    assert Preprocessor.load(tmp_path).static_transform(
        pl.Series("merchant", ["missing", "known"])
    ).to_list() == [0, 1]


def test_history_selection_requires_rematerialization():
    config = Config.load("tests/fixtures/tecd-small/retail-features.yaml")
    contract = DataContract.from_config(config)
    config.inputs["lifelong"].selection = "recent"
    with pytest.raises(ValueError, match="histories"):
        contract.require_compatible(config)
