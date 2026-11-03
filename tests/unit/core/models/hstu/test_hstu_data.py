"""Full-SID identity recovery, compatible recipes and point-in-time lifelong reads."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

from genrecsys.core.config import Config
from genrecsys.core.data_contract import DataContract
from genrecsys.core.models.hstu.data import Samples, collate, item_tokens, sid_lookup
from scripts.experiments import DATASETS, models


def _history_config():
    config = Config.load("configs/tecd-small/hstu-retail-md-to-sd.yaml")
    config.inputs = {
        name: value for name, value in config.inputs.items() if value.type == "sequence"
    }
    config.inputs["lifelong"].features = ("item", "event", "brand_id")
    return config


def test_prepared_user_context_is_collated_and_missing_context_stays_zero():
    from datetime import UTC, datetime

    config = Config.load("configs/tecd-small/hstu-retail-md-to-sd.yaml")
    samples = pl.DataFrame(
        {
            "client_id": ["warm", "cold"],
            "timestamp": [datetime(2026, 1, 3, tzinfo=UTC)] * 2,
            "context": [{"region": 2, "socdem_cluster": 3}, None],
        }
    )
    lookup = sid_lookup(pl.DataFrame({"sid": [[2, 4, 7]], "item_token": [1]}))
    rows = Samples(
        samples, config, lookup, events=pl.DataFrame(schema={"client_id": pl.String})
    )
    result = collate([rows[0], rows[1]])
    assert result["context"]["features"]["region"].tolist() == [[[2]], [[0]]]
    assert result["context"]["features"]["socdem_cluster"].tolist() == [[[3]], [[0]]]
    assert result["history"]["attention_mask"].tolist() == [[0], [0]]


@pytest.mark.parametrize(
    "path",
    sorted(
        path
        for path in Path("configs").glob("*/hstu-*.yaml")
        if path.name != "hstu-base.yaml"
    ),
)
def test_hstu_recipes_preserve_the_prepared_data_contract(path):
    if path.parent.name in DATASETS:
        model = next(
            model for model in models(path.parent.name) if model["name"] == path.stem
        )
        parent = path.parent / f"onerec-{model['prepared']}.yaml"
    else:
        parent = path.parent / OmegaConf.load(path).extends
    config = Config.load(path)
    original = Config.load(parent)
    DataContract.from_config(original).require_compatible(config)
    assert set(config.model.params) == {
        "dim",
        "embedding_dim",
        "num_heads",
        "num_layers",
        "dropout",
        "catalog_chunk_size",
        "max_len",
        "num_time_buckets",
        "weight_initialization",
        "feature_fusion",
    }
    encoder_budget = sum(
        value.compression.params["num_queries"]
        if value.compression
        else (value.max_events_per_sequence if value.type == "sequence" else 1)
        for value in original.inputs.values()
    )
    assert config.model.params["max_len"] == 512
    assert config.model.params["embedding_dim"] == 256
    assert config.model.params["weight_initialization"] == "perseus"
    assert config.task.objective.loss["type"] == "sampled_cross_entropy"
    assert config.task.objective.loss["num_negatives"] == 4096
    assert config.training.optimizer.params["lr"] == 2e-3

    if original.model.type == "onerec":
        assert encoder_budget == 512
    assert "beam_size" not in config.inference.predict_kwargs
    assert "item" in config.inputs["lifelong"].features
    assert config.inputs["lifelong"].compression is None
    assert (
        config.training.dataloader.batch_size
        * config.training.gradient_accumulation_steps
        * 8
        == 4096
    )
    if "md-to-sd" in path.stem:
        assert "item" in config.inputs["lifelong"].optional_features


def test_full_sid_including_collision_level_recovers_distinct_items():
    tokens = pl.DataFrame(
        {"sid": [[2, 4, 7], [2, 4, 8], [3, 5, 7]], "item_token": [3, 1, 2]}
    )
    lookup = sid_lookup(tokens)
    assert lookup[0].dtype.kind == "V"
    assert item_tokens(pl.Series("item", [[2, 4, 8], None, [2, 4, 7]]), lookup) == [
        1,
        0,
        3,
    ]
    with pytest.raises(ValueError, match="SID absent"):
        item_tokens(pl.Series("item", [[2, 4, 9]]), lookup)


def test_only_lifelong_is_read_and_history_cutoff_retains_itemless_context():
    from datetime import UTC, datetime

    config = _history_config()
    config.inputs["lifelong"].max_events_per_sequence = 10
    config.inputs["lifelong"].max_priority_events_per_sequence = None
    main = "t-ecd-small-retail-click"
    auxiliary = "t-ecd-payments"
    dates = [datetime(2026, 1, day, tzinfo=UTC) for day in (1, 2, 3, 4)]
    events = pl.DataFrame(
        {
            "client_id": ["u"] * 4,
            "timestamp": dates,
            "_event": [main, auxiliary, main, main],
            "event": [1, 2, 1, 1],
            "brand_id": [1, 2, 1, 1],
            "item": [[2, 4, 7], None, [2, 4, 8], [2, 4, 7]],
        }
    )
    samples = pl.DataFrame(
        {
            "client_id": ["u", "cold"],
            "timestamp": [dates[3]] * 2,
            "history_timestamp": [dates[2]] * 2,
        }
    )
    lookup = sid_lookup(
        pl.DataFrame({"sid": [[2, 4, 7], [2, 4, 8]], "item_token": [1, 2]})
    )
    data = Samples(samples, config, lookup, events=events)
    result = collate([data[0], data[1]])
    assert result["history"]["features"]["item"].squeeze(-1).tolist() == [
        [1, 0],
        [0, 0],
    ]
    assert result["history"]["attention_mask"].tolist() == [[1, 1], [0, 0]]
    np.testing.assert_allclose(
        result["history"]["timestamps"].numpy(),
        [[-259200.0, -172800.0], [0.0, 0.0]],
        rtol=0,
        atol=1e-9,
    )
    assert set(result) == {"user_id", "history"}
    assert np.isfinite(result["history"]["features"]["brand_id"].numpy()).all()
    config.inputs["lifelong"].optional_features = ()
    required = collate([data[0]])
    assert required["history"]["features"]["item"].squeeze(-1).tolist() == [[1]]


@pytest.mark.parametrize("history_limit,expected", [(10000, 3), (2, 2)])
def test_model_budget_truncates_history_without_changing_prepared_contract(
    history_limit, expected
):
    from datetime import UTC, datetime, timedelta

    config = _history_config()
    config.inputs["lifelong"].max_events_per_sequence = history_limit
    contract = DataContract.from_config(config)
    config.model.params["max_len"] = 4
    day = datetime(2026, 1, 1, tzinfo=UTC)
    events = pl.DataFrame(
        {
            "client_id": ["u"] * 8,
            "timestamp": [day + timedelta(days=i) for i in range(8)],
            "_event": ["t-ecd-small-retail-click"] * 8,
            "event": [1] * 8,
            "brand_id": [1] * 8,
            "item": [[2, 4, 7]] * 8,
        }
    )
    samples = pl.DataFrame({"client_id": ["u"], "timestamp": [day + timedelta(days=9)]})
    lookup = sid_lookup(pl.DataFrame({"sid": [[2, 4, 7]], "item_token": [1]}))
    row = Samples(samples, config, lookup, events=events)[0]
    assert len(row["history"]["features"]["item"]["values"]) == expected
    np.testing.assert_allclose(
        row["history"]["timestamps"],
        [-345600.0, -259200.0, -172800.0][-expected:],
        rtol=0,
        atol=1e-9,
    )
    contract.require_compatible(config)
