"""Independent pathway limits must survive the EventStore read."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from genrecsys.core.config import Config
from genrecsys.core.event_store import EventStore
from genrecsys.core.models.onerec.samples import Samples


@pytest.mark.parametrize("sid_history", [True, False])
def test_shared_vocabulary_covers_history_and_decoder(sid_history):
    from types import SimpleNamespace

    from genrecsys.core.checkpoint import Prepared
    from genrecsys.core.data_contract import DataContract
    from genrecsys.core.encoders.semantic_id import Preprocessor
    from genrecsys.core.models.onerec.config import _model_config

    config = Config.load("tests/fixtures/framework/multi-history.yaml")
    config.model.params["decoder"]["generation_prefix_length"] = 1
    if not sid_history:
        config.inputs = {"payments": config.inputs["transactions"]}
    mapping = Preprocessor(pl.DataFrame({"item_key": [1], "sid": [[2, 30, 300]]}))
    preprocessors = {
        name: SimpleNamespace(num_embeddings=2) for name in config.encoders
    }
    preprocessors["item"] = mapping
    resolved = _model_config(
        Prepared(config, preprocessors, DataContract.from_config(config))
    )
    assert resolved["code_len"] == 1
    assert resolved["vocab_size"] == (31 if sid_history else 3)


def test_frozen_event_encoders_do_not_observe_training_events():
    from importlib import import_module
    from types import SimpleNamespace

    preparation = import_module("genrecsys.core.preparation")
    config = Config.load("tests/fixtures/framework/sequence.yaml")
    config.encoders["interaction_type"].preprocessor = {"categories": ["fixed"]}

    def unexpected_observe(values):
        raise AssertionError("A frozen encoder must not observe values")

    observers = {
        name: SimpleNamespace(observe=unexpected_observe) for name in config.encoders
    }
    # With no unfinished encoder this returns without even requiring a samples join.
    preparation.observe_events(pl.DataFrame(), pl.DataFrame(), observers, config)


def test_event_observers_ignore_validation_users_and_post_train_events():
    from importlib import import_module

    from genrecsys.core.encoders.id import Observer

    preparation = import_module("genrecsys.core.preparation")
    config = Config.load("tests/fixtures/framework/sequence.yaml")
    cutoff = datetime(2026, 8, 11, tzinfo=UTC)
    events = pl.DataFrame(
        {
            "client_id": ["train", "train", "train", "validation"],
            "timestamp": [
                cutoff - timedelta(days=1),
                cutoff,
                cutoff + timedelta(hours=1),
                cutoff - timedelta(days=1),
            ],
            "interaction_type": ["train-only", "at-cutoff", "future", "holdout-only"],
        }
    )
    samples = pl.DataFrame(
        {
            "client_id": ["train", "train", "validation"],
            "_sample_timestamp": [
                cutoff,
                cutoff + timedelta(days=1),
                cutoff + timedelta(days=1),
            ],
            "_fold": ["train", "validation", "validation"],
        }
    )
    observer = Observer()
    preparation.observe_events(events, samples, {"interaction_type": observer}, config)
    assert observer.value_counts["value"].to_list() == ["train-only"]


@pytest.mark.parametrize("backend", ["disk", "ram"])
def test_old_positives_are_not_cut_by_recent_views_and_transactions(backend):
    config = Config.load("tests/fixtures/framework/multi-history.yaml")
    timestamp = datetime(2026, 9, 7, tzinfo=UTC)
    rows = []
    for event, count, age in (
        ("item-click", 100, 100),
        ("item-view", 500, 2),
        ("merchant-visit", 1000, 1),
    ):
        for index in range(count):
            rows.append(
                {
                    "client_id": "u",
                    "timestamp": timestamp - timedelta(days=age, seconds=index),
                    "_event": event,
                    "event": list(config.events).index(event) + 1,
                    "item": [2, 4, 6] if event != "merchant-visit" else None,
                    "merchant_id": 1 if event == "merchant-visit" else None,
                    "interaction_weekday": 1,
                    "interaction_time_of_day": 1,
                }
            )
    for value in (timestamp, timestamp - timedelta(days=181)):
        rows.append({**rows[0], "timestamp": value})
    events = (
        pl.DataFrame(rows, schema_overrides={"merchant_id": pl.Int64})
        .sort("timestamp", descending=True)
        .with_row_index("position", offset=1)
    )
    store = EventStore(backend=backend)
    with store.open_writer() as writer:
        writer.write(events)
    samples = pl.DataFrame(
        {
            "client_id": ["u", "u"],
            "timestamp": [timestamp, timestamp - timedelta(days=90)],
        }
    )
    data = Samples(samples, config, code_length=3, event_store=store)
    current = data[0]
    assert "source_padding" not in current
    assert len(current["inputs"]["short"]["features"]["item"]["values"]) == 20
    assert len(current["inputs"]["lifelong"]["features"]["item"]["values"]) == 500
    assert len(current["inputs"]["positive"]["features"]["item"]["values"]) == 100
    assert (
        len(current["inputs"]["transactions"]["features"]["merchant_id"]["values"])
        == 1000
    )
    # A second sample uses its own 180-day window, not the latest sample's.
    earlier = data[1]
    assert len(earlier["inputs"]["positive"]["features"]["item"]["values"]) == 100
    assert (
        len(earlier["inputs"]["transactions"]["features"]["merchant_id"]["values"]) == 0
    )


def test_daily_recency_gives_synthetic_same_day_events_one_position():
    config = Config.load("tests/fixtures/tecd-small/retail-features.yaml")
    cutoff = datetime(2026, 8, 25, 18, tzinfo=UTC)
    timestamps = [
        cutoff - timedelta(days=1, hours=1),
        cutoff - timedelta(hours=12),
        cutoff - timedelta(hours=1),
    ]
    events = pl.DataFrame(
        {
            "client_id": ["u"] * 3,
            "timestamp": timestamps,
            "_event": ["t-ecd-small-retail-click"] * 3,
            "event": [1, 1, 1],
            "item": [[2], [3], [4]],
            "os": [1, 1, 1],
            "subdomain": [1, 1, 1],
            "brand_id": [1, 1, 1],
            "category": [1, 1, 1],
            "subcategory": [1, 1, 1],
            "price": [1.0, 1.0, 1.0],
        }
    )
    samples = pl.DataFrame(
        {
            "client_id": ["u"],
            "timestamp": [cutoff],
            "context": [{"socdem_cluster": 1, "region": 1}],
        }
    )

    row = Samples(samples, config, code_length=1, events=events)[0]

    assert row["inputs"]["lifelong"]["daily_recency"] == [2, 1, 1]
    assert row["inputs"]["lifelong"]["padding_length"] == 512


@pytest.mark.parametrize(
    "mode,expected", [("random", [1, 1]), ("label_weighted", [1, 10])]
)
def test_sampled_targets_use_explicit_sampling_policy(monkeypatch, mode, expected):
    from genrecsys.core.models.onerec import samples as onerec_samples

    config = Config.load("tests/fixtures/framework/sequence.yaml")
    config.task.objective.target_mode = mode
    samples = pl.DataFrame(
        {
            "client_id": ["u"],
            "timestamp": [datetime(2026, 8, 25, tzinfo=UTC)],
            "target": [[1, 2]],
            "target_labels": [[1.0, 10.0]],
        }
    )

    def choose(indices, *, weights, k):
        assert weights == expected
        return [indices[0]]

    monkeypatch.setattr(onerec_samples.random, "choices", choose)
    data = Samples(
        samples,
        config,
        code_length=1,
        events=pl.DataFrame(schema={"client_id": pl.String}),
        index_to_sid={1: (2,), 2: (3,)},
        train=True,
    )
    row = data[0]
    assert row["target"] == [2]
    assert {name: value["padding_length"] for name, value in row["inputs"].items()} == {
        "short": 50,
        "positive": 100,
        "lifelong": 500,
    }
