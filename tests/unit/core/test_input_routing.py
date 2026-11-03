"""Point-in-time routing tests for shared model inputs."""

from datetime import datetime, timedelta

import polars as pl

from genrecsys.core.config import Config
from genrecsys.core.inputs import select_sequence


def test_sequence_applies_global_history_duration_per_sample() -> None:
    config = Config.load("tests/fixtures/framework/multi-history.yaml")
    timestamp = datetime.fromisoformat("2026-09-23")
    events = pl.DataFrame(
        {
            "timestamp": [
                timestamp - timedelta(days=181),
                timestamp - timedelta(days=180),
                timestamp - timedelta(days=1),
            ],
            "_event": ["item-view"] * 3,
        }
    )

    routed = select_sequence(
        config,
        "short",
        events,
        sample_timestamp=timestamp,
    )

    assert routed["timestamp"].to_list() == [
        timestamp - timedelta(days=180),
        timestamp - timedelta(days=1),
    ]


def test_sequence_limit_prefers_high_priority_events_before_recency() -> None:
    config = Config.load("tests/fixtures/tecd-small/retail-features.yaml")
    timestamp = datetime.fromisoformat("2026-09-23")
    events = pl.DataFrame(
        {
            "timestamp": [
                timestamp - timedelta(days=3),
                timestamp - timedelta(days=2),
                timestamp - timedelta(days=1),
            ],
            "_event": [
                "t-ecd-small-retail-like",
                "t-ecd-small-retail-click",
                "t-ecd-small-retail-added-to-cart",
            ],
        }
    )
    config.inputs["lifelong"].max_events_per_sequence = 2

    routed = select_sequence(
        config,
        "lifelong",
        events,
        sample_timestamp=timestamp,
    )

    assert routed["_event"].to_list() == [
        "t-ecd-small-retail-like",
        "t-ecd-small-retail-added-to-cart",
    ]

    config.inputs["renamed_history"] = config.inputs.pop("lifelong")
    renamed = select_sequence(
        config, "renamed_history", events, sample_timestamp=timestamp
    )
    assert renamed.equals(routed)


def test_short_sequence_keeps_recency_with_prioritized_target_events() -> None:
    config = Config.load("tests/fixtures/tecd-small/retail-features.yaml")
    timestamp = datetime.fromisoformat("2026-09-23")
    events = pl.DataFrame(
        {
            "timestamp": [
                timestamp - timedelta(days=3),
                timestamp - timedelta(days=2),
                timestamp - timedelta(days=1),
            ],
            "_event": [
                "t-ecd-small-retail-like",
                "t-ecd-small-retail-click",
                "t-ecd-small-retail-added-to-cart",
            ],
        }
    )
    config.inputs["short"].max_events_per_sequence = 2

    routed = select_sequence(
        config,
        "short",
        events,
        sample_timestamp=timestamp,
    )

    assert routed["_event"].to_list() == [
        "t-ecd-small-retail-click",
        "t-ecd-small-retail-added-to-cart",
    ]


def test_sequence_can_cap_priority_events_and_fill_with_recent_auxiliary_events() -> (
    None
):
    config = Config.load("configs/tecd-small/onerec-retail-cross-global.yaml")
    timestamp = datetime.fromisoformat("2026-09-23")
    events = pl.DataFrame(
        {
            "timestamp": [
                timestamp - timedelta(hours=value) for value in range(8, 0, -1)
            ],
            "_event": [
                "t-ecd-small-marketplace-click",
                "t-ecd-small-retail-like",
                "t-ecd-small-marketplace-like",
                "t-ecd-small-retail-like",
                "t-ecd-small-marketplace-clickout",
                "t-ecd-small-retail-like",
                "t-ecd-small-marketplace-click",
                "t-ecd-small-retail-like",
            ],
        }
    )
    config.inputs["lifelong"].max_events_per_sequence = 5
    config.inputs["lifelong"].max_priority_events_per_sequence = 2

    routed = select_sequence(
        config,
        "lifelong",
        events,
        sample_timestamp=timestamp,
    )

    assert routed["_event"].to_list() == [
        "t-ecd-small-marketplace-like",
        "t-ecd-small-marketplace-clickout",
        "t-ecd-small-retail-like",
        "t-ecd-small-marketplace-click",
        "t-ecd-small-retail-like",
    ]
