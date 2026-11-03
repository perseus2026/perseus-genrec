"""Reject malformed retrieval bundles before expensive preparation."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from genrecsys.core.basis import validate_samples


@pytest.fixture
def samples():
    now = datetime(2026, 9, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            "sample_id": [1, 2],
            "client_id": ["u", "u"],
            "timestamp": [now, now + timedelta(days=1)],
            "history_timestamp": [now - timedelta(days=1), now],
            "target": [["a"], ["b"]],
            "target_labels": [[1.0], [2.0]],
        }
    )


@pytest.mark.parametrize("case", ["identity", "cutoff", "labels", "null_target"])
def test_invalid_queries_fail_early(samples, case):
    if case == "identity":
        samples = pl.concat([samples.head(1), samples.head(1)])
        message = "duplicate query identity"
    elif case == "cutoff":
        samples = samples.with_columns(
            history_timestamp=pl.col("timestamp") + pl.duration(days=1)
        )
        message = "invalid history cutoff"
    elif case == "labels":
        samples = samples.with_columns(target_labels=pl.lit([1.0, 2.0]))
        message = "lengths differ"
    else:
        samples = samples.with_columns(target=pl.lit([None], dtype=pl.List(pl.String)))
        message = "null target identity"
    with pytest.raises(ValueError, match=message):
        validate_samples(samples, pl.DataFrame({"item_id": ["a", "b"]}), split="train")


def test_sample_id_distinguishes_queries_at_the_same_clock(samples):
    samples = samples.with_columns(timestamp=pl.col("timestamp").min())
    validate_samples(samples, pl.DataFrame({"item_id": ["a", "b"]}), split="validation")
    with pytest.raises(ValueError, match="duplicate query identity"):
        validate_samples(
            samples.drop("sample_id"),
            pl.DataFrame({"item_id": ["a", "b"]}),
            split="validation",
        )


@pytest.mark.parametrize("items", [["a", "a"], ["a", None]])
def test_catalog_identity_must_be_unambiguous(samples, items):
    with pytest.raises(ValueError, match="unique and non-null"):
        validate_samples(samples, pl.DataFrame({"item_id": items}), split="test")


def test_unavailable_holdout_target_is_retained(samples):
    catalog = pl.DataFrame({"item_id": ["a"]})
    validate_samples(samples, catalog, split="validation")
    validate_samples(samples, catalog, split="test")
    assert samples["target"].to_list() == [["a"], ["b"]]
    with pytest.raises(ValueError, match="training targets are absent"):
        validate_samples(samples, catalog, split="train")
