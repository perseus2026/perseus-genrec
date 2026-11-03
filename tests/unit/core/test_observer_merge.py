"""Merging serialized worker statistics must match a single train-only fit."""

import polars as pl
import pytest

from genrecsys.core.encoders import registry


@pytest.mark.parametrize("kind", ["id", "numeric", "ple", "semantic_id"])
def test_serialized_observers_match_whole_dataset_fit(tmp_path, kind):
    encoder = registry[kind]
    values = pl.Series("value", [0.0, 0.0, 10.0, 20.0, 30.0, None])
    whole, merged = encoder.observer(), encoder.observer()
    whole.observe(values)
    for index, part in enumerate((values.head(2), values.slice(2), values.head(0))):
        worker = encoder.observer()
        worker.observe(part)
        path = tmp_path / str(index)
        worker.save(path)
        merged.merge(encoder.observer.load(path))
    empty = tmp_path / "empty"
    encoder.observer().save(empty)
    merged.merge(encoder.observer.load(empty))
    kwargs = {"max_bins": 4} if kind == "ple" else {}
    if kind == "semantic_id":
        kwargs = {
            "artifact": pl.DataFrame(
                {
                    "item_key": [0.0, 10.0, 20.0, 30.0],
                    "sid": [[2, 6], [3, 6], [4, 6], [5, 6]],
                }
            )
        }
    expected = encoder.preprocessor.fit(whole, **kwargs)
    actual = encoder.preprocessor.fit(merged, **kwargs)
    if kind == "ple":
        assert actual.thresholds == expected.thresholds
    elif kind == "numeric":
        assert actual.mean == pytest.approx(expected.mean)
        assert actual.scale == pytest.approx(expected.scale)
    elif kind == "id":
        assert actual.values.equals(expected.values)
    else:
        assert set(merged.values) == set(whole.values)
    assert actual.static_transform(values).equals(expected.static_transform(values))
