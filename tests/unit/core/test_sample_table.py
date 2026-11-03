"""Memory mapping must survive spawn without serializing the sample payload."""

import pickle
from datetime import UTC, datetime

import polars as pl

from genrecsys.core.sample_table import MappedSamples


def test_mapped_samples_reuse_cache_and_pickle_only_path(tmp_path):
    path = tmp_path / "samples.pq"
    frame = pl.DataFrame(
        {
            "client_id": ["u1", "u2"],
            "timestamp": [datetime(2026, 1, 1, tzinfo=UTC)] * 2,
            "context": [{"x": "a"}, {"x": "b"}],
            "target": [[10, 20], []],
        }
    )
    frame.write_parquet(path)
    samples = MappedSamples.from_parquet(path)
    assert len(samples) == 2
    for index in range(2):
        assert samples.row(index) == frame.row(index, named=True)
    stat = samples.path.stat()
    assert MappedSamples.from_parquet(path).path.stat().st_mtime_ns == stat.st_mtime_ns
    restored = pickle.loads(pickle.dumps(samples))
    assert "_table" not in restored.__dict__
    assert restored.row(0) == samples.row(0)
    assert list(samples.__getstate__()) == ["path"]

    frame.head(1).write_parquet(path)
    replacement = MappedSamples.from_parquet(path)
    assert replacement.path != samples.path
    assert len(replacement) == 1


def test_empty_mapped_samples(tmp_path):
    path = tmp_path / "samples.pq"
    pl.DataFrame(schema={"target": pl.List(pl.Int64)}).write_parquet(path)
    assert len(MappedSamples.from_parquet(path)) == 0
