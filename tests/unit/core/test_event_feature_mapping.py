"""Exact item metadata lookups and their model-neutral manifest contract."""

import json
from datetime import UTC, datetime

import polars as pl
import pytest

from genrecsys.core.config import Config, Encoder, Feature, LocatedIn
from genrecsys.core.dataset_manifest import DatasetManifest
from genrecsys.core.event_feature_mapping import EventFeatureMapper, Spec
from genrecsys.core.resources import Resources


@pytest.mark.parametrize("value_column", ["topic", "topic_value"])
def test_mapping_retains_output_and_preserves_rows(value_column):
    mapper = EventFeatureMapper(
        feature="topic",
        attributes=("item",),
        value_column=value_column,
        events=("click",),
        mapping=pl.DataFrame({"item": ["entity"], value_column: ["topic-a"]}),
    )
    events = pl.DataFrame(
        {
            "event": ["click", "click", "other"],
            "item": ["entity", "unknown", "entity"],
        }
    )
    result = mapper.transform(events)
    assert result.select(events.columns).equals(events)
    assert result["topic"].to_list() == ["topic-a", None, None]
    assert set(result.columns) == {*events.columns, "topic"}


def test_mapping_does_not_replace_an_unrelated_raw_value_column():
    mapper = EventFeatureMapper(
        feature="topic",
        attributes=("item",),
        value_column="topic_value",
        events=("click",),
        mapping=pl.DataFrame({"item": ["entity"], "topic_value": ["topic-a"]}),
    )
    result = mapper.transform(
        pl.DataFrame({"event": ["click"], "item": ["entity"], "topic_value": ["raw"]})
    )
    assert result["topic"].to_list() == ["topic-a"]
    assert result["topic_value"].to_list() == ["raw"]


def test_mapping_preserves_raw_feature_values_on_unrelated_events():
    mapper = EventFeatureMapper(
        feature="price",
        attributes=("item",),
        value_column="price",
        events=("target",),
        mapping=pl.DataFrame({"item": ["mapped"], "price": [10.0]}),
    )
    events = pl.DataFrame(
        {
            "event": ["target", "payment"],
            "item": ["mapped", None],
            "price": [None, 99.0],
        }
    )

    result = mapper.transform(events)

    assert result["price"].to_list() == [10.0, 99.0]
    assert result.columns == events.columns


def test_mapping_rejects_ambiguous_item_metadata():
    config = Config.load_data("tests/fixtures/framework/sequence.yaml")
    config.encoders["topic"] = Encoder(type="id")
    config.features["topic"] = Feature(LocatedIn(event=True), encoder="topic")
    resources = Resources(
        {"topics": pl.DataFrame({"item": ["entity", "entity"], "topic": ["a", "b"]})}
    )
    with pytest.raises(ValueError, match="ambiguous keys"):
        EventFeatureMapper.from_spec(
            Spec("topic", "topics", ("item",), "topic"), config, resources
        )


def test_metadata_mapping_accepts_frozen_vocabulary_without_catalog_mapping_column(
    tmp_path,
):
    config = Config.load_data("tests/fixtures/framework/sequence.yaml")
    config.encoders["topic"] = Encoder(
        type="id", preprocessor={"artifact": "topic_vocab"}
    )
    config.features["topic"] = Feature(LocatedIn(event=True), encoder="topic")
    for split in ("train", "validation", "test"):
        root = tmp_path / "basis" / split
        (root / "artifacts").mkdir(parents=True)
        pl.DataFrame(
            {
                "client_id": ["reader"],
                "timestamp": [datetime(2026, 8, 23, tzinfo=UTC)],
                "target": [["entity"]],
            }
        ).write_parquet(root / "samples.pq")
        pl.DataFrame({"item_id": ["entity"]}).write_parquet(root / "artifacts/items.pq")
    resources = tmp_path / "resources"
    resources.mkdir()
    pl.DataFrame({"item_id": ["entity"], "sid": [[2, 3]]}).write_parquet(
        resources / "sids.pq"
    )
    pl.DataFrame({"item": ["entity"], "topic": ["topic-a"]}).write_parquet(
        resources / "topics.pq"
    )
    pl.DataFrame({"value": ["topic-a"], "index": [1]}).write_parquet(
        resources / "vocab.pq"
    )
    (resources / "manifest.json").write_text(
        json.dumps(
            {"item_to_sid": "sids.pq", "topics": "topics.pq", "topic_vocab": "vocab.pq"}
        )
    )
    manifest = DatasetManifest.build(
        tmp_path, event_feature_mappings=(Spec("topic", "topics", ("item",), "topic"),)
    )
    manifest.validate(tmp_path, config)
    config.features["topic"].located_in.artifacts = True
    with pytest.raises(ValueError, match="retrieval artifacts miss columns.*topic"):
        manifest.validate(tmp_path, config)
