"""Manifest-free CLI inputs preserve resource bindings, joins and bundle identity."""

import json

import numpy as np
import polars as pl
import pytest
from cyclopts.exceptions import CoercionError

from genrecsys import core
from genrecsys.cli import prepare_shards, train
from genrecsys.cli.preparation_inputs import parse_mappings, parse_resources
from genrecsys.core.preparation_inputs import prepare_inputs
from item_representations.semantic_ids.mapping import fit_sid_mapper
from tests.integration.quickstart_data import make_data


@pytest.fixture
def inputs(tmp_path):
    workdir = tmp_path / "inputs"
    make_data(workdir)
    mapping = workdir / "resources/sid/item_to_sid.parquet"
    mapping.parent.mkdir(parents=True)
    pl.DataFrame(
        {"item_id": range(16), "sid": [[i + 2, 18, 19] for i in range(16)]}
    ).write_parquet(mapping)
    return workdir, core.Config.load_data(workdir / "config.yaml")


def test_defaults_load_only_required_resources_and_preserve_identity(inputs):
    workdir, config = inputs
    # Unused common inputs must not be read or carried into the checkpoint.
    (workdir / "resources/user_features.parquet").write_text("not a parquet")
    assert not (workdir / "manifest.json").exists()
    manifest = prepare_inputs(workdir, config)
    assert set(manifest.resources) == {"item_to_sid"}
    assert (
        manifest.resources["item_to_sid"]["path"] == "resources/sid/item_to_sid.parquet"
    )
    identity = manifest.identity(workdir)
    resource_bytes = (workdir / "resources/manifest.json").read_bytes()
    repeated = prepare_inputs(workdir, config)
    assert repeated.identity(workdir) == identity
    assert (workdir / "resources/manifest.json").read_bytes() == resource_bytes
    assert (
        core.Resources.load(workdir / "resources").resource("item_to_sid").height == 16
    )


@pytest.mark.parametrize("absolute", [False, True])
def test_explicit_named_resource_overrides_existing_mapping(inputs, absolute):
    workdir, config = inputs
    original = prepare_inputs(workdir, config)
    changed = pl.read_parquet(
        workdir / "resources/sid/item_to_sid.parquet"
    ).with_columns(sid=pl.col("sid").list.eval(pl.element() + 20))
    path = (
        workdir.parent / "another-mapping.parquet"
        if absolute
        else workdir / "another.parquet"
    )
    changed.write_parquet(path)
    manifest = prepare_inputs(
        workdir,
        config,
        resources={"item_to_sid": str(path) if absolute else "another.parquet"},
    )
    assert (
        core.Resources.load(workdir / "resources")
        .resource("item_to_sid")
        .equals(changed)
    )
    assert manifest.identity(workdir)["sha256"] != original.identity(workdir)["sha256"]
    assert (
        workdir / manifest.resources["item_to_sid"]["path"]
    ).resolve() == path.resolve()


def test_missing_default_does_not_fall_back_to_train_fit(inputs):
    workdir, config = inputs
    mapping = workdir / "resources/sid/item_to_sid.parquet"
    (workdir / "sid-fit").mkdir()
    mapping.rename(workdir / "sid-fit/item_to_sid.parquet")
    with pytest.raises(FileNotFoundError):
        prepare_inputs(workdir, config)
    assert not (workdir / "manifest.json").exists()
    assert not (workdir / "resources/manifest.json").exists()


def test_existing_manifest_changes_are_not_silently_accepted(inputs):
    workdir, config = inputs
    prepare_inputs(workdir, config)
    path = workdir / "basis/train/samples.pq"
    pl.read_parquet(path).head(1).write_parquet(path)
    with pytest.raises(ValueError, match="sample count differs"):
        prepare_inputs(workdir, config)


@pytest.mark.parametrize("filename", ["manifest.json", "resources/manifest.json"])
def test_malformed_manifest_does_not_trigger_defaults(inputs, filename):
    workdir, config = inputs
    (workdir / filename).write_text("broken json")
    with pytest.raises(json.JSONDecodeError):
        prepare_inputs(workdir, config)


def test_custom_names_require_explicit_paths(inputs):
    workdir, config = inputs
    config.encoders["item"].preprocessor["artifact"] = "custom_codes"
    with pytest.raises(ValueError, match="--resource custom_codes=PATH"):
        prepare_inputs(workdir, config)
    manifest = prepare_inputs(
        workdir,
        config,
        resources={"custom_codes": "resources/sid/item_to_sid.parquet"},
    )
    assert set(manifest.resources) == {"custom_codes"}


@pytest.mark.parametrize("sharded", [False, True])
def test_cli_preserves_feature_joins_and_metadata(inputs, monkeypatch, sharded):
    workdir, config = inputs
    # Resolve the model's item feature from another raw event attribute.
    config.events["view"].attributes = {
        "raw_id": core.config.Attribute(field="item_id")
    }
    config.save(workdir / "config.yaml")
    pl.DataFrame({"raw_id": range(16), "item_id": range(16)}).write_parquet(
        workdir / "resources/item_features.parquet"
    )
    metadata = {
        "dataset": "test",
        "source": {"revision": "1"},
        "protocol": {"name": "strict-prefix"},
    }
    spec = {
        "feature": "item",
        "resource": "item_features",
        "attributes": ["raw_id"],
        "value_column": "item_id",
    }
    app = prepare_shards.app if sharded else train.app
    args = [
        "plan" if sharded else "prepare-dataset",
        f"--workdir={workdir}",
        "--resource",
        "item_to_sid",
        "--resource",
        "item_features",
        "--event-feature-mapping",
        json.dumps(spec),
        "--metadata",
        json.dumps(metadata),
    ]
    if sharded:
        args += [f"--output-dir={workdir / 'plan'}", "--shards=2"]
    monkeypatch.setenv("INTERNAL_STORAGE_EVENT_HUB", str(workdir / "eventhub"))
    command, bound, _ = app.parse_args(args, exit_on_error=False)
    command(*bound.args, **bound.kwargs)
    manifest = core.DatasetManifest.load(workdir / "manifest.json")
    assert manifest.metadata == metadata
    assert manifest.event_feature_mappings[0].to_dict() == {**spec, "events": []}
    if sharded:
        state = json.loads((workdir / "plan/plan.json").read_text())
        assert state["dataset_identity"] == manifest.identity(workdir)
        assert (workdir / "plan/00/mappers/item").exists()
    else:
        prepared = core.checkpoint.Prepared.load(workdir / "checkpoint")
        assert prepared.dataset_identity == manifest.identity(workdir)
        dataset = core.Dataset.load(workdir / "dataset", config)
        assert len(dataset.train_samples) == 32


def test_deferred_sid_dependencies_use_full_catalog_defaults(inputs, monkeypatch):
    workdir, config = inputs
    full = workdir / "embeddings/inference"
    full.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"item_id": range(16)}).write_parquet(full / "items.pq")
    np.save(
        full / "embeddings.npy", np.load(workdir / "embeddings/train/embeddings.npy")
    )
    fit_sid_mapper(
        full / "items.pq",
        full / "embeddings.npy",
        workdir / "sid-fit",
        codebook_size=(4, 4),
        max_iter=2,
        n_init=1,
        mapper_only=True,
        device="cpu",
    )
    recipe = {
        "path": "sid-fit/checkpoint.pkl",
        "format": "sid_mapper",
        "items": "embedding_items",
        "embeddings": "item_embeddings",
    }
    resources = parse_resources(("item_to_sid=" + json.dumps(recipe),))
    manifest = prepare_inputs(workdir, config, resources=resources)
    assert set(manifest.resources) == {
        "item_to_sid",
        "embedding_items",
        "item_embeddings",
    }
    loaded = core.Resources.load(workdir / "resources")
    assert loaded.files["item_embeddings"].resolve() == full / "embeddings.npy"
    assert loaded.representations["item_to_sid"]["items"] == "embedding_items"
    monkeypatch.setenv("INTERNAL_STORAGE_EVENT_HUB", str(workdir / "eventhub"))
    train.prepare_dataset(workdir=workdir)
    prepared = core.checkpoint.Prepared.load(workdir / "checkpoint")
    mapping = prepared.encoder_to_preprocessor["item"].mapping
    assert set(mapping["item_id"]) == set(range(16))
    assert mapping["sid"].list.len().to_list() == [3] * 16
    assert prepared.files["item_embeddings"].is_file()
    np.save(full / "embeddings.npy", np.ones((2, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="not aligned"):
        prepare_inputs(workdir, config, resources=resources)


def test_explicit_empty_mapping_list_and_resource_errors():
    assert parse_mappings(()) == ()
    assert parse_mappings(None) is None
    for values in (("user_features", "user_features"), ("item_to_sid=",), ("unknown",)):
        with pytest.raises(ValueError):
            parse_resources(values)


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("metadata", ["[]", "invalid-json"])
def test_cli_rejects_invalid_metadata_before_writing_inputs(
    tmp_path, sharded, metadata
):
    app = prepare_shards.app if sharded else train.app
    command = "plan" if sharded else "prepare-dataset"
    args = [command, f"--workdir={tmp_path / 'inputs'}", "--metadata", metadata]
    if sharded:
        args.append(f"--output-dir={tmp_path / 'plan'}")
    with pytest.raises(CoercionError):
        app.parse_args(args, exit_on_error=False)
    assert not list(tmp_path.iterdir())


def test_recipe_without_frozen_resources_needs_no_mapping(inputs):
    workdir, config = inputs
    # Preparation must remain independent of model plugins and SID-specific recipes.
    config.encoders["item"] = core.config.Encoder(type="id")
    (workdir / "resources/sid/item_to_sid.parquet").unlink()
    manifest = prepare_inputs(workdir, config)
    assert manifest.resources == {}
    assert core.Resources.load(workdir / "resources").tables == {}
