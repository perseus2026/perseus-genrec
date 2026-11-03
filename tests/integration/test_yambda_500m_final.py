"""500M like protocol and real sharded preparation preserve the native lifecycle."""

import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from genrecsys import core
from genrecsys.cli import prepare_shards, train
from model_datasets.yambda.basis import build_basis, download_source
from model_datasets.yambda.materialize import materialize_basis
from scripts.experiments import run_stage
from tests.integration.artifacts import copy_artifacts
from tests.integration.test_yambda_50m_final import _raw

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "tests/fixtures/yambda-500m/onerec-like-only.yaml"


@pytest.fixture
def basis(tmp_path):
    source = tmp_path / "source"
    _raw(source)
    (source / "flat/50m").rename(source / "flat/500m")
    # Only likes/audio are available: unrelated events and metadata must never be read.
    for path in source.glob("*_item_mapping.parquet"):
        path.unlink()
    for name in ("listens", "dislikes"):
        (source / f"flat/500m/{name}.parquet").unlink()
    path = source / "flat/500m/likes.parquet"
    raw = pl.read_parquet(path)
    # A change in organic flag must not create a second 500M like target.
    duplicate = raw.head(1).with_columns((1 - pl.col("is_organic")).alias("is_organic"))
    pl.concat([raw, duplicate]).write_parquet(path)
    output = tmp_path / "basis"
    build_basis(
        source_dir=source,
        output_dir=output,
        dataset_size="500m",
        like_only=True,
        validation_size=7,
    )
    return output


@pytest.fixture
def full_basis(tmp_path):
    source = tmp_path / "full-source"
    _raw(source)
    (source / "flat/50m").rename(source / "flat/500m")
    output = tmp_path / "full-basis"
    build_basis(
        source_dir=source, output_dir=output, dataset_size="500m", validation_size=7
    )
    return output


def test_full_500m_basis_matches_50m_protocol(full_basis, tmp_path):
    source = tmp_path / "reference-source"
    _raw(source)
    reference = tmp_path / "reference-basis"
    build_basis(source_dir=source, output_dir=reference, validation_size=7)
    report = json.loads((full_basis / "report.json").read_text())
    old = json.loads((reference / "report.json").read_text())
    assert set(report["event_rows"]) == {"like", "dislike", "listen"}
    for key in (
        "rows",
        "candidate_digest",
        "test_target_digest",
        "validation_subset_digest",
        "event_rows",
    ):
        assert report[key] == old[key]
    assert set(pl.read_parquet(full_basis / "item_features.pq").columns) == {
        "item_id",
        "album_id",
        "artist_id",
        "item_track_length_seconds",
    }
    assert pl.read_parquet(full_basis / "representations/all/items.pq")[
        "item_id"
    ].to_list() == ["0", "1", "2", "3", "4"]
    for event in ("like", "dislike", "listen"):
        for path in (full_basis / f"eventhub/yambda-500m-{event}").rglob("*.pq"):
            other = (
                reference
                / f"eventhub/yambda-50m-{event}"
                / path.relative_to(full_basis / f"eventhub/yambda-500m-{event}")
            )
            assert pl.read_parquet(path).equals(pl.read_parquet(other))


def test_download_only_likes_and_shared_audio(tmp_path, monkeypatch):
    import huggingface_hub

    requested = []

    def download(**kwargs):
        assert kwargs["revision"] == "fixture"
        requested.append(kwargs["filename"])
        path = kwargs["local_dir"] / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    download_source(
        output_dir=tmp_path, revision="fixture", dataset_size="500m", like_only=True
    )
    assert requested == ["flat/500m/likes.parquet", "embeddings.parquet"]


def test_basis_matches_50m_like_samples_with_separate_event_namespace(basis, tmp_path):
    source = tmp_path / "reference-source"
    _raw(source)
    reference = tmp_path / "reference"
    build_basis(source_dir=source, output_dir=reference, validation_size=7)
    report = json.loads((basis / "report.json").read_text())
    old = json.loads((reference / "report.json").read_text())
    assert report["dataset"] == "yambda-500m"
    assert set(report["event_rows"]) == {"like"}
    assert report["seen_filtering"] is False
    assert report["candidate_digest"] == old["candidate_digest"]
    assert report["test_target_digest"] == old["test_target_digest"]
    assert report["validation_subset_digest"] == old["validation_subset_digest"]
    for fold in ("train", "validation", "validation-full", "test"):
        assert pl.read_parquet(basis / f"basis/{fold}/samples.pq").equals(
            pl.read_parquet(reference / f"basis/{fold}/samples.pq")
        )
    assert pl.read_parquet(basis / "item_features.pq").columns == ["item_id"]
    assert sorted(path.name for path in (basis / "eventhub").iterdir()) == [
        "yambda-500m-like"
    ]
    for path in (basis / "eventhub/yambda-500m-like").rglob("*.pq"):
        previous = (
            reference
            / "eventhub/yambda-50m-like"
            / path.relative_to(basis / "eventhub/yambda-500m-like")
        )
        assert pl.read_parquet(path).equals(
            pl.read_parquet(previous).drop("is_organic")
        )
    assert set(pl.read_parquet(basis / "representations/all/items.pq")["item_id"]) == {
        "0",
        "1",
    }
    assert np.load(basis / "representations/all/embeddings.npy").shape == (2, 128)


@pytest.mark.parametrize("prepared_source", ["like-features", "all-features"])
def test_shared_prepare_matches_native_lifecycle(
    full_basis, tmp_path, monkeypatch, prepared_source
):
    basis = full_basis
    public = tmp_path / "public checkout"
    for name in ("src", "scripts", "configs/yambda-500m"):
        shutil.copytree(
            ROOT / name, public / name, ignore=shutil.ignore_patterns("__pycache__")
        )
    (public / ".venv").symlink_to(ROOT / ".venv", target_is_directory=True)
    data = public / "work/data"
    shutil.copytree(basis, data / "basis")
    mapping = data / "sids/item_to_sid.parquet"
    mapping.parent.mkdir()
    pl.DataFrame(
        {
            "item_id": [str(i) for i in range(5)],
            "sid": [[2 + i, 130 + i, 258 + i, 386] for i in range(5)],
        }
    ).write_parquet(mapping)
    environment = {
        **os.environ,
        "PYTHONPATH": "src",
        "OMP_NUM_THREADS": "1",
        "ACCELERATE_USE_CPU": "true",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }

    def run(*args):
        result = subprocess.run(
            args,
            cwd=public,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stderr[-8000:]

    run_stage(
        "yambda-500m",
        "prepare_" + prepared_source.replace("-", "_"),
        data_dir=data,
        run_dir=public / "work/run",
    )
    local = public / "work/run/prepared" / prepared_source
    monkeypatch.setenv("INTERNAL_STORAGE_YAMBDA", str(data / "basis/eventhub"))
    reference_output = tmp_path / "reference-prepared"
    materialize_basis(
        config=ROOT / f"configs/yambda-500m/onerec-{prepared_source}.yaml",
        basis=data / "basis",
        sid_mapping=mapping,
        workdir=reference_output,
    )
    train.prepare_dataset(workdir=reference_output)
    for folder in ("dataset", "checkpoint"):
        files = {
            p.relative_to(local) for p in (local / folder).rglob("*") if p.is_file()
        }
        assert files == {
            p.relative_to(reference_output)
            for p in (reference_output / folder).rglob("*")
            if p.is_file()
        }
        for path in files:
            if path == Path("checkpoint/config.yaml"):
                continue  # Only the tracker differs.
            if (
                path.parts[:2] == ("checkpoint", "event_feature_mappers")
                and path.suffix == ".pq"
            ):
                # Lookup row order is irrelevant; fitted encoders and encoded
                # EventStore bytes below must still match exactly.
                assert (
                    pl.read_parquet(local / path)
                    .sort("item")
                    .equals(pl.read_parquet(reference_output / path).sort("item"))
                ), str(path)
                continue
            assert (local / path).read_bytes() == (
                reference_output / path
            ).read_bytes(), str(path)
    assert (data / "basis/eventhub").is_dir() and mapping.is_file()
    # Exercise the same train/predict/evaluate CLI with a tiny CPU model.
    smoke = core.Config.load(
        public / f"configs/yambda-500m/onerec-{prepared_source}.yaml"
    )
    smoke.encoders["is_organic"].embedder["dim"] = 32
    smoke.model.params.update(dim=32, dim_feedforward=64)
    smoke.model.params["encoder"]["num_layers"] = 1
    smoke.model.params["decoder"]["num_layers"] = 1
    smoke.training.num_epochs = 1
    smoke.training.mixed_precision = smoke.inference.mixed_precision = "no"
    for loader in (
        smoke.training.dataloader,
        smoke.training.validation_dataloader,
        smoke.inference.dataloader,
    ):
        loader.batch_size, loader.num_workers, loader.pin_memory = 2, 0, False
    smoke.inference.predict_kwargs = {"k": 2, "beam_size": 2}
    smoke.task.metrics = {
        name.replace("@100", "@2"): metric
        for name, metric in smoke.task.metrics.items()
    }
    for metric in smoke.task.metrics.values():
        metric.params["k"] = 2
    smoke.training.early_stopping.metric = "Recall@2"
    smoke.save(public / "work/smoke.yaml")
    (data / "basis/eventhub").rename(data / "basis/unused-eventhub")
    identity = (local / "checkpoint/dataset_identity.json").read_bytes()
    run(
        "bash",
        "scripts/train.sh",
        f"work/run/prepared/{prepared_source}",
        "work/smoke.yaml",
        "1",
    )
    run("bash", "scripts/predict.sh", f"work/run/prepared/{prepared_source}", "1")
    run("bash", "scripts/evaluate.sh", f"work/run/prepared/{prepared_source}")
    assert (local / "checkpoint/dataset_identity.json").read_bytes() == identity
    metrics = json.loads((local / "metrics.json").read_text())
    assert metrics["population_verified"]
    assert metrics["samples"] == len(pl.read_parquet(local / "dataset/test/samples.pq"))


@pytest.mark.parametrize("shards", [1, 3, 10])
def test_sharded_like_matches_sequential_after_relocating_artifacts(
    basis, tmp_path, monkeypatch, shards
):
    mapping = tmp_path / "item_to_sid.parquet"
    pl.DataFrame(
        {"item_id": ["0", "1"], "sid": [[2, 130, 258, 386], [3, 131, 259, 386]]}
    ).write_parquet(mapping)
    materialized = tmp_path / "materialized"
    materialize_basis(
        config=RECIPE,
        basis=basis,
        sid_mapping=mapping,
        workdir=materialized,
    )
    monkeypatch.setenv("INTERNAL_STORAGE_YAMBDA", str(basis / "eventhub"))
    train.prepare_dataset(workdir=materialized, collect_statistics=True)
    recipe = core.Config.load(materialized / "config.yaml")
    reference = core.Dataset.load(materialized / "dataset", recipe, include_test=True)
    archives = tmp_path / "archives"
    archives.mkdir()
    with tarfile.open(archives / "eventhub.tar", "w") as archive:
        archive.add(basis / "eventhub", arcname="eventhub")
    partitioned, plan, selected = (
        tmp_path / "partitioned",
        tmp_path / "plan",
        tmp_path / "selected",
    )
    prepare_shards.partition_histories(
        history_dir=archives, output_dir=partitioned, shards=shards
    )
    prepare_shards.plan(workdir=materialized, output_dir=plan, shards=shards)
    read_mask = "**/*.pq;report.json"
    uploaded = tmp_path / "uploaded"
    for shard in range(shards):
        worker_output = selected / f"{shard:02d}"
        prepare_shards.read(
            input_dir=plan / f"{shard:02d}",
            history_archive=partitioned / f"{shard:02d}.tar",
            output_dir=worker_output,
        )
        assert not (worker_output / "observers").exists()
        copy_artifacts(worker_output, uploaded / f"{shard:02d}", read_mask)
    fit_inputs = tmp_path / "fit-inputs"
    fit_mask = "**/report.json;**/observers/**/*;**/counts/**/*;**/train-counts/**/*"
    copy_artifacts(uploaded, fit_inputs, fit_mask)
    assert not list(fit_inputs.rglob("histories/*.pq"))
    fitted = tmp_path / "fitted"
    prepare_shards.fit(
        workdir=materialized, plan_dir=plan, shards_dir=fit_inputs, output_dir=fitted
    )
    encode_fitted = tmp_path / "encode-fitted"
    copy_artifacts(fitted, encode_fitted, "checkpoint/**/*;fit.json")
    for shard in range(shards):
        encode_input = tmp_path / "encode-inputs" / f"{shard:02d}"
        encode_mask = "**/*.pq;report.json"
        copy_artifacts(uploaded / f"{shard:02d}", encode_input, encode_mask)
        prepare_shards.encode(
            input_dir=encode_input,
            fitted_dir=encode_fitted,
            output_dir=tmp_path / "encoded" / f"{shard:02d}",
        )
    assembly_inputs = tmp_path / "assembly-inputs"
    copy_artifacts(tmp_path / "encoded", assembly_inputs, "**/index.pq;**/report.json")
    assert not list(assembly_inputs.rglob("*.arrow"))
    prepared = tmp_path / "prepared"
    prepare_shards.assemble(
        fitted_dir=fitted, encoded_dir=assembly_inputs, output_dir=prepared
    )
    for path in (tmp_path / "encoded").rglob("*.arrow"):
        shutil.copyfile(path, prepared / "dataset/events" / path.name)
    actual = core.Dataset.load(prepared / "dataset", recipe, include_test=True)
    for fold in ("train", "validation", "test"):
        assert getattr(actual, f"{fold}_samples").equals(
            getattr(reference, f"{fold}_samples")
        )
        assert getattr(actual, f"{fold}_artifacts").items.equals(
            getattr(reference, f"{fold}_artifacts").items
        )
    previous, current = (
        reference.event_store.open_reader(None),
        actual.event_store.open_reader(None),
    )
    for client in ("0", "1", "2", "3"):
        assert previous.read(client).equals(current.read(client))
    checkpoint = core.checkpoint.Prepared.load(prepared / "checkpoint")
    assert checkpoint.encoder_to_preprocessor["item"].code_length == 4
    assert actual.statistics["samples"] == reference.statistics["samples"]
    assert (
        actual.statistics["event_store"]["feature_counts"]
        == reference.statistics["event_store"]["feature_counts"]
    )
    assert (archives / "eventhub.tar").exists()
