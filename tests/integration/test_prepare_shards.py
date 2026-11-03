"""Sharded T-ECD Small preparation must match the native sequential lifecycle."""

from __future__ import annotations

import json
import shutil
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from data_sources import event_hub
from genrecsys import core
from genrecsys.cli import prepare_shards, train
from tests.integration.artifacts import copy_artifacts


def _fixture(root: Path, config_path: str, numeric_encoder=None) -> tuple[Path, Path]:
    basis = root / "basis"
    basis.mkdir()
    recipe = core.Config.load_data(config_path)
    recipe.inputs = {"lifelong": recipe.inputs["lifelong"]}
    recipe.inputs["lifelong"].features = ("item", "event", "brand_id")
    recipe.inputs["lifelong"].compression = None
    recipe.features = {
        name: recipe.features[name] for name in ("item", "event", "brand_id")
    }
    recipe.encoders = {
        feature.encoder: recipe.encoders[feature.encoder]
        for feature in recipe.features.values()
    }
    recipe.events = {
        name: event
        for name, event in recipe.events.items()
        if name in recipe.inputs["lifelong"].events
    }
    for event in recipe.events.values():
        event.attributes = {
            name: attribute
            for name, attribute in event.attributes.items()
            if name in {"item", "brand_id"}
        }
    event_encoder = recipe.inputs["lifelong"].event_encoder
    recipe.event_encoders = {event_encoder: recipe.event_encoders[event_encoder]}
    if numeric_encoder:
        recipe.encoders["amount"] = core.config.Encoder(
            type=numeric_encoder,
            preprocessor={"max_bins": 4} if numeric_encoder == "ple" else {},
        )
        recipe.features["amount"] = core.config.Feature(
            core.config.LocatedIn(event=True), encoder="amount"
        )
        for event in recipe.events.values():
            event.attributes["amount"] = core.config.Attribute(field="amount")
        for value in recipe.inputs.values():
            if value.type == "sequence":
                value.features = (*value.features, "amount")
    recipe.save(basis / "config.yaml")
    day = datetime(2026, 9, 1, tzinfo=UTC).replace(tzinfo=None)
    catalogs = pl.DataFrame(
        {"item_id": ["a", "b"], "brand_id": ["catalog-a", "catalog-b"]}
    )
    for fold, clients, days in (
        ("train", ["u1", "u1", "u2"], [2, 3, 3]),
        ("validation", ["u1", "u2", "empty"], [5, 5, 5]),
        ("test", ["u2", "holdout"], [7, 7]),
    ):
        path = basis / "basis" / fold
        path.mkdir(parents=True)
        pl.DataFrame(
            {
                "client_id": clients,
                "timestamp": [day + timedelta(days=value) for value in days],
                "history_timestamp": [
                    day + timedelta(days=value, hours=-12) for value in days
                ],
                "target": [["a"] for _ in clients],
            }
        ).write_parquet(path / "samples.pq")
        core.tasks.retrieval.Artifacts(catalogs).save(path / "artifacts")
    resources = basis / "resources"
    resources.mkdir()
    pl.DataFrame(
        {"item_id": ["a", "b"], "sid": [[2, 258, 514], [3, 259, 514]]}
    ).write_parquet(resources / "item_to_sid.pq")
    (resources / "manifest.json").write_text(
        json.dumps({"item_to_sid": "item_to_sid.pq"})
    )
    core.DatasetManifest.build(basis).save(basis / "manifest.json")
    eventhub = root / "events" / "eventhub"
    target = next(
        name
        for name, event in recipe.events.items()
        if "item" in event.attributes and event.priority > 0
    )
    auxiliary = next(
        (
            name
            for name, event in recipe.events.items()
            if "item" not in event.attributes
        ),
        next(name for name in recipe.events if name != target),
    )
    for name in (target, auxiliary):
        events = pl.DataFrame(
            [
                (
                    client,
                    day + timedelta(days=offset, minutes=int(name == auxiliary)),
                    "a" if offset % 2 else "b",
                    brand,
                )
                for client in ("u1", "u2", "holdout")
                for offset, brand in (
                    (0, "past-a"),
                    (1, "past-b"),
                    (2, "past-a"),
                    (3, "future-only"),
                    (5, "test-only"),
                    (8, "too-late"),
                )
            ],
            schema=["client_id", "timestamp", "item_id", "brand_id"],
            orient="row",
        )
        if numeric_encoder:
            events = events.with_columns(
                amount=pl.col("timestamp").dt.day().cast(pl.Float64)
            )
        for (partition,), frame in event_hub.distribute_to_partitions(events).group_by(
            "partition_id"
        ):
            path = (
                eventhub / name / "2026-09-01" / event_hub.partition_filename(partition)
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.drop("partition_id").write_parquet(path)
    archives = root / "archives"
    archives.mkdir()
    with tarfile.open(archives / "eventhub.tar", "w") as archive:
        archive.add(eventhub, arcname="eventhub")
    return basis, archives


@pytest.mark.parametrize(
    ("shards", "manifestless"), [(1, False), (3, True), (10, False)]
)
@pytest.mark.parametrize("numeric_encoder", [None, "numeric", "ple"])
@pytest.mark.parametrize(
    "config_path",
    [
        "tests/fixtures/tecd-small/retail-features.yaml",
        "configs/tecd-small/onerec-retail-cross-global.yaml",
    ],
)
def test_sharded_prepare_matches_sequential(
    tmp_path, monkeypatch, shards, manifestless, config_path, numeric_encoder
):
    basis, archives = _fixture(tmp_path, config_path, numeric_encoder)
    if manifestless:
        (basis / "resources/sid").mkdir()
        (basis / "resources/item_to_sid.pq").rename(
            basis / "resources/sid/item_to_sid.parquet"
        )
        (basis / "manifest.json").unlink()
        (basis / "resources/manifest.json").unlink()
    monkeypatch.setenv("INTERNAL_STORAGE_TECD", str(tmp_path / "events" / "eventhub"))
    monkeypatch.setenv(
        "INTERNAL_STORAGE_TECD_CROSS", str(tmp_path / "events" / "eventhub")
    )
    train.prepare_dataset(workdir=basis, collect_statistics=True)
    recipe = core.Config.load_data(basis / "config.yaml")
    sequential_checkpoint = core.checkpoint.Prepared.load(basis / "checkpoint")
    sequential = core.Dataset.load(basis / "dataset", recipe, include_test=True)

    inputs, plan = tmp_path / "inputs", tmp_path / "plan"
    multi_source = "cross-global" in config_path
    if multi_source:
        # MD jobs repack sources independently. The reader must merge their
        # own shard files without changing histories or train-only fitting.
        source_inputs = []
        for source in sorted((tmp_path / "events/eventhub").iterdir()):
            download = tmp_path / "source-downloads" / source.name
            download.mkdir(parents=True)
            source_archive = download / "eventhub.tar"
            with tarfile.open(source_archive, "w") as archive:
                archive.add(source, arcname=f"eventhub/{source.name}")
            output = inputs / source.name
            prepare_shards.partition_histories(
                history_dir=download,
                output_dir=output,
                shards=shards,
                delete_inputs=True,
            )
            assert not source_archive.exists()
            source_inputs.append(output)
    else:
        prepare_shards.partition_histories(
            history_dir=archives, output_dir=inputs, shards=shards
        )
        source_inputs = [inputs]
    assert (archives / "eventhub.tar").exists()
    if manifestless:
        # Independent CLI entrypoints must derive the same bindings and identity.
        (basis / "manifest.json").unlink()
        (basis / "resources/manifest.json").unlink()
    prepare_shards.plan(workdir=basis, output_dir=plan, shards=shards)
    # The repacked files are a disjoint, byte-identical cover of original files.
    names = []
    for shard in range(shards):
        download = tmp_path / "reader-downloads" / f"{shard:02d}"
        download.mkdir(parents=True)
        for source_index, source in enumerate(source_inputs):
            archive_path = source / f"{shard:02d}.tar"
            with tarfile.open(archive_path) as archive:
                for member in archive:
                    names.append(member.name)
                    original = tmp_path / "events" / member.name
                    assert archive.extractfile(member).read() == original.read_bytes()
                    partition = event_hub.partition_id_from_filename(
                        Path(member.name).name
                    )
                    assert (partition - 1) % shards == shard
            shutil.copyfile(archive_path, download / f"source-{source_index}.tar")
        prepare_shards.read(
            input_dir=plan / f"{shard:02d}",
            history_archive=download if multi_source else download / "source-0.tar",
            output_dir=tmp_path / "selected" / f"{shard:02d}",
            delete_inputs=True,
        )
        assert not list(download.glob("*.tar"))
        assert not (tmp_path / "selected" / f"{shard:02d}" / "input").exists()
    assert len(names) == len(set(names))
    assert set(names) == {
        str(path.relative_to(tmp_path / "events"))
        for path in (tmp_path / "events").rglob("*.pq")
    }
    fitted = tmp_path / "fitted"
    fit_inputs = tmp_path / "fit-inputs"
    fit_mask = "**/report.json;**/observers/**/*;**/counts/**/*;**/train-counts/**/*"
    copy_artifacts(tmp_path / "selected", fit_inputs, fit_mask)
    assert not list(fit_inputs.rglob("histories/*.pq"))
    prepare_shards.fit(
        workdir=basis,
        plan_dir=plan,
        shards_dir=fit_inputs,
        output_dir=fitted,
    )
    encode_fitted = tmp_path / "encode-fitted"
    encode_masks = {
        "/app/fitted/": "checkpoint/**/*;fit.json",
        "/app/input/": "histories/**/*;report.json",
    }
    copy_artifacts(fitted, encode_fitted, encode_masks["/app/fitted/"])
    for shard in range(shards):
        encode_input = tmp_path / "encode-inputs" / f"{shard:02d}"
        copy_artifacts(
            tmp_path / "selected" / f"{shard:02d}",
            encode_input,
            encode_masks["/app/input/"],
        )
        prepare_shards.encode(
            input_dir=encode_input,
            fitted_dir=encode_fitted,
            output_dir=tmp_path / "encoded" / f"{shard:02d}",
            delete_inputs=True,
        )
        assert not list((encode_input / "histories").glob("*.pq"))
    output = tmp_path / "prepared"
    assemble_inputs = tmp_path / "assemble-inputs"
    assemble_mask = "**/index.pq;**/report.json"
    copy_artifacts(tmp_path / "encoded", assemble_inputs, assemble_mask)
    assert not list(assemble_inputs.rglob("*.arrow"))
    prepare_shards.assemble(
        fitted_dir=fitted, encoded_dir=assemble_inputs, output_dir=output
    )
    # Payloads bypass assembly, mirroring independent uploads into final events/.
    assert not list((output / "dataset/events").glob("*.arrow"))
    for path in (tmp_path / "encoded").rglob("*.arrow"):
        shutil.copyfile(path, output / "dataset/events" / path.name)
    checkpoint = core.checkpoint.Prepared.load(output / "checkpoint")
    assert checkpoint.dataset_identity == sequential_checkpoint.dataset_identity
    checkpoint.save(tmp_path / "restored-checkpoint")
    assert (
        core.checkpoint.Prepared.load(tmp_path / "restored-checkpoint").dataset_identity
        == checkpoint.dataset_identity
    )
    actual = core.Dataset.load(output / "dataset", recipe, include_test=True)
    assert checkpoint.data_contract == sequential_checkpoint.data_contract
    assert checkpoint.encoder_to_preprocessor["brand_id"].values.equals(
        sequential_checkpoint.encoder_to_preprocessor["brand_id"].values
    )
    if numeric_encoder:
        actual_encoder = checkpoint.encoder_to_preprocessor["amount"]
        expected_encoder = sequential_checkpoint.encoder_to_preprocessor["amount"]
        if numeric_encoder == "ple":
            assert actual_encoder.thresholds == expected_encoder.thresholds
            assert actual_encoder.thresholds[-1] == 3.0
        else:
            assert actual_encoder.mean == pytest.approx(expected_encoder.mean)
            assert actual_encoder.scale == pytest.approx(expected_encoder.scale)
    assert (
        "future-only"
        not in checkpoint.encoder_to_preprocessor["brand_id"].values["value"].to_list()
    )
    assert (
        "test-only"
        not in checkpoint.encoder_to_preprocessor["brand_id"].values["value"].to_list()
    )
    for fold in ("train", "validation", "test"):
        assert getattr(actual, f"{fold}_samples").equals(
            getattr(sequential, f"{fold}_samples")
        )
        assert getattr(actual, f"{fold}_artifacts").items.equals(
            getattr(sequential, f"{fold}_artifacts").items
        )
    assert actual.task_preprocessor.items.equals(sequential.task_preprocessor.items)
    old_reader, new_reader = (
        sequential.event_store.open_reader(None),
        actual.event_store.open_reader(None),
    )
    for client in ("u1", "u2", "holdout"):
        for offset in (2, 4, 7):
            before = datetime(2026, 9, 1, tzinfo=UTC).replace(tzinfo=None) + timedelta(
                days=offset, hours=-12
            )
            assert new_reader.read(client, before=before).equals(
                old_reader.read(client, before=before)
            )
    # Statistics use compact worker counters, not a global events reread.
    for key in ("samples", "catalogs", "events", "history"):
        assert actual.statistics[key] == sequential.statistics[key]
    for key in ("rows", "clients"):
        assert (
            actual.statistics["event_store"][key]
            == sequential.statistics["event_store"][key]
        )
    for feature in actual.statistics["event_store"]["feature_counts"]:
        order = lambda values, feature=feature: sorted(
            values, key=lambda row: str(row[feature])
        )
        assert order(
            actual.statistics["event_store"]["feature_counts"][feature]
        ) == order(sequential.statistics["event_store"]["feature_counts"][feature])

    # A stale checkpoint or incomplete worker cannot publish a final index.
    config_file = encode_fitted / "checkpoint/config.yaml"
    config_file.write_text(config_file.read_text() + "\n")
    with pytest.raises(ValueError, match="checkpoint does not match"):
        prepare_shards.encode(
            input_dir=tmp_path / "encode-inputs/00",
            fitted_dir=encode_fitted,
            output_dir=tmp_path / "invalid-encode",
        )
    (assemble_inputs / "00/report.json").unlink()
    with pytest.raises(FileNotFoundError):
        prepare_shards.assemble(
            fitted_dir=fitted,
            encoded_dir=assemble_inputs,
            output_dir=tmp_path / "incomplete-output",
        )
    assert not (tmp_path / "incomplete-output").exists()


def test_worker_rejects_mixed_plan_inputs(tmp_path):
    basis, archives = _fixture(
        tmp_path, "tests/fixtures/tecd-small/retail-features.yaml"
    )
    prepare_shards.plan(workdir=basis, output_dir=tmp_path / "plan", shards=1)
    path = next((tmp_path / "plan/00/samples").glob("*.pq"))
    pl.read_parquet(path).head(0).write_parquet(path)
    with pytest.raises(ValueError, match="do not match plan"):
        prepare_shards.read(
            input_dir=tmp_path / "plan/00",
            history_archive=archives / "eventhub.tar",
            output_dir=tmp_path / "selected",
        )


@pytest.mark.parametrize("changed", ["samples", "mapping", "catalog", "protocol"])
def test_fit_rejects_another_dataset_version_before_reading_workers(tmp_path, changed):
    basis, _ = _fixture(tmp_path, "tests/fixtures/tecd-small/retail-features.yaml")
    plan = tmp_path / "plan"
    prepare_shards.plan(workdir=basis, output_dir=plan, shards=1)
    previous = core.DatasetManifest.load(basis / "manifest.json").identity(basis)
    other = tmp_path / "basis-v2"
    shutil.copytree(basis, other)
    assert (
        core.DatasetManifest.load(other / "manifest.json").identity(other) == previous
    )
    if changed == "samples":
        path = other / "basis/train/samples.pq"
        pl.read_parquet(path).with_columns(
            timestamp=pl.col("timestamp") + pl.duration(days=20),
            history_timestamp=pl.col("history_timestamp") + pl.duration(days=20),
        ).write_parquet(path)
    elif changed == "mapping":
        path = other / "resources/item_to_sid.pq"
        pl.read_parquet(path).reverse().write_parquet(path)
    elif changed == "catalog":
        path = other / "basis/train/artifacts/items.pq"
        pl.read_parquet(path).reverse().write_parquet(path)
    else:
        core.DatasetManifest.build(other, metadata={"protocol": {"name": "v2"}}).save(
            other / "manifest.json"
        )
    recipe = core.Config.load_data(other / "config.yaml")
    core.DatasetManifest.load(other / "manifest.json").validate(other, recipe)
    with pytest.raises(ValueError, match="bundle does not match preparation plan"):
        prepare_shards.fit(
            workdir=other,
            plan_dir=plan,
            shards_dir=tmp_path / "missing-workers",
            output_dir=tmp_path / "bad-fit",
        )
    assert not (tmp_path / "bad-fit").exists()
    assert (
        core.DatasetManifest.load(basis / "manifest.json").identity(basis) == previous
    )


def test_repack_failure_keeps_source_download(tmp_path, monkeypatch):
    _, archives = _fixture(tmp_path, "tests/fixtures/tecd-small/retail-features.yaml")

    def failed_write(*args, **kwargs):
        raise OSError("output disk full")

    monkeypatch.setattr(tarfile.TarFile, "addfile", failed_write)
    with pytest.raises(OSError, match="output disk full"):
        prepare_shards.partition_histories(
            history_dir=archives,
            output_dir=tmp_path / "inputs",
            delete_inputs=True,
        )
    assert (archives / "eventhub.tar").exists()


def test_reader_rejects_overlapping_sources(tmp_path):
    basis, archives = _fixture(
        tmp_path,
        "configs/tecd-small/onerec-retail-cross-global.yaml",
    )
    prepare_shards.plan(workdir=basis, output_dir=tmp_path / "plan", shards=1)
    shutil.copyfile(archives / "eventhub.tar", archives / "duplicate.tar")
    with pytest.raises(ValueError, match="overlapping history member"):
        prepare_shards.read(
            input_dir=tmp_path / "plan/00",
            history_archive=archives,
            output_dir=tmp_path / "selected",
        )
    assert (archives / "eventhub.tar").exists()
    assert (archives / "duplicate.tar").exists()


def test_reader_cleans_raw_partitions_only_after_success(tmp_path, monkeypatch):
    basis, archives = _fixture(
        tmp_path, "tests/fixtures/tecd-small/retail-features.yaml"
    )
    plan = tmp_path / "plan"
    prepare_shards.plan(workdir=basis, output_dir=plan, shards=1)
    partitions = json.loads((plan / "00/plan.json").read_text())["partitions"]
    original_read = event_hub.Reader.read

    def read_then_fail(reader, partition, samples):
        if partition == partitions[1]:
            raw = tmp_path / "selected/input/eventhub"
            assert not list(
                raw.glob(f"*/*/{event_hub.partition_filename(partitions[0])}")
            )
            assert list(raw.glob(f"*/*/{event_hub.partition_filename(partition)}"))
            raise RuntimeError("partition failed")
        return original_read(reader, partition, samples)

    monkeypatch.setattr(event_hub.Reader, "read", read_then_fail)
    with pytest.raises(RuntimeError, match="partition failed"):
        prepare_shards.read(
            input_dir=plan / "00",
            history_archive=archives / "eventhub.tar",
            output_dir=tmp_path / "selected",
        )
    assert (archives / "eventhub.tar").exists()
    assert list(
        (tmp_path / "selected/input/eventhub").glob(
            f"*/*/{event_hub.partition_filename(partitions[1])}"
        )
    )


def test_sharded_prepare_logs_failure_and_heartbeat(monkeypatch, caplog):
    from genrecsys.api import prepare_shards as operations

    monkeypatch.setattr(operations, "PROGRESS_INTERVAL_SECONDS", 0.01)
    with (
        caplog.at_level("INFO", logger=operations.__name__),
        pytest.raises(RuntimeError, match="failed"),
        operations._phase("test phase"),
    ):
        import time

        time.sleep(0.04)
        raise RuntimeError("failed")
    assert "Starting test phase" in caplog.text
    assert "test phase in progress elapsed=" in caplog.text
    assert "Stopped test phase elapsed=" in caplog.text
