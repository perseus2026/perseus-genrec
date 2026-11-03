"""Independent CPU history jobs followed by one train-only vocabulary fit.

Each client belongs to one worker. Encoded Arrow files are uploaded directly to
the final EventStore; assembly combines only small indexes, never event payloads.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tarfile
import time
from contextlib import ExitStack
from pathlib import Path

import polars as pl
import pyarrow as pa

from data_sources import event_hub
from genrecsys import core
from genrecsys.core import preparation
from genrecsys.core.data_statistics import preparation_statistics
from genrecsys.core.preparation_inputs import prepare_inputs
from genrecsys.utils.files import discard_files
from genrecsys.utils.progress import log_phase, logged_stage

logger = logging.getLogger(__name__)
PROGRESS_INTERVAL_SECONDS = 30


def _phase(name: str):
    return log_phase(logger, name, interval_seconds=PROGRESS_INTERVAL_SECONDS)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _discard_files(paths: list[Path]) -> None:
    discard_files(paths, logger=logger)


def _digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


@logged_stage
def partition_histories(
    *,
    history_dir: Path,
    output_dir: Path,
    shards: int = 10,
    delete_inputs: bool = False,
) -> None:
    """Repack tars; delete_inputs is only for disposable job-local downloads."""
    if not 1 <= shards <= event_hub.PARTITIONS_NUM:
        raise ValueError("shards must be between 1 and 100")
    archives = sorted(history_dir.glob("*.tar"))
    if not archives:
        raise FileNotFoundError(f"no history archives under {history_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    with (
        _phase(f"partition history archives into {shards} workers"),
        ExitStack() as stack,
    ):
        writers = [
            stack.enter_context(tarfile.open(output_dir / f"{shard:02d}.tar", "w"))
            for shard in range(shards)
        ]
        seen = set()
        total_bytes = 0
        started = time.monotonic()
        for archive_path in archives:
            with (
                _phase(f"repack {archive_path.name}"),
                tarfile.open(archive_path, "r|*") as archive,
            ):
                for member in archive:
                    if member.isdir():
                        continue
                    path = Path(member.name)
                    if not member.isfile() or path.is_absolute() or ".." in path.parts:
                        raise ValueError(
                            f"unexpected history archive member: {member.name}"
                        )
                    if path.suffix != ".pq":
                        continue
                    partition = event_hub.partition_id_from_filename(path.name)
                    if (
                        not 1 <= partition <= event_hub.PARTITIONS_NUM
                        or member.name in seen
                    ):
                        raise ValueError(
                            f"invalid or overlapping history member: {member.name}"
                        )
                    seen.add(member.name)
                    with archive.extractfile(member) as stream:
                        writers[(partition - 1) % shards].addfile(member, stream)
                    total_bytes += member.size
            elapsed = time.monotonic() - started
            logger.info(
                "repacked files=%s bytes=%s throughput=%.1fMiB/s elapsed=%.1fs",
                len(seen),
                total_bytes,
                total_bytes / 2**20 / elapsed,
                elapsed,
            )
            if delete_inputs:
                for writer in writers:
                    writer.fileobj.flush()
                _discard_files([archive_path])


@logged_stage
def plan(
    *,
    workdir: Path,
    output_dir: Path,
    config: Path | None = None,
    shards: int = 10,
    resources: dict[str, str | dict] | None = None,
    event_feature_mappings: tuple[core.event_feature_mapping.Spec, ...] | None = None,
    metadata: dict | None = None,
) -> None:
    """Partition all split cutoffs, retaining all samples of each client together."""
    if not 1 <= shards <= event_hub.PARTITIONS_NUM:
        raise ValueError("shards must be between 1 and 100")
    with _phase(f"plan {shards} history workers"):
        recipe = core.Config.load_data(config or workdir / "config.yaml")
        manifest = prepare_inputs(
            workdir,
            recipe,
            resources=resources,
            event_feature_mappings=event_feature_mappings,
            metadata=metadata,
        )
        observers = preparation.create_observers(recipe)
        basis = preparation.load_basis(workdir, recipe)
        preparation.fit_task_preprocessor(
            basis["train_samples"], basis["train_artifacts"], observers, recipe
        )
        resources = core.Resources.load(workdir / "resources")
        preparation.resolve_representations(resources)
        mappers = core.event_feature_mapping.build_mappers(
            manifest.event_feature_mappings, recipe, resources
        )
        output_dir.mkdir(parents=True, exist_ok=False)
        recipe.save(output_dir / "config.yaml")
        identity = manifest.identity(workdir)
        workers = [
            {"shard": shard, "partitions": [], "bundle_sha256": identity["sha256"]}
            for shard in range(shards)
        ]
        for worker in workers:
            root = output_dir / f"{worker['shard']:02d}"
            (root / "samples").mkdir(parents=True)
            recipe.save(root / "config.yaml")
            for feature, mapper in mappers.items():
                mapper.save(root / "mappers" / feature)
        for _, _, partition, samples in preparation.distribute_samples(
            basis["train_samples"], basis["validation_samples"], basis["test_samples"]
        ):
            shard = (partition - 1) % shards
            samples.write_parquet(
                output_dir
                / f"{shard:02d}"
                / "samples"
                / event_hub.partition_filename(partition)
            )
            workers[shard]["partitions"].append(partition)
        for worker in workers:
            root = output_dir / f"{worker['shard']:02d}"
            worker["input_sha256"] = _digest(list(root.rglob("*.*")))
            _write_json(root / "plan.json", worker)
        _write_json(
            output_dir / "plan.json", {"workers": workers, "dataset_identity": identity}
        )


@logged_stage
def read(
    *,
    input_dir: Path,
    history_archive: Path,
    output_dir: Path,
    delete_inputs: bool = False,
) -> None:
    """Select histories from one tar or a directory of disjoint source tars."""
    worker = _json(input_dir / "plan.json")
    recipe = core.Config.load_data(input_dir / "config.yaml")
    observers = preparation.create_observers(recipe)
    # Checking the plan's exact input files catches mixed basis/config uploads.
    if (
        _digest([path for path in input_dir.rglob("*.*") if path.name != "plan.json"])
        != worker["input_sha256"]
    ):
        raise ValueError("history worker inputs do not match plan")
    archives = (
        sorted(history_archive.glob("*.tar"))
        if history_archive.is_dir()
        else [history_archive]
    )
    if not archives:
        raise FileNotFoundError(f"no history archives under {history_archive}")
    output_dir.mkdir(parents=True, exist_ok=False)
    histories = output_dir / "input"
    histories.mkdir()
    for archive_path in archives:
        with (
            _phase(
                f"extract history shard {worker['shard']} source {archive_path.name}"
            ),
            tarfile.open(archive_path, "r|*") as archive,
        ):
            for member in archive:
                if member.isfile() and (histories / member.name).exists():
                    raise ValueError(f"overlapping history member: {member.name}")
                archive.extract(member, histories, filter="data")
        if delete_inputs:
            _discard_files([archive_path])
    sources = {event.source: histories / "eventhub" for event in recipe.events.values()}
    reader = event_hub.Reader(recipe, source_to_path=sources)
    mappers = {
        path.name: core.event_feature_mapping.EventFeatureMapper.load(path)
        for path in (input_dir / "mappers").glob("*")
    }
    counts, train_counts = {}, {}
    rows = clients = 0
    started = time.monotonic()
    for index, partition in enumerate(worker["partitions"], 1):
        with _phase(
            f"history shard {worker['shard']} partition {partition} ({index}/{len(worker['partitions'])})"
        ):
            samples = pl.read_parquet(
                input_dir / "samples" / event_hub.partition_filename(partition)
            )
            events = reader.read(partition, samples)
            if events is not None:
                events = core.feature_engineering.prepare_events(
                    events, recipe, event_feature_mappers=mappers
                )
                rows += len(events)
                clients += events["client_id"].n_unique()
                for feature in recipe.feature_to_encoder(event=True):
                    if (
                        feature != recipe.task.objective.target
                        and recipe.encoders[recipe.features[feature].encoder].type
                        == "id"
                    ):
                        counts.setdefault(feature, []).append(
                            events.group_by(feature).agg(
                                rows=pl.len(), clients=pl.col("client_id").n_unique()
                            )
                        )
                preparation.observe_events(
                    events, samples, observers, recipe, feature_count_parts=train_counts
                )
                destination = (
                    output_dir / "histories" / event_hub.partition_filename(partition)
                )
                destination.parent.mkdir(exist_ok=True)
                events.write_parquet(destination)
            # The reader has materialized this partition. Discard only the
            # temporary files extracted by this function.
            _discard_files(
                list(
                    histories.glob(
                        f"eventhub/*/*/{event_hub.partition_filename(partition)}"
                    )
                )
            )
        elapsed = time.monotonic() - started
        logger.info(
            "history shard=%s completed=%s/%s rows=%s clients=%s elapsed=%.1fs eta=%.1fs",
            worker["shard"],
            index,
            len(worker["partitions"]),
            rows,
            clients,
            elapsed,
            elapsed / index * (len(worker["partitions"]) - index),
        )
    for name, observer in observers.items():
        if not preparation.has_frozen_mapping(recipe.encoders[name]):
            observer.save(output_dir / "observers" / name)
    for directory, parts_by_feature in (
        ("counts", counts),
        ("train-counts", train_counts),
    ):
        for feature, parts in parts_by_feature.items():
            root = output_dir / directory
            root.mkdir(exist_ok=True)
            group = feature if directory == "counts" else "value"
            pl.concat(parts).group_by(group).agg(pl.exclude(group).sum()).write_parquet(
                root / f"{feature}.pq"
            )
    _write_json(
        output_dir / "report.json", {**worker, "rows": rows, "clients": clients}
    )
    shutil.rmtree(histories)


@logged_stage
def fit(*, workdir: Path, plan_dir: Path, shards_dir: Path, output_dir: Path) -> None:
    """Fit one shared vocabulary from worker counts plus global train basis."""
    with _phase("fit shared train-only preprocessors"):
        recipe = core.Config.load_data(plan_dir / "config.yaml")
        plan = _json(plan_dir / "plan.json")
        manifest = core.DatasetManifest.load(workdir / "manifest.json")
        identity = manifest.identity(workdir)
        if identity["sha256"] != plan["dataset_identity"]["sha256"]:
            raise ValueError("dataset bundle does not match preparation plan")
        observers = preparation.create_observers(recipe)
        basis = preparation.load_basis(workdir, recipe)
        resources = core.Resources.load(workdir / "resources")
        preparation.resolve_representations(resources)
        task = preparation.fit_task_preprocessor(
            basis["train_samples"], basis["train_artifacts"], observers, recipe
        )
        expected = plan["workers"]
        counts, train_counts = {}, {}
        rows = clients = 0
        reports = []
        for worker in expected:
            root = shards_dir / f"{worker['shard']:02d}"
            report = _json(root / "report.json")
            if any(report[key] != worker[key] for key in worker):
                raise ValueError(f"history shard {worker['shard']} does not match plan")
            reports.append(report)
            rows += report["rows"]
            clients += report["clients"]
            for name, observer in observers.items():
                if not preparation.has_frozen_mapping(recipe.encoders[name]):
                    observer.merge(type(observer).load(root / "observers" / name))
            for directory, result in (
                ("counts", counts),
                ("train-counts", train_counts),
            ):
                for path in (root / directory).glob("*.pq"):
                    result.setdefault(path.stem, []).append(pl.read_parquet(path))
        if not rows:
            raise ValueError("EventHub contains no histories for basis samples")
        preparation.observe_samples(basis["train_samples"], observers, recipe)
        preprocessors = preparation.fit_encoder_preprocessors(
            observers, resources, recipe
        )
        manifest = core.DatasetManifest.load(workdir / "manifest.json")
        mappers = core.event_feature_mapping.build_mappers(
            manifest.event_feature_mappings, recipe, resources
        )
        checkpoint = core.checkpoint.Prepared(
            recipe,
            preprocessors,
            core.DataContract.from_config(recipe),
            mappers,
            files=resources.files,
            dataset_identity=identity,
        )
        transformed = preparation.static_transform_basis(
            basis["train_samples"],
            basis["train_artifacts"],
            basis["validation_samples"],
            basis["validation_artifacts"],
            basis["test_samples"],
            basis["test_artifacts"],
            preprocessors,
            task,
            recipe,
        )
        (
            train,
            train_artifacts,
            validation,
            validation_artifacts,
            test,
            test_artifacts,
        ) = transformed
        statistics = preparation_statistics(
            recipe,
            preprocessors,
            train,
            validation,
            test,
            train_artifacts,
            validation_artifacts,
            test_artifacts,
            rows,
            clients,
            counts,
            train_counts,
        )
        output_dir.mkdir(parents=True, exist_ok=False)
        checkpoint.save(output_dir / "checkpoint")
        core.Dataset(
            train,
            train_artifacts,
            validation,
            validation_artifacts,
            None,
            task,
            test,
            test_artifacts,
            statistics,
        ).save(output_dir / "dataset")
        recipe.save(output_dir / "config.yaml")
        for name in ("manifest.json", "provenance.json"):
            path = workdir / name
            if path.exists():
                shutil.copyfile(path, output_dir / name)
        fit_id = _digest(list((output_dir / "checkpoint").rglob("*.*")))
        _write_json(output_dir / "fit.json", {"workers": reports, "fit_sha256": fit_id})
        logger.info(
            "shared preprocessors fitted shards=%s rows=%s clients=%s",
            len(expected),
            rows,
            clients,
        )


@logged_stage
def encode(
    *,
    input_dir: Path,
    fitted_dir: Path,
    output_dir: Path,
    delete_inputs: bool = False,
    discard_unused_attributes: bool = False,
) -> None:
    """Encode and write one Arrow file with the globally fitted preprocessors."""
    report = _json(input_dir / "report.json")
    fitted = _json(fitted_dir / "fit.json")
    if report not in fitted["workers"]:
        raise ValueError("encoding worker does not match fitted history plan")
    if _digest(list((fitted_dir / "checkpoint").rglob("*.*"))) != fitted["fit_sha256"]:
        raise ValueError("checkpoint does not match fitted preprocessing state")
    checkpoint = core.checkpoint.Prepared.load(fitted_dir / "checkpoint")
    output_dir.mkdir(parents=True, exist_ok=False)
    if report["rows"]:
        store = core.EventStore(output_dir, backend="disk")
        encoded_rows = 0
        with store.open_writer() as writer:
            for path in sorted((input_dir / "histories").glob("*.pq")):
                with _phase(f"encode shard {report['shard']} partition {path.name}"):
                    events = core.feature_engineering.encode_events(
                        pl.read_parquet(path),
                        checkpoint.encoder_to_preprocessor,
                        checkpoint.config,
                    )
                    if discard_unused_attributes:
                        events = events.select(
                            "client_id",
                            "timestamp",
                            "position",
                            "_event",
                            *checkpoint.config.feature_to_encoder(event=True),
                        )
                    writer.write(events)
                    encoded_rows += len(events)
                if delete_inputs:
                    _discard_files([path])
        if encoded_rows != report["rows"]:
            raise ValueError("encoded shard row count differs from selected histories")
        file = f"data-{report['shard']:02d}.arrow"
        (output_dir / "data.arrow").rename(output_dir / file)
        index = (
            pl.read_parquet(output_dir / "index.pq")
            .with_row_index("batch_index")
            .with_columns(file=pl.lit(file))
        )
        if len(index) != report["clients"] or index["client_id"].n_unique() != len(
            index
        ):
            raise ValueError("encoded shard contains missing or repeated clients")
        index.write_parquet(output_dir / "index.pq")
    else:
        # Empty workers still publish a file so the output loader's file mask
        # remains valid. No client index points at this empty Arrow file.
        with pa.ipc.new_file(
            output_dir / f"data-{report['shard']:02d}.arrow", pa.schema([])
        ):
            pass
        pl.DataFrame(
            schema={"client_id": pl.String, "batch_index": pl.UInt32, "file": pl.String}
        ).write_parquet(output_dir / "index.pq")
    _write_json(
        output_dir / "report.json", {**report, "fit_sha256": fitted["fit_sha256"]}
    )


@logged_stage
def assemble(*, fitted_dir: Path, encoded_dir: Path, output_dir: Path) -> None:
    """Combine indexes only; workers upload Arrow payloads to final events/ directly."""
    with _phase("assemble prepared dataset indexes"):
        fitted = _json(fitted_dir / "fit.json")
        indexes = []
        for worker in fitted["workers"]:
            root = encoded_dir / f"{worker['shard']:02d}"
            report = _json(root / "report.json")
            if report != {**worker, "fit_sha256": fitted["fit_sha256"]}:
                raise ValueError(
                    f"encoded shard {worker['shard']} does not match fitted state"
                )
            if worker["rows"]:
                index = pl.read_parquet(root / "index.pq")
                if (
                    len(index) != worker["clients"]
                    or set(index["file"]) != {f"data-{worker['shard']:02d}.arrow"}
                    or index["batch_index"].to_list() != list(range(len(index)))
                ):
                    raise ValueError(
                        f"invalid encoded index for shard {worker['shard']}"
                    )
                indexes.append(index)
        combined = pl.concat(indexes)
        if combined["client_id"].n_unique() != len(combined):
            raise ValueError("clients overlap between EventStore shards")
        shutil.copytree(fitted_dir, output_dir)
        events = output_dir / "dataset" / "events"
        events.mkdir()
        combined.write_parquet(events / "index.pq")
        logger.info(
            "assembled shards=%s clients=%s without copying Arrow payloads",
            len(fitted["workers"]),
            len(combined),
        )
