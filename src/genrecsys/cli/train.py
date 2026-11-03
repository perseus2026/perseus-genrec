"""Commands for the shared GenRecSys training lifecycle."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from accelerate import PartialState
from cyclopts import App

from genrecsys import api, core
from genrecsys.cli.preparation_inputs import Metadata, parse_mappings, parse_resources
from genrecsys.core.models import get as get_model
from genrecsys.core.preparation import load_basis
from genrecsys.core.preparation_inputs import prepare_inputs
from genrecsys.utils.progress import logged_stage

app = App()


@app.command
def plan(
    *,
    config: Path,
    workdir: Path = Path("workdir/"),
) -> None:
    """Print whether a config can reuse the prepared data contract.

    Report required stages and model requirements as JSON without writing artifacts
    or launching training.
    """

    recipe = core.Config.load(config)
    candidate = core.DataContract.from_config(recipe)
    checkpoint_root = workdir / "checkpoint"
    contract_path = checkpoint_root / "data_contract.json"
    existing = None
    if contract_path.exists():
        existing = core.DataContract.load(contract_path)
    compatible = False
    reason = "prepared checkpoint is absent"
    if existing is not None:
        try:
            existing.require_compatible(recipe)
        except ValueError as error:
            reason = str(error)
        else:
            compatible = True
            reason = "experiment reuses the immutable prepared data contract"
    implementation = get_model(recipe.model.type)
    print(
        json.dumps(
            {
                "read_only": True,
                "data_contract_sha256": candidate.sha256,
                "reuses_prepared_dataset": compatible,
                "reason": reason,
                "stages": [10, 12, 13] if compatible else [8, 10, 12, 13],
                "model_requirements": asdict(implementation.data_requirements(recipe)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


@app.command
@logged_stage
def prepare_dataset(
    *,
    collect_statistics: bool = False,
    resource: tuple[str, ...] = (),
    event_feature_mapping: tuple[str, ...] | None = None,
    metadata: Metadata | None = None,
    config: Path | None = None,
    workdir: Path = Path("workdir/"),
) -> None:
    """Prepare dataset/ and checkpoint/ from the materialized bundle.

    Resolve named resources and freeze preprocessing state; no model weights are trained.

    resource: Repeat NAME=PATH (relative to workdir or absolute), or NAME for a default.
        Advanced resources accept NAME=JSON with path/format and dependency names.
    event_feature_mapping: Repeat JSON objects describing exact event-feature joins.
    metadata: JSON object with dataset, source and protocol information.
    """

    recipe = core.Config.load_data(config or workdir / "config.yaml")
    manifest = prepare_inputs(
        workdir,
        recipe,
        resources=parse_resources(resource),
        event_feature_mappings=parse_mappings(event_feature_mapping),
        metadata=metadata,
    )
    basis = load_basis(workdir, recipe)
    checkpoint, dataset = api.prepare_dataset(
        recipe,
        basis.pop("train_samples"),
        basis.pop("validation_samples"),
        **basis,
        resources=core.Resources.load(workdir / "resources"),
        event_feature_mappings=manifest.event_feature_mappings,
        collect_statistics=collect_statistics,
    )
    checkpoint.dataset_identity = manifest.identity(workdir)
    # This directory contains only preprocessing state; model weights appear at fit-model.
    checkpoint.save(workdir / "checkpoint")
    dataset.save(workdir / "dataset")


@app.command
@logged_stage
def fit_model(
    *,
    config: Path | None = None,
    workdir: Path = Path("workdir/"),
    memory_map_samples: bool = True,
) -> None:
    """Train on a compatible prepared dataset and save the selected checkpoint.

    With memory_map_samples=True, share samples through an Arrow disk cache across
    local ranks and workers. This costs disk space and first-run I/O, not GPU memory.
    False loads samples into RAM. Only the main rank saves model weights.
    """

    recipe = core.Config.load_data(config or workdir / "checkpoint" / "config.yaml")
    checkpoint = core.checkpoint.Prepared.load(workdir / "checkpoint", config=recipe)
    # Build the local IPC cache once, then let ranks/workers map the same file.
    with PartialState().local_main_process_first():
        dataset = core.Dataset.load(
            workdir / "dataset",
            checkpoint.config,
            memory_map_samples=memory_map_samples,
        )
    trained = api.fit_model(checkpoint, dataset)
    if getattr(trained.model, "is_main_process", True):
        trained.save(workdir / "checkpoint")


@app.command
@logged_stage
def fit_baseline(
    *,
    config: Path | None = None,
    workdir: Path = Path("workdir/"),
) -> None:
    """Fit the configured baseline and evaluate prepared holdout splits.

    Write baselines/<type> without replacing the model checkpoint.
    """

    recipe = core.Config.load_data(config or workdir / "checkpoint" / "config.yaml")
    contract_path = workdir / "checkpoint" / "data_contract.json"
    core.DataContract.load(contract_path).require_compatible(recipe)
    dataset = core.Dataset.load(
        workdir / "dataset",
        recipe,
        include_test=True,
    )
    baseline = recipe.task.baseline
    if baseline is None:
        raise ValueError("task.baseline is not configured")
    api.fit_baseline(
        recipe,
        dataset,
        output_dir=workdir / "baselines" / baseline.type,
    )


__all__ = ["fit_baseline", "fit_model", "plan", "prepare_dataset"]
