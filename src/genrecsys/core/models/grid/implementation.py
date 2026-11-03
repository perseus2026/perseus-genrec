"""GRID plugin for the shared GenRecSys lifecycle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import polars as pl
import torch

from genrecsys.core.checkpoint import Prepared, Trained
from genrecsys.core.config import Config
from genrecsys.core.dataset import Dataset
from genrecsys.core.models.base import DataRequirements, Implementation, TrainedModel
from genrecsys.core.models.grid import config as grid_config
from genrecsys.core.models.grid import inference, training
from genrecsys.core.models.grid.catalog import (
    _catalog,
    _item_preprocessor,
    _training_targets,
)
from genrecsys.core.tasks import retrieval


def data_requirements(config: Config, /) -> DataRequirements:
    """Describe candidates consumed alongside the full frozen SID mapper."""

    encoder = config.encoders[config.features[config.task.objective.target].encoder]
    key = encoder.preprocessor.get("item_id_column", "item_key")
    return DataRequirements(artifact_columns=tuple(dict.fromkeys(("item_id", key))))


def validate_prepared(checkpoint: Prepared, dataset: Dataset, /) -> None:
    """Reject incompatible preprocessing and incomplete SID catalogs before allocating a GPU."""

    checkpoint.data_contract.require_compatible(checkpoint.config)
    grid_config._model_config(checkpoint)
    if not len(dataset.train_samples) or not len(dataset.validation_samples):
        raise ValueError("training and validation splits must not be empty")
    if (
        dataset.event_store is None
        or not (dataset.event_store.path / "index.pq").exists()
    ):
        raise ValueError("GRID requires a prepared EventStore")
    item = _item_preprocessor(checkpoint)
    for split, artifacts in (
        ("train", dataset.train_artifacts),
        ("validation", dataset.validation_artifacts),
        ("test", dataset.test_artifacts),
    ):
        if artifacts is None and split == "test":
            continue
        if not isinstance(artifacts, retrieval.Artifacts):
            raise TypeError(f"GRID {split} requires retrieval.Artifacts")
        _catalog(artifacts, item)
    _training_targets(dataset, dataset.train_artifacts, item)
    if grid_config.augmentation_parameters(checkpoint.config) is not None:
        samples = dataset.train_samples
        targets = (
            samples["target"].to_list()
            if isinstance(samples, pl.DataFrame)
            else samples.column("target")
        )
        if any(target is None or len(target) != 1 for target in targets):
            raise ValueError(
                "contiguous augmentation requires one next-item target per training row"
            )


@dataclass
class State(TrainedModel):
    """Portable GRID weights and frozen candidate catalog."""

    state_dict: dict[str, torch.Tensor]
    catalog: pl.DataFrame
    training: dict[str, Any]
    model_config: dict[str, Any]
    is_main_process: bool = True

    def save(self, path: Path, /) -> None:
        """Save CPU weights, catalog.pq, training.json, and model.json."""

        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict, path / "weights.pt")
        self.catalog.write_parquet(path / "catalog.pq")
        (path / "training.json").write_text(
            json.dumps(self.training, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (path / "model.json").write_text(
            json.dumps(
                {"config": self.model_config},
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path, /) -> Self:
        """Load weights on CPU and restore the catalog and model settings."""

        model = json.loads((path / "model.json").read_text(encoding="utf-8"))
        return cls(
            state_dict=torch.load(
                path / "weights.pt",
                map_location="cpu",
                weights_only=True,
            ),
            catalog=pl.read_parquet(path / "catalog.pq"),
            training=json.loads((path / "training.json").read_text(encoding="utf-8")),
            model_config=model["config"],
        )


def fit_model(checkpoint: Prepared, dataset: Dataset, /) -> State:
    """Train the single-history GRID implementation."""

    return State(**training.fit_model(checkpoint, dataset))


def create_inference_session(
    checkpoint: Trained,
    /,
    *,
    artifacts: retrieval.Artifacts | None = None,
) -> inference.InferenceSession:
    """Load one reusable GPU inference session for this process."""

    state = checkpoint.model
    if not isinstance(state, State):
        raise TypeError("checkpoint does not contain a native GRID state")
    if artifacts is not None and not isinstance(artifacts, retrieval.Artifacts):
        raise TypeError("GRID inference requires retrieval.Artifacts")
    session = inference.InferenceSession(checkpoint, state, artifacts=artifacts)
    if artifacts is not None:
        # The checkpoint catalog belongs to validation. Final inference uses the
        # split-specific catalog above, so release the otherwise duplicated frame.
        state.catalog = state.catalog.clear()
    return session


implementation = Implementation(
    trained_model=State,
    fit_model=fit_model,
    validate_config=grid_config.validate,
    data_requirements=data_requirements,
    create_inference_session=create_inference_session,
    validate_prepared=validate_prepared,
)

__all__ = [
    "State",
    "create_inference_session",
    "data_requirements",
    "fit_model",
    "implementation",
    "validate_prepared",
]
