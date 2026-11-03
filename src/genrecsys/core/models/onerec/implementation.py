"""OneRec plugin for the shared GenRecSys lifecycle."""

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
from genrecsys.core.models.onerec import config as onerec_config
from genrecsys.core.models.onerec import inference, training
from genrecsys.core.tasks import retrieval


def data_requirements(config: Config, /) -> DataRequirements:
    """Describe the prepared retrieval state consumed by this OneRec recipe."""

    return DataRequirements(
        artifact_columns=("item_id",),
        checkpoint_files=tuple(
            encoder.embedder["artifact"]
            for encoder in config.encoders.values()
            if encoder.type == "id" and encoder.embedder.get("artifact")
        ),
    )


def validate_prepared(checkpoint: Prepared, dataset: Dataset, /) -> None:
    """Validate catalog columns and feature files before GPU allocation.

    The shared lifecycle validates data schemas and task settings.
    """
    requirements = data_requirements(checkpoint.config)
    if missing := set(requirements.checkpoint_files) - set(checkpoint.files):
        raise ValueError(f"OneRec checkpoint misses files: {sorted(missing)}")
    for split, artifacts in (
        ("train", dataset.train_artifacts),
        ("validation", dataset.validation_artifacts),
        ("test", dataset.test_artifacts),
    ):
        if artifacts is None:
            continue
        items = artifacts.items
        if missing := set(requirements.artifact_columns) - set(items.columns):
            raise ValueError(
                f"OneRec {split} artifacts miss columns: {sorted(missing)}"
            )


@dataclass
class State(TrainedModel):
    """Portable production-OneRec weights and frozen candidate catalog."""

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
    """Train the production multi-path OneRec implementation."""

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
        raise TypeError("checkpoint does not contain a native OneRec state")
    if artifacts is not None and not isinstance(artifacts, retrieval.Artifacts):
        raise TypeError("OneRec inference requires retrieval.Artifacts")
    session = inference.InferenceSession(checkpoint, state, artifacts=artifacts)
    if artifacts is not None:
        # The checkpoint catalog belongs to validation. Final inference uses the
        # split-specific catalog above, so release the otherwise duplicated frame.
        state.catalog = state.catalog.clear()
    return session


implementation = Implementation(
    trained_model=State,
    fit_model=fit_model,
    validate_config=onerec_config.validate,
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
