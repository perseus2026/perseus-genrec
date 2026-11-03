"""Native HSTU plugin and portable checkpoint state."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import polars as pl
import torch

from genrecsys.core.models.base import DataRequirements, Implementation, TrainedModel
from genrecsys.core.models.hstu import config, inference, training
from genrecsys.core.models.hstu.data import catalog, vocabulary


def data_requirements(config):
    return DataRequirements()


def validate_prepared(prepared, dataset):
    """Fail before GPU allocation if catalogs/targets cannot use frozen identities."""
    if not len(dataset.train_samples) or not len(dataset.validation_samples):
        raise ValueError("HSTU train and validation samples must not be empty")
    if dataset.event_store is None:
        raise ValueError("HSTU training requires a prepared EventStore")
    tokens = vocabulary(prepared)
    for artifacts in (
        dataset.train_artifacts,
        dataset.validation_artifacts,
        dataset.test_artifacts,
    ):
        if artifacts is not None:
            catalog(artifacts.items, tokens)
    missing = dataset.task_preprocessor.items.select("item_id").join(
        dataset.train_artifacts.items.select("item_id"),
        on="item_id",
        how="anti",
    )
    if len(missing):
        raise ValueError(
            "HSTU training targets are outside the training candidate catalog"
        )


@dataclass
class State(TrainedModel):
    state_dict: dict[str, torch.Tensor]
    catalog: pl.DataFrame
    training: dict[str, Any]
    model_config: dict[str, Any]
    is_main_process: bool = True

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict, path / "weights.pt")
        self.catalog.write_parquet(path / "catalog.pq")
        for name, payload in (
            ("training", self.training),
            ("model", self.model_config),
        ):
            (path / f"{name}.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
            )

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls(
            state_dict=torch.load(
                path / "weights.pt", map_location="cpu", weights_only=True
            ),
            catalog=pl.read_parquet(path / "catalog.pq"),
            training=json.loads((path / "training.json").read_text()),
            model_config=json.loads((path / "model.json").read_text()),
        )


def fit_model(prepared, dataset):
    return State(**training.fit_model(prepared, dataset))


def create_inference_session(checkpoint, *, artifacts=None):
    if not isinstance(checkpoint.model, State):
        raise TypeError("checkpoint does not contain an HSTU model")
    return inference.InferenceSession(checkpoint, checkpoint.model, artifacts=artifacts)


implementation = Implementation(
    trained_model=State,
    fit_model=fit_model,
    validate_config=config.validate,
    data_requirements=data_requirements,
    create_inference_session=create_inference_session,
    validate_prepared=validate_prepared,
)
