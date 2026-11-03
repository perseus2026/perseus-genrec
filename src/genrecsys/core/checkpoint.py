"""Prepared and trained lifecycle states."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from genrecsys.core.config import Config
from genrecsys.core.data_contract import DataContract
from genrecsys.core.encoders.base import Preprocessor
from genrecsys.core.event_feature_mapping import EventFeatureMapper
from genrecsys.core.models.base import TrainedModel
from genrecsys.core.resources import copy_resource


@dataclass
class Prepared:
    """Configuration plus train-fitted feature preprocessors."""

    config: Config
    encoder_to_preprocessor: dict[str, Preprocessor]
    data_contract: DataContract
    event_feature_mappers: dict[str, EventFeatureMapper] = field(default_factory=dict)
    files: dict[str, Path] = field(default_factory=dict)
    dataset_identity: dict[str, Any] | None = None

    def save(self, path: Path, /) -> None:
        """Save config, data contract, preprocessors, and shared NPY resources."""

        path.mkdir(parents=True, exist_ok=True)
        self.config.save(path / "config.yaml")
        self.data_contract.save(path / "data_contract.json")
        if self.dataset_identity is not None:
            (path / "dataset_identity.json").write_text(
                json.dumps(self.dataset_identity, indent=2) + "\n", encoding="utf-8"
            )
        for name, source in self.files.items():
            copy_resource(source, path / "files" / f"{name}.npy")
        for encoder, preprocessor in self.encoder_to_preprocessor.items():
            preprocessor.save(path / "encoders" / "preprocessors" / encoder)
        for feature, mapper in self.event_feature_mappers.items():
            mapper.save(path / "event_feature_mappers" / feature)

    @classmethod
    def load(
        cls,
        path: Path,
        /,
        *,
        config: Config | None = None,
    ) -> Self:
        """Load preprocessing and its required contract without constructing a model.

        An explicit compatible config replaces the stored training/model recipe,
        while the original materialization contract is preserved.
        """

        from genrecsys.core.encoders import registry

        contract = DataContract.load(path / "data_contract.json")
        config = (
            config if config is not None else Config.load_data(path / "config.yaml")
        )
        contract.require_compatible(config)
        preprocessors = {
            name: registry[description.type].preprocessor.load(
                path / "encoders" / "preprocessors" / name
            )
            for name, description in config.encoders.items()
        }
        mappers_path = path / "event_feature_mappers"
        mappers = (
            {
                feature_path.name: EventFeatureMapper.load(feature_path)
                for feature_path in sorted(mappers_path.iterdir())
                if feature_path.is_dir()
            }
            if mappers_path.exists()
            else {}
        )
        files = {file.stem: file for file in (path / "files").glob("*.npy")}
        identity_path = path / "dataset_identity.json"
        identity = (
            json.loads(identity_path.read_text()) if identity_path.exists() else None
        )
        return cls(config, preprocessors, contract, mappers, files, identity)


@dataclass
class Trained:
    """One prepared state plus one indivisible trained model."""

    prepared: Prepared
    model: TrainedModel

    def save(self, path: Path, /) -> None:
        """Save the original preprocessing contract and model-owned state."""

        self.prepared.save(path)
        self.model.save(path / "model")

    @classmethod
    def load(cls, path: Path, /) -> Self:
        """Restore preprocessing and the selected model implementation."""

        from genrecsys.core.models import get

        prepared = Prepared.load(path)
        implementation = get(prepared.config.model.type)
        return cls(prepared, implementation.trained_model.load(path / "model"))
