"""Resolve named preparation inputs without requiring adapter-authored manifests."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from genrecsys.core.config import Config
from genrecsys.core.dataset_manifest import DatasetManifest
from genrecsys.core.event_feature_mapping import Spec

# Paths are relative to the workdir, independent of dataset and model names.
DEFAULT_RESOURCES = {
    "item_to_sid": "resources/sid/item_to_sid.parquet",
    "user_features": "resources/user_features.parquet",
    "item_features": "resources/item_features.parquet",
    "embedding_items": "embeddings/inference/items.pq",
    "item_embeddings": "embeddings/inference/embeddings.npy",
}


def prepare_inputs(
    workdir: Path,
    config: Config,
    *,
    resources: dict[str, str | dict] | None = None,
    event_feature_mappings: tuple[Spec, ...] | None = None,
    metadata: dict[str, Any] | None = None,
) -> DatasetManifest:
    """Validate inputs and persist generated manifests for later preparation stages.

    Explicit paths are relative to workdir (or absolute). Existing resource
    manifests keep their resources-relative paths. Unused defaults are not loaded.
    An existing manifest is validated, not rebuilt, unless overrides are supplied.
    """
    manifest_path = workdir / "manifest.json"
    existing = DatasetManifest.load(manifest_path) if manifest_path.exists() else None
    if (
        existing is not None
        and not resources
        and event_feature_mappings is None
        and metadata is None
    ):
        existing.validate(workdir, config)
        return existing

    resource_root = workdir / "resources"
    resource_path = resource_root / "manifest.json"
    descriptions = (
        json.loads(resource_path.read_text()) if resource_path.exists() else {}
    )
    if not isinstance(descriptions, dict):
        raise TypeError("resources manifest must be an object")

    def describe(value: str | dict) -> str | dict:
        description = {"path": value} if isinstance(value, str) else dict(value)
        path = workdir / description["path"]
        description["path"] = os.path.relpath(path, resource_root)
        if path.suffix == ".npy":
            description.setdefault("format", "npy")
        return description["path"] if set(description) == {"path"} else description

    descriptions.update(
        {name: describe(value) for name, value in (resources or {}).items()}
    )
    mappings = event_feature_mappings
    if mappings is None:
        mappings = existing.event_feature_mappings if existing else ()
    required = {
        params["artifact"]
        for encoder in config.encoders.values()
        for params in (encoder.preprocessor, encoder.embedder)
        if isinstance(params.get("artifact"), str)
    }
    required.update(spec.resource for spec in mappings)
    # Deferred SID application reuses the same full-catalog vectors, never train-only vectors.
    for description in descriptions.values():
        if isinstance(description, dict) and description.get("format") == "sid_mapper":
            required.update((description["items"], description["embeddings"]))
    for name in sorted(required - descriptions.keys()):
        if name not in DEFAULT_RESOURCES:
            raise ValueError(
                f"No default path for resource {name!r}; pass --resource {name}=PATH"
            )
        descriptions[name] = describe(DEFAULT_RESOURCES[name])

    manifest = DatasetManifest.build(
        workdir,
        resource_descriptions=descriptions,
        event_feature_mappings=mappings,
        metadata={**(existing.metadata if existing else {}), **(metadata or {})},
    )
    manifest.validate(workdir, config)
    resource_root.mkdir(parents=True, exist_ok=True)
    resource_path.write_text(
        json.dumps(descriptions, indent=2) + "\n", encoding="utf-8"
    )
    manifest.save(manifest_path)
    return manifest
