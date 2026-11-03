"""Generated contract between dataset materialization and the core lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Self

import numpy as np
import polars as pl

from genrecsys.core.basis import validate_basis
from genrecsys.core.config import Config
from genrecsys.core.event_feature_mapping import Spec
from genrecsys.utils.files import file_sha256

_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class DatasetManifest:
    """Capabilities derived from materialized files, never authored by users."""

    format_version: int
    splits: dict[str, dict[str, Any]]
    resources: dict[str, dict[str, Any]]
    event_feature_mappings: tuple[Spec, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        workdir: Path,
        /,
        *,
        event_feature_mappings: tuple[Spec, ...] = (),
        metadata: dict[str, Any] | None = None,
        resource_descriptions: dict[str, str | dict] | None = None,
    ) -> Self:
        """Build a manifest from split schemas, row counts, and bundle resources.

        Memory-map embedding arrays and record event mapping specs; do not read EventHub.
        """

        splits: dict[str, dict[str, Any]] = {}
        for split in _SPLITS:
            root = workdir / "basis" / split
            samples = root / "samples.pq"
            items = root / "artifacts" / "items.pq"
            splits[split] = {
                "samples": str(samples.relative_to(workdir)),
                "artifacts": str((root / "artifacts").relative_to(workdir)),
                "rows": _parquet_rows(samples),
                "artifact_rows": _parquet_rows(items),
                "sample_schema": _schema(samples),
                "artifact_schema": {"items": _schema(items)},
            }
        resources_root = workdir / "resources"
        resource_manifest = resource_descriptions
        if resource_manifest is None:
            resource_manifest = json.loads(
                (resources_root / "manifest.json").read_text(encoding="utf-8")
            )
        resources = {}
        for name, description in resource_manifest.items():
            filename = (
                description["path"] if isinstance(description, dict) else description
            )
            path = resources_root / filename
            logical = (
                {
                    key: value
                    for key, value in description.items()
                    if key not in {"path", "format"}
                }
                if isinstance(description, dict)
                else {}
            )
            resources[str(name)] = {
                "path": os.path.relpath(path, workdir),
                **_resource_description(path),
                **logical,
            }
        return cls(2, splits, resources, event_feature_mappings, metadata or {})

    @classmethod
    def load(cls, path: Path, /) -> Self:
        """Load the manifest and event mapping specs from JSON."""

        value = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            format_version=int(value["format_version"]),
            splits=dict(value["splits"]),
            resources=dict(value["resources"]),
            event_feature_mappings=tuple(
                Spec.from_dict(spec) for spec in value.get("event_feature_mappings", ())
            ),
            metadata=dict(value.get("metadata", {})),
        )

    def save(self, path: Path, /) -> None:
        """Write split schemas, resource descriptions, and mapping specs to JSON."""

        path.write_text(
            json.dumps(
                {
                    "format_version": self.format_version,
                    "splits": self.splits,
                    "resources": self.resources,
                    "event_feature_mappings": [
                        spec.to_dict() for spec in self.event_feature_mappings
                    ],
                    "metadata": self.metadata,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    def identity(self, workdir: Path, /) -> dict[str, Any]:
        """Fingerprint actual basis/resources and their declared dataset protocol.

        This is separate from DataContract, which describes preparation settings.
        Raw EventHub payloads are identified by the source revision in metadata.
        Hash files during preparation, never inside training or validation epochs.
        """
        paths = {"resources/manifest.json"}
        for split in self.splits.values():
            paths.update((split["samples"], f"{split['artifacts']}/items.pq"))
        paths.update(resource["path"] for resource in self.resources.values())
        files = {name: file_sha256(workdir / name) for name in sorted(paths)}
        payload = json.dumps(
            {"manifest": asdict(self), "files": files},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "files": files,
            "metadata": self.metadata,
        }

    def validate(self, workdir: Path, config: Config, /) -> None:
        """Validate bundle files against the data configuration.

        Check schemas, counts, context, resources, SID alignment, and event features;
        do not validate model architecture.
        """

        if self.format_version != 2:
            raise ValueError(
                f"unsupported dataset manifest version {self.format_version}"
            )
        required_context = set(config.feature_to_encoder(context=True))
        for split in _SPLITS:
            try:
                description = self.splits[split]
            except KeyError as error:
                raise ValueError(f"dataset manifest misses split {split!r}") from error
            samples_path = workdir / description["samples"]
            artifacts_path = workdir / description["artifacts"]
            sample_schema = pl.read_parquet_schema(samples_path)
            if _schema(samples_path) != description["sample_schema"]:
                raise ValueError(f"{split} sample schema differs from manifest")
            if _parquet_rows(samples_path) != description["rows"]:
                raise ValueError(f"{split} sample count differs from manifest")
            required_sample_columns = {"client_id", "timestamp", "target"}
            if missing := required_sample_columns - set(sample_schema):
                raise ValueError(f"{split} samples miss columns: {sorted(missing)}")
            if required_context:
                context = sample_schema.get("context")
                if not isinstance(context, pl.Struct):
                    raise ValueError(
                        f"{split} samples require a context struct for "
                        f"{sorted(required_context)}"
                    )
                available = {field.name for field in context.fields}
                if missing := required_context - available:
                    raise ValueError(
                        f"{split} context misses configured features: {sorted(missing)}"
                    )
            items_path = artifacts_path / "items.pq"
            item_schema = pl.read_parquet_schema(items_path)
            required_item_columns = {
                "item_id",
                *config.feature_to_encoder(artifacts=True),
            }
            if missing := required_item_columns - set(item_schema):
                raise ValueError(
                    f"{split} retrieval artifacts miss columns: {sorted(missing)}"
                )
            if _parquet_rows(items_path) != description["artifact_rows"]:
                raise ValueError(f"{split} artifact count differs from manifest")
            if _schema(items_path) != description["artifact_schema"]["items"]:
                raise ValueError(f"{split} artifact schema differs from manifest")

        required_resources = {
            value
            for encoder in config.encoders.values()
            for key, value in encoder.preprocessor.items()
            if key == "artifact" and isinstance(value, str)
        }
        required_resources.update(spec.resource for spec in self.event_feature_mappings)
        if missing := required_resources - set(self.resources):
            raise ValueError(
                f"dataset resources do not cover config requirements: {sorted(missing)}"
            )
        for name, description in self.resources.items():
            path = workdir / description["path"]
            physical = _resource_description(path)
            if physical != {key: description[key] for key in physical}:
                raise ValueError(f"resource {name!r} schema differs from manifest")
            if description.get("format") == "sid_mapper":
                for dependency in ("items", "embeddings"):
                    if description.get(dependency) not in self.resources:
                        raise ValueError(
                            f"SID resource {name!r} misses {dependency} resource"
                        )
                items = self.resources[description["items"]]
                embeddings = self.resources[description["embeddings"]]
                item_rows = items.get("rows")
                shape = embeddings.get("shape", [])
                if item_rows is not None and (not shape or item_rows != shape[0]):
                    raise ValueError(
                        f"SID resource {name!r} items and embeddings are not aligned"
                    )

        configured_event_features = config.feature_to_encoder(event=True)
        for spec in self.event_feature_mappings:
            if spec.feature not in configured_event_features:
                raise ValueError(
                    f"event mapping produces unconfigured feature {spec.feature!r}"
                )
            events = spec.events or tuple(config.events)
            if unknown := set(events) - set(config.events):
                raise ValueError(
                    f"event mapping {spec.feature!r} references unknown events: "
                    f"{sorted(unknown)}"
                )
            for event_name in events:
                if missing := set(spec.attributes) - set(
                    config.events[event_name].attributes
                ):
                    raise ValueError(
                        f"event mapping {spec.feature!r} misses attributes in "
                        f"{event_name!r}: {sorted(missing)}"
                    )
            resource_schema = self.resources[spec.resource]["schema"]
            required_mapping_columns = {*spec.attributes, spec.value_column}
            if missing := required_mapping_columns - set(resource_schema):
                raise ValueError(
                    f"event mapping resource {spec.resource!r} misses columns: "
                    f"{sorted(missing)}"
                )
            encoder_name = configured_event_features[spec.feature]
            encoder = config.encoders[encoder_name]
            artifact_name = encoder.preprocessor.get("artifact")
            if encoder.type == "semantic_id" and isinstance(artifact_name, str):
                encoder_schema = self.resources[artifact_name].get("schema")
                if (
                    encoder_schema is not None
                    and spec.value_column not in encoder_schema
                ):
                    raise ValueError(
                        f"encoder resource {artifact_name!r} misses mapping key "
                        f"{spec.value_column!r}"
                    )
            if spec.feature == config.task.objective.target:
                for split in _SPLITS:
                    artifact_schema = self.splits[split]["artifact_schema"]["items"]
                    if spec.value_column not in artifact_schema:
                        raise ValueError(
                            f"{split} artifacts miss mapping key {spec.value_column!r}"
                        )

        mapped_features = {spec.feature for spec in self.event_feature_mappings}
        available_event_features = {"event"}
        for event in config.events.values():
            available_event_features.update(event.attributes)
        available_event_features.update(mapped_features)
        available_event_features.update(
            name
            for name, feature in config.features.items()
            if feature.transform is not None
        )
        required_event_features = set(config.feature_to_encoder(event=True))
        if missing := required_event_features - available_event_features:
            raise ValueError(
                f"dataset events do not provide configured features: {sorted(missing)}"
            )
        validate_basis(workdir, config)


def finalize_bundle(
    workdir: Path,
    config: Config,
    *,
    dataset: str,
    source: dict[str, str],
    protocol: dict[str, Any],
    event_feature_mappings: tuple[Spec, ...] = (),
) -> DatasetManifest:
    """Finish an adapter's bundle using the common validation and identity contract."""
    if not dataset or not source.get("repository") or not source.get("revision"):
        raise ValueError("dataset source requires a repository and revision")
    if not protocol.get("name"):
        raise ValueError("dataset protocol requires a name")
    manifest = DatasetManifest.build(
        workdir,
        event_feature_mappings=event_feature_mappings,
        metadata={"dataset": dataset, "source": source, "protocol": protocol},
    )
    manifest.validate(workdir, config)
    config.save(workdir / "config.yaml")
    manifest.save(workdir / "manifest.json")
    (workdir / "dataset_identity.json").write_text(
        json.dumps(manifest.identity(workdir), indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _schema(path: Path) -> dict[str, str]:
    """Read a Parquet schema and serialize column types as strings."""

    return {name: str(dtype) for name, dtype in pl.read_parquet_schema(path).items()}


def _resource_description(path: Path) -> dict[str, Any]:
    """Describe a SID checkpoint, embedding matrix, or Parquet resource.

    Memory-map NPY arrays and require a two-dimensional floating-point matrix.
    """

    if path.suffix == ".pkl":
        return {"format": "sid_mapper", "bytes": path.stat().st_size}
    if path.suffix == ".npy":
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.ndim != 2 or values.dtype.kind != "f":
            raise ValueError(f"embedding resource must be a floating matrix: {path}")
        return {
            "format": "npy",
            "shape": list(values.shape),
            "dtype": str(values.dtype),
        }
    return {"schema": _schema(path), "rows": _parquet_rows(path)}


def _parquet_rows(path: Path) -> int:
    """Return the Parquet row count without loading its columns."""

    return int(pl.scan_parquet(path).select(pl.len()).collect().item())


__all__ = ["DatasetManifest", "finalize_bundle"]
