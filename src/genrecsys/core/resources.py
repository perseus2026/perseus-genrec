"""Named immutable resources shared by every dataset split."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import polars as pl


@dataclass
class Resources:
    """Parquet-backed mapper and representation artifacts."""

    tables: dict[str, pl.DataFrame]
    files: dict[str, Path] = field(default_factory=dict)
    # Frozen representation models are applied by preparation, never by adapters.
    representations: dict[str, dict] = field(default_factory=dict)

    def resource(self, name: str, /) -> pl.DataFrame:
        """Return a named table or raise an error listing available resources."""

        try:
            return self.tables[name]
        except KeyError as error:
            raise ValueError(
                f"resource {name!r} is required; available: {sorted(self.tables)}"
            ) from error

    @classmethod
    def load(cls, path: Path, /) -> Self:
        """Load Parquet tables, NPY file paths, and deferred SID recipes from the manifest.

        SID application is deferred to prepare_dataset.
        """

        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise TypeError("resources manifest must be an object")
        tables: dict[str, pl.DataFrame] = {}
        files: dict[str, Path] = {}
        representations = {}
        for name, description in manifest.items():
            filename = (
                description["path"] if isinstance(description, dict) else description
            )
            if not isinstance(filename, str):
                raise TypeError(f"resource {name!r} path must be a string")
            if (
                isinstance(description, dict)
                and description.get("format") == "sid_mapper"
            ):
                representations[str(name)] = {**description, "path": path / filename}
            elif isinstance(description, dict) and description.get("format") == "npy":
                files[str(name)] = path / filename
            else:
                tables[str(name)] = pl.read_parquet(path / filename)
        return cls(tables, files, representations)


def copy_resource(source: Path, destination: Path) -> None:
    """Preserve immutable large resources without a second local disk copy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and source.samefile(destination):
        return
    if not destination.exists():
        try:
            destination.hardlink_to(source)
            return
        except OSError:
            pass  # Cross-filesystem copies are required by some job runners.
    shutil.copyfile(source, destination)


__all__ = ["Resources"]
