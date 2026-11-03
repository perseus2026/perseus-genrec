"""Immutable description of the state materialized by dataset preparation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

from genrecsys.core.config import Config, materialization_histories


@dataclass(frozen=True)
class DataContract:
    """Canonical subset of config fields that affect prepared dataset bytes."""

    format_version: int
    recipe: dict[str, Any]
    sha256: str

    @classmethod
    def from_config(cls, config: Config, /) -> Self:
        """Hash the canonical config fields that affect prepared data.

        Include history materialization bounds; exclude model and training settings.
        """

        config.validate_data()
        recipe = {
            "task": {
                "type": config.task.type,
                "target": config.task.objective.target,
            },
            # Keep the default priority byte-compatible with prepared artifacts.
            "events": {
                name: {
                    "attributes": {
                        attribute_name: asdict(attribute)
                        for attribute_name, attribute in value.attributes.items()
                    },
                    "source": value.source,
                    **({"priority": value.priority} if value.priority != 0 else {}),
                }
                for name, value in config.events.items()
            },
            "max_duration_per_sequence": config.max_duration_per_sequence,
            "encoders": {
                name: {
                    "type": value.type,
                    "preprocessor": value.preprocessor,
                }
                for name, value in config.encoders.items()
            },
            "features": {
                name: asdict(value) for name, value in config.features.items()
            },
            "histories": materialization_histories(config),
        }
        payload = json.dumps(
            recipe,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        return cls(2, json.loads(payload), hashlib.sha256(payload).hexdigest())

    def save(self, path: Path, /) -> None:
        """Write the data recipe and its checksum to JSON."""

        path.write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path, /) -> Self:
        """Load the contract and verify its format version and recipe checksum."""

        payload = json.loads(path.read_text(encoding="utf-8"))
        result = cls(
            format_version=int(payload["format_version"]),
            recipe=dict(payload["recipe"]),
            sha256=str(payload["sha256"]),
        )
        if result.format_version != 2:
            raise ValueError(
                f"unsupported prepared data contract version {result.format_version}"
            )
        encoded = json.dumps(
            result.recipe,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        if hashlib.sha256(encoded).hexdigest() != result.sha256:
            raise ValueError("prepared data contract checksum is invalid")
        return result

    def require_compatible(self, config: Config, /) -> None:
        """Reject incompatible data settings and list the changed sections.

        Ignore model/training settings; incompatible prepared data requires Stage 08.
        """

        candidate = type(self).from_config(config)
        if candidate.sha256 == self.sha256:
            return
        changed = sorted(
            key
            for key in set(self.recipe) | set(candidate.recipe)
            if self.recipe.get(key) != candidate.recipe.get(key)
        )
        raise ValueError(
            "experiment config changes the prepared data contract; rerun Stage 08 "
            f"(changed sections: {changed})"
        )


__all__ = ["DataContract"]
