"""Resolve local data source directories from environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

SOURCE_ENV_PREFIX = "internal_storage_"


def parse_from_env(
    environ: Mapping[str, str] | None = None,
    *,
    prefix: str = SOURCE_ENV_PREFIX,
) -> dict[str, Path]:
    """Resolve an ``INTERNAL_STORAGE_<SOURCE>`` mount from the environment."""

    values = os.environ if environ is None else environ
    normalized_prefix = prefix.lower()
    return {
        key.lower().removeprefix(normalized_prefix): Path(value)
        for key, value in values.items()
        if key.lower().startswith(normalized_prefix)
    }
