"""Common options for sequential and sharded dataset preparation."""

import json
from collections.abc import Sequence
from typing import Annotated, Any

from cyclopts import Parameter, Token

from genrecsys.core.event_feature_mapping import Spec
from genrecsys.core.preparation_inputs import DEFAULT_RESOURCES


def parse_metadata(_: Any, tokens: Sequence[Token]) -> dict[str, Any]:
    """Parse one JSON object for preparation metadata."""
    if len(tokens) != 1:
        raise ValueError("metadata requires one JSON value")
    value = json.loads(tokens[0].value)
    if not isinstance(value, dict):
        raise TypeError("metadata requires a JSON object")
    return value


Metadata = Annotated[dict[str, Any], Parameter(converter=parse_metadata)]


def parse_resources(values: tuple[str, ...]) -> dict[str, str | dict]:
    """Parse repeated NAME=PATH or NAME=JSON descriptors; NAME alone selects a default."""
    resources = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not name or name in resources:
            raise ValueError(f"Empty or repeated resource name: {name!r}")
        if not separator:
            if name not in DEFAULT_RESOURCES:
                raise ValueError(f"No default path for {name!r}; use NAME=PATH")
            path = DEFAULT_RESOURCES[name]
        if not path:
            raise ValueError(f"Empty resource path: {name!r}")
        resources[name] = json.loads(path) if path.startswith("{") else path
    return resources


def parse_mappings(values: tuple[str, ...] | None) -> tuple[Spec, ...] | None:
    """Parse exact feature joins; omission preserves an existing bundle's rules."""
    return (
        None
        if values is None
        else tuple(Spec.from_dict(json.loads(value)) for value in values)
    )
