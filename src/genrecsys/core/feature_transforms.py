"""Config-selected event feature transformations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import polars as pl

from genrecsys.core.registry import Registry

BuildExpression = Callable[[Mapping[str, Any]], pl.Expr]
ValidateParameters = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class Implementation:
    """One declarative transformation from raw EventHub columns to a feature."""

    build: BuildExpression
    validate: ValidateParameters


registry: Registry[Implementation] = Registry("event feature transform")


def register(
    name: str,
    build: BuildExpression,
    /,
    *,
    validate: ValidateParameters | None = None,
) -> None:
    """Register a Polars expression builder and its parameter validator."""

    registry.register(name, Implementation(build, validate or _require_no_parameters))


def expression(name: str, params: Mapping[str, Any], /) -> pl.Expr:
    """Validate transform parameters and build the registered Polars expression."""

    implementation = registry[name]
    implementation.validate(params)
    return implementation.build(params)


def validate(name: str, params: Mapping[str, Any], /) -> None:
    """Validate transform parameters without building an expression or reading data."""

    registry[name].validate(params)


def _require_no_parameters(params: Mapping[str, Any]) -> None:
    """Reject parameters for a transform with no configurable settings."""

    if params:
        raise ValueError("feature transform does not accept parameters")


def _namespace_validate(params: Mapping[str, Any]) -> None:
    """Require at least two namespace fields and a string separator."""

    if set(params) - {"fields", "separator"}:
        raise ValueError("namespace transform accepts only fields and separator")
    fields = params.get("fields")
    if (
        not isinstance(fields, (list, tuple))
        or len(fields) < 2
        or any(not isinstance(value, str) or not value for value in fields)
    ):
        raise ValueError("namespace transform requires at least two field names")
    separator = params.get("separator", ":")
    if not isinstance(separator, str):
        raise TypeError("namespace transform separator must be a string")


def _namespace(params: Mapping[str, Any]) -> pl.Expr:
    """Join namespace fields into a string key; missing components produce null."""

    return pl.concat_str(
        [pl.col(name).cast(pl.String) for name in params["fields"]],
        separator=params.get("separator", ":"),
        ignore_nulls=False,
    )


register("weekday", lambda _: pl.col("timestamp").dt.weekday() - 1)
register("time_of_day", lambda _: pl.col("timestamp").dt.hour() // 6)
register(
    "event_with_optional_subtype",
    lambda _: (
        pl.when(
            pl.col("interaction_subtype").is_not_null()
            & (pl.col("interaction_subtype").str.strip_chars() != "")
        )
        .then(
            pl.concat_str(
                pl.col("event"),
                pl.lit(":"),
                pl.col("interaction_subtype").str.strip_chars(),
            )
        )
        .otherwise(pl.col("event"))
    ),
)
register("namespace", _namespace, validate=_namespace_validate)


__all__ = ["Implementation", "expression", "register", "registry", "validate"]
