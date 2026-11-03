"""Generic exact mappings from raw event attributes to model features."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

import polars as pl

from genrecsys.core.config import Config
from genrecsys.core.resources import Resources


@dataclass(frozen=True)
class Spec:
    """Declarative binding stored in the generated dataset manifest."""

    feature: str
    resource: str
    attributes: tuple[str, ...]
    value_column: str
    events: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any], /) -> Self:
        """Parse the binding between raw event attributes and a named resource."""

        return cls(
            feature=str(value["feature"]),
            resource=str(value["resource"]),
            attributes=tuple(str(name) for name in value["attributes"]),
            value_column=str(value["value_column"]),
            events=tuple(str(name) for name in value.get("events", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the mapping spec, converting tuple fields to JSON lists."""

        value = asdict(self)
        value["attributes"] = list(self.attributes)
        value["events"] = list(self.events)
        return value


@dataclass
class EventFeatureMapper:
    """Apply an immutable exact lookup without dataset-specific code."""

    feature: str
    attributes: tuple[str, ...]
    value_column: str
    events: tuple[str, ...]
    mapping: pl.DataFrame

    @classmethod
    def from_spec(
        cls,
        spec: Spec,
        config: Config,
        resources: Resources,
        /,
    ) -> Self:
        """Build a frozen exact lookup after validating events, columns, and key uniqueness."""

        if spec.feature not in config.feature_to_encoder(event=True):
            raise ValueError(
                f"event mapping produces unconfigured feature {spec.feature!r}"
            )
        events = spec.events or tuple(config.events)
        unknown_events = set(events) - set(config.events)
        if unknown_events:
            raise ValueError(
                f"event mapping {spec.feature!r} references unknown events: "
                f"{sorted(unknown_events)}"
            )
        for event_name in events:
            missing = set(spec.attributes) - set(config.events[event_name].attributes)
            if missing:
                raise ValueError(
                    f"event mapping {spec.feature!r} misses attributes in "
                    f"{event_name!r}: {sorted(missing)}"
                )
        mapping = resources.resource(spec.resource)
        required = {*spec.attributes, spec.value_column}
        if missing := required - set(mapping.columns):
            raise ValueError(
                f"event mapping resource {spec.resource!r} misses columns: "
                f"{sorted(missing)}"
            )
        mapping = mapping.select(*spec.attributes, spec.value_column).unique()
        ambiguous = (
            mapping.group_by(*spec.attributes)
            .agg(pl.col(spec.value_column).n_unique().alias("__values"))
            .filter(pl.col("__values") != 1)
        )
        if not ambiguous.is_empty():
            raise ValueError(
                f"event mapping resource {spec.resource!r} has ambiguous keys"
            )
        return cls(
            feature=spec.feature,
            attributes=spec.attributes,
            value_column=spec.value_column,
            events=tuple(events),
            mapping=mapping.unique(spec.attributes),
        )

    def transform(self, events: pl.DataFrame, /) -> pl.DataFrame:
        """Add a mapped feature with a left join, preserving event row order.

        Unknown keys receive nulls. Preserve raw values on unrelated event streams.
        """

        eligible = pl.col("event").is_in(self.events)
        value_dtype = self.mapping.schema[self.value_column]
        existing = f"__raw_{self.feature}"
        mapped = f"__mapped_{self.feature}"
        values = events
        if self.feature in values.columns:
            if existing in values.columns:
                raise ValueError(f"reserved event feature column exists: {existing!r}")
            values = values.rename({self.feature: existing})
        result = (
            values.with_row_index("__event_mapping_index")
            .join(
                self.mapping.rename({self.value_column: mapped}),
                how="left",
                on=list(self.attributes),
                maintain_order="left",
                validate="m:1",
                nulls_equal=True,
            )
            .with_columns(
                pl.when(eligible)
                .then(pl.col(mapped))
                .otherwise(
                    pl.col(existing)
                    if existing in values.columns
                    else pl.lit(None, dtype=value_dtype)
                )
                .alias(self.feature)
            )
            .sort("__event_mapping_index")
            .drop("__event_mapping_index", mapped)
        )
        return result.drop(existing) if existing in result.columns else result

    def save(self, path: Path, /) -> None:
        """Save mapping.pq and its key/event settings in config.json."""

        path.mkdir(parents=True, exist_ok=True)
        self.mapping.write_parquet(path / "mapping.pq")
        (path / "config.json").write_text(
            json.dumps(
                {
                    "feature": self.feature,
                    "attributes": list(self.attributes),
                    "value_column": self.value_column,
                    "events": list(self.events),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path, /) -> Self:
        """Load the lookup table and its key/event settings."""

        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        return cls(
            feature=str(config["feature"]),
            attributes=tuple(str(name) for name in config["attributes"]),
            value_column=str(config["value_column"]),
            events=tuple(str(name) for name in config["events"]),
            mapping=pl.read_parquet(path / "mapping.pq"),
        )


def build_mappers(
    specs: tuple[Spec, ...],
    config: Config,
    resources: Resources,
    /,
) -> dict[str, EventFeatureMapper]:
    """Build manifest-declared lookup mappers, rejecting duplicate output features."""

    mappers = {
        spec.feature: EventFeatureMapper.from_spec(spec, config, resources)
        for spec in specs
    }
    if len(mappers) != len(specs):
        raise ValueError("event feature mapping outputs must be unique")
    return mappers


__all__ = ["EventFeatureMapper", "Spec", "build_mappers"]
