"""Compose OneRec samples and bounded rank-local DataLoaders."""

from __future__ import annotations

import random
from datetime import UTC
from typing import Any

import polars as pl
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from genrecsys.core.config import Config, duration_days
from genrecsys.core.event_store import EventStore
from genrecsys.core.inputs import select_sequence
from genrecsys.core.models.onerec import config as onerec_config
from genrecsys.core.models.onerec.collate import onerec_collate_fn
from genrecsys.core.sample_table import MappedSamples


class Samples(TorchDataset):
    """Build configured pathways from one shared native EventStore."""

    def __init__(
        self,
        samples: pl.DataFrame | MappedSamples,
        config: Config,
        /,
        *,
        code_length: int,
        event_store: EventStore | None = None,
        events: pl.DataFrame | None = None,
        index_to_sid: dict[int, tuple[int, ...]] | None = None,
        train: bool = False,
    ) -> None:
        """Configure sample routing over exactly one event source.

        Use event_store or events, not both. Apply config sequence limits and open the
        history reader lazily inside each DataLoader process.
        """

        if (event_store is None) == (events is None):
            raise ValueError("provide exactly one of event_store or events")
        self.samples = samples
        self.config = config
        self.code_length = code_length
        self.event_store_path = event_store.path if event_store is not None else None
        self.event_store_backend = (
            event_store.backend if event_store is not None else None
        )
        self.events = events
        self.index_to_sid = index_to_sid or {}
        self.train = train
        self.feature_types = {
            feature: config.encoders[description.encoder].type
            for feature, description in config.features.items()
        }
        self.source_position_layout = onerec_config.model_parameters(config).get(
            "source_position_layout", "dynamic"
        )
        self._reader = None
        self._events_by_client: dict[str, pl.DataFrame] | None = None

    def __len__(self) -> int:
        """Return the number of basis sample rows."""

        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Build one sample's context, histories, and training or evaluation targets.

        Read history strictly before an explicit history cutoff when present, or the
        sample timestamp otherwise. Training samples select a target or retain
        independent SID targets; evaluation keeps raw item targets. The returned
        user_id identifies the sample row, not an external client.
        """

        sample = self.samples.row(index, named=True)
        events = self._read_events(
            sample["client_id"], sample.get("history_timestamp", sample["timestamp"])
        )
        output: dict[str, Any] = {
            "user_id": int(sample.get("__sample_index", index)),
            "inputs": {
                name: self._compose_input(name, events, sample)
                for name in self.config.inputs
            },
            "reference_time": sample["timestamp"].replace(tzinfo=UTC).timestamp(),
        }
        if "target" not in sample:
            return output
        if self.train:
            if self.config.task.objective.target_mode in {"random", "label_weighted"}:
                indices, labels = _raw_targets(
                    sample["target"], sample.get("target_labels")
                )
                weights = (
                    [1.0] * len(indices)
                    if self.config.task.objective.target_mode == "random"
                    else [max(label, 0) for label in labels]
                )
                selected = (
                    random.choices(indices, weights=weights, k=1)[0]
                    if sum(weights)
                    else random.choice(indices)
                )
                output["target"] = list(
                    self.index_to_sid[int(selected)][: self.code_length]
                )
                return output
            targets = [
                self.index_to_sid[int(value)]
                for value in (sample["target"] or [])
                if value is not None and int(value) in self.index_to_sid
            ]
            if not targets:
                raise ValueError("training sample has no SID-mapped target items")
            output["independent_targets"] = [
                list(value[: self.code_length]) for value in targets
            ]
        else:
            items, labels = _raw_targets(
                sample["target"],
                sample.get("target_labels"),
            )
            output["target_items"] = items
            output["target_item_labels"] = labels
        return output

    def _read_events(self, client_id: Any, before: Any) -> pl.DataFrame:
        """Read client history strictly before the cutoff; unknown clients return no events."""

        if self.event_store_path is not None:
            if self._reader is None:
                assert self.event_store_backend is not None
                self._reader = EventStore(
                    self.event_store_path,
                    backend=self.event_store_backend,
                ).open_reader(None)  # Each pathway applies its own time/count limit.
            try:
                return self._reader.read(str(client_id), before=before)
            except KeyError:
                return pl.DataFrame()
        if self._events_by_client is None:
            assert self.events is not None
            self._events_by_client = {
                str(key): values
                for (key,), values in self.events.group_by(
                    "client_id", maintain_order=True
                )
            }
        values = self._events_by_client.get(str(client_id))
        if values is None:
            return pl.DataFrame()
        return values.filter(pl.col("timestamp") < before)

    def _compose_input(self, name: str, events: pl.DataFrame, sample: dict) -> dict:
        """Route one named input and return aligned values with padding metadata."""
        description = self.config.input(name)
        if description.type == "context":
            context = sample.get("context") or {}
            rows = [{feature: context.get(feature) for feature in description.features}]
            daily_recency = [0]
        else:
            frame = select_sequence(
                self.config, name, events, sample_timestamp=sample["timestamp"]
            )
            # Context conditions existing event tokens without adding encoder positions.
            context_features = [
                feature
                for feature in description.features
                if self.config.features[feature].located_in.context
                and not self.config.features[feature].located_in.event
            ]
            if context_features and not frame.is_empty():
                context = sample["context"]
                frame = frame.with_columns(
                    pl.lit(context[feature]).alias(feature)
                    for feature in context_features
                )
            maximum_days = (
                duration_days(
                    self.config.max_duration_per_sequence,
                    field_name="max_duration_per_sequence",
                )
                if self.config.max_duration_per_sequence is not None
                else 0
            )
            daily_recency = [
                min(
                    (sample["timestamp"].date() - timestamp.date()).days + 1,
                    maximum_days + 1,
                )
                for timestamp in (
                    frame["timestamp"].to_list() if not frame.is_empty() else []
                )
            ]
            rows = (
                frame.select(description.features).to_dicts()
                if not frame.is_empty()
                else []
            )
            sid_features = [
                feature
                for feature in description.features
                if self.feature_types[feature] == "semantic_id"
                and feature not in description.optional_features
            ]
            retained = [
                (row, recency)
                for row, recency in zip(rows, daily_recency)
                if all(row[feature] is not None for feature in sid_features)
            ]
            rows = [row for row, _ in retained]
            daily_recency = [recency for _, recency in retained]
            if not sid_features and all(
                self.feature_types[feature] == "id" for feature in description.features
            ):
                rows = [row for row in rows if any(row.values())]
        features = {}
        for feature in description.features:
            kind = self.feature_types[feature]
            if kind == "semantic_id":
                encoder = self.config.encoders[self.config.features[feature].encoder]
                width = int(encoder.embedder["prefix_length"])
                values = [
                    list(row[feature][:width])
                    if row[feature] is not None
                    else [0] * width
                    for row in rows
                ]
            else:
                width = 1
                cast = float if kind in {"numeric", "ple"} else int
                values = [[cast(row[feature] or 0)] for row in rows]
            features[feature] = {
                "values": values,
                "width": width,
                "dtype": "float32" if kind in {"numeric", "ple"} else "long",
            }
        return {
            "features": features,
            "daily_recency": daily_recency,
            "padding_length": description.max_events_per_sequence
            if self.source_position_layout == "fixed" and description.type == "sequence"
            else 1,
        }


def _raw_targets(
    value: Any,
    raw_labels: Any | None = None,
) -> tuple[list[Any], list[float]]:
    """Return aligned items and labels, treating missing labels as relevance one."""

    items = list(value or [])
    labels = (
        [float(label) for label in raw_labels]
        if raw_labels is not None
        else [1.0] * len(items)
    )
    if len(labels) != len(items):
        raise ValueError("target and target_labels must have equal lengths")
    return items, labels


def _loader(data: TorchDataset, config: Any) -> DataLoader:
    """Build a DataLoader with OneRec collation and persistent nonzero workers."""

    return DataLoader(
        data,
        batch_size=config.batch_size,
        shuffle=getattr(config, "shuffle", False),
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=onerec_collate_fn,
        persistent_workers=config.num_workers > 0,
    )
