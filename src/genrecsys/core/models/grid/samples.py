"""Compose full-SID GRID samples from one already prepared history store."""

from __future__ import annotations

import hashlib
import random
from datetime import UTC
from functools import partial
from numbers import Integral
from typing import Any

import polars as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from genrecsys.core.config import Config
from genrecsys.core.event_store import EventStore
from genrecsys.core.inputs import select_sequence
from genrecsys.core.models.grid import config as grid_config
from genrecsys.core.models.grid.collate import grid_collate_fn
from genrecsys.core.sample_table import MappedSamples


class Samples(TorchDataset):
    """Read the configured history and keep independent target SIDs or raw evaluation IDs."""

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
        user_id_buckets: int | None = None,
        user_id_column: str | None = None,
        user_id_hash: str | None = None,
    ) -> None:
        """Configure exactly one event source; defer opening its reader until first use."""

        if (event_store is None) == (events is None):
            raise ValueError("provide exactly one of event_store or events")
        if (
            not isinstance(code_length, int)
            or isinstance(code_length, bool)
            or code_length <= 0
        ):
            raise ValueError("SID code_length must be a positive integer")
        self.samples, self.config, self.code_length = samples, config, code_length
        self.history_name = grid_config.history_input(config)
        self.target_feature = config.task.objective.target
        self.max_history_items = config.model.params.get("max_history_items")
        self.augmentation = (
            grid_config.augmentation_parameters(config) if train else None
        )
        self.user_id_buckets = (
            grid_config.model_parameters(config)["user_id_buckets"]
            if user_id_buckets is None
            else user_id_buckets
        )
        self.user_id_column = (
            config.model.params.get("user_id_column", "client_id")
            if user_id_column is None
            else user_id_column
        )
        self.user_id_hash = (
            config.model.params.get("user_id_hash", "sha256")
            if user_id_hash is None
            else user_id_hash
        )
        if (
            self.user_id_buckets
            and self.user_id_column != "client_id"
            and len(samples)
            and self.user_id_column not in samples.row(0, named=True)
        ):
            raise ValueError(
                f"GRID user tokens require sample column {self.user_id_column!r}"
            )
        self.event_store_path = event_store.path if event_store is not None else None
        self.event_store_backend = (
            event_store.backend if event_store is not None else None
        )
        self.events = events
        self.index_to_sid = index_to_sid or {}
        self.train = train
        self._reader = None
        self._events_by_client: dict[str, pl.DataFrame] | None = None

    def __len__(self) -> int:
        """Return the number of basis sample rows."""

        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Keep sample indices separate from the external client ID used for user tokens."""

        sample = self.samples.row(index, named=True)
        events = self._read_events(
            sample["client_id"], sample.get("history_timestamp", sample["timestamp"])
        )
        output = {
            "user_id": int(sample.get("__sample_index", index)),
            "input_ids": self._compose_inputs(events, sample),
            "code_length": self.code_length,
            "reference_time": (
                sample["timestamp"].replace(tzinfo=UTC)
                if sample["timestamp"].tzinfo is None
                else sample["timestamp"]
            ).timestamp(),
        }
        if self.user_id_buckets:
            output["user_bucket"] = _user_bucket(
                sample[self.user_id_column],
                self.user_id_buckets,
                hashing=self.user_id_hash,
            )
        if "target" not in sample:
            if self.train:
                raise ValueError("training sample must contain target")
            return output
        if not self.train:
            output["target_items"], output["target_item_labels"] = _raw_targets(
                sample["target"], sample.get("target_labels")
            )
            return output
        targets, labels = _raw_targets(sample["target"], sample.get("target_labels"))
        if not targets or any(
            value is None or int(value) not in self.index_to_sid for value in targets
        ):
            raise ValueError("every training target must have a frozen SID mapping")
        mode = self.config.task.objective.target_mode
        if self.augmentation is not None:
            if len(targets) != 1:
                raise ValueError(
                    "contiguous augmentation requires one chronological next-item target per row"
                )
            output["training_sequence"] = [
                *output.pop("input_ids"),
                *self._sid_tokens(self.index_to_sid[int(targets[0])]),
            ]
            return output
        if mode in {"random", "label_weighted"}:
            weights = (
                [1.0] * len(targets)
                if mode == "random"
                else [max(label, 0) for label in labels]
            )
            selected = (
                random.choices(targets, weights=weights, k=1)[0]
                if sum(weights)
                else random.choice(targets)
            )
            output["target"] = self._sid_tokens(self.index_to_sid[int(selected)])
        elif mode == "all_targets_independent":
            output["independent_targets"] = [
                self._sid_tokens(self.index_to_sid[int(value)]) for value in targets
            ]
        else:
            raise ValueError("unsupported GRID target_mode")
        return output

    def _compose_inputs(self, events: pl.DataFrame, sample: dict) -> list[int]:
        """Apply the shared selector before discarding null SIDs, preserving source semantics."""

        if sample.get("history_start_timestamp") is not None and not events.is_empty():
            # Materialized windows need both bounds: a common target/cutoff with
            # different starts must not silently yield the same history.
            events = events.filter(
                pl.col("timestamp") >= sample["history_start_timestamp"]
            )
        frame = select_sequence(
            self.config, self.history_name, events, sample_timestamp=sample["timestamp"]
        )
        tokens = (
            []
            if frame.is_empty()
            else [
                token
                for sid in frame[self.target_feature].to_list()
                if sid is not None
                for token in self._sid_tokens(sid)
            ]
        )
        if self.max_history_items is not None and self.augmentation is None:
            tokens = tokens[-self.max_history_items * self.code_length :]
        return tokens

    def _sid_tokens(self, sid) -> list[int]:
        """Keep every SID level, including the collision level, without truncating tokens."""

        if len(sid) != self.code_length:
            raise ValueError(f"GRID SID must contain exactly {self.code_length} tokens")
        return list(sid)

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
        if self.events is not None and self.events.is_empty():
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


def _user_bucket(client_id: Any, buckets: int, *, hashing: str = "sha256") -> int:
    """Hash external identities consistently across processes, workers and restarts.

    SHA-256 preserves the existing external-client behavior. Explicit modulo
    recipes use an integer identity such as Amazon's original grid_user_id,
    independently of the EventStore history-segment key and sample row index.
    """

    if client_id is None:
        raise ValueError("GRID user tokens require a non-null user identity")
    if hashing == "modulo":
        if isinstance(client_id, bool) or not isinstance(client_id, (Integral, str)):
            raise ValueError("GRID modulo user identity must be an integer")
        try:
            identity = int(client_id)
        except ValueError as error:
            raise ValueError("GRID modulo user identity must be an integer") from error
        return identity % buckets
    if hashing != "sha256":
        raise ValueError("GRID user identity hashing must be sha256 or modulo")
    return (
        int.from_bytes(hashlib.sha256(str(client_id).encode("utf-8")).digest(), "big")
        % buckets
    )


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
    """Build a DataLoader with GRID collation and persistent nonzero workers."""

    return DataLoader(
        data,
        batch_size=config.batch_size,
        shuffle=getattr(config, "shuffle", False),
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=partial(
            grid_collate_fn,
            augmentation=getattr(data, "augmentation", None),
            max_history_items=getattr(data, "max_history_items", None),
        ),
        persistent_workers=config.num_workers > 0,
    )


def _to_device(value: Any, device: torch.device) -> Any:
    """Move nested dict tensors to the device, leaving other values unchanged."""

    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value
