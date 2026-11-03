"""Read native prepared lifelong histories without rebuilding datasets or SIDs."""

from dataclasses import replace
from typing import Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

from genrecsys.core.checkpoint import Prepared
from genrecsys.core.config import Config
from genrecsys.core.event_store import EventStore
from genrecsys.core.inputs import select_sequence
from genrecsys.core.inputs.collate import _pad_input
from genrecsys.core.models.hstu.config import context_features, item_preprocessor
from genrecsys.core.sample_table import MappedSamples


def vocabulary(prepared: Prepared) -> pl.DataFrame:
    """Assign stable item tokens independently of the particular SID codebook."""
    item = item_preprocessor(prepared)
    return (
        item.mapping.select(
            pl.col(item.item_id_column).alias("item_id"),
            pl.col(item.sid_column).alias("sid"),
        )
        .sort("item_id")
        .with_row_index("item_token", offset=1)
    )


def catalog(items: pl.DataFrame, tokens: pl.DataFrame) -> pl.DataFrame:
    """Keep the exact split candidate universe and map it to the shared item table."""
    result = items.select("item_id").join(
        tokens.select("item_id", "item_token"),
        on="item_id",
        how="left",
        maintain_order="left",
        validate="1:1",
    )
    if result.is_empty() or result.null_count().row(0) != (0, 0):
        raise ValueError(
            "HSTU candidate catalog is empty or misses frozen item identities"
        )
    return result


def sid_lookup(tokens: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build a compact sorted SID index, avoiding a Python tuple dict per item."""
    width = tokens["sid"].list.len().first()
    codes = np.ascontiguousarray(tokens["sid"].explode().to_numpy(), dtype=np.int64)
    keys = codes.view(np.dtype(f"V{int(width) * 8}"))
    order = np.argsort(keys)
    return keys[order], tokens["item_token"].to_numpy()[order]


def item_tokens(values: pl.Series, lookup: tuple[np.ndarray, np.ndarray]) -> list[int]:
    """Recover exact identities from full SIDs; missing item attributes stay zero."""
    if len(values) == 0:
        return []
    keys, tokens = lookup
    width = keys.dtype.itemsize // 8
    missing = values.is_null().to_numpy()
    codes = np.ascontiguousarray(
        values.fill_null([0] * width).explode().to_numpy(),
        dtype=np.int64,
    )
    query = codes.view(keys.dtype)
    positions = np.searchsorted(keys, query)
    matches = (positions < len(keys)) & (
        keys[positions.clip(max=len(keys) - 1)] == query
    )
    if not np.all(matches | missing):
        raise ValueError(
            "prepared HSTU history contains a SID absent from its frozen mapping"
        )
    output = np.zeros(len(values), dtype=np.int64)
    output[~missing] = tokens[positions[~missing]]
    return output.tolist()


class Samples(Dataset):
    """Compose lifelong and user context; labels use candidates, holdout IDs stay raw."""

    def __init__(
        self,
        samples: pl.DataFrame | MappedSamples,
        config: Config,
        lookup,
        *,
        event_store: EventStore | None = None,
        events: pl.DataFrame | None = None,
        train_targets: dict[int, int] | None = None,
    ):
        if (event_store is None) == (events is None):
            raise ValueError("provide exactly one event source")
        self.samples, self.config, self.lookup = samples, config, lookup
        self.context_features = context_features(config)
        history = config.inputs["lifelong"]
        # Respect both the prepared-history limit and the model's readout budget.
        limit = min(history.max_events_per_sequence, config.model.params["max_len"] - 1)
        priority = history.max_priority_events_per_sequence
        self.history_config = replace(
            config,
            inputs={
                **config.inputs,
                "lifelong": replace(
                    history,
                    max_events_per_sequence=limit,
                    max_priority_events_per_sequence=(
                        min(priority, limit) if priority is not None else None
                    ),
                ),
            },
        )
        self.path = event_store.path if event_store is not None else None
        self.backend = event_store.backend if event_store is not None else None
        self.events = events
        self.train_targets = train_targets
        self._reader = None
        self._by_client = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples.row(index, named=True)
        cutoff = sample.get("history_timestamp", sample["timestamp"])
        events = self._read_events(sample["client_id"], cutoff)
        history = self.config.inputs["lifelong"]
        frame = select_sequence(
            self.history_config,
            "lifelong",
            events,
            sample_timestamp=sample["timestamp"],
        )
        target = self.config.task.objective.target
        if len(frame) and target not in history.optional_features:
            frame = frame.filter(pl.col(target).is_not_null())
        length = len(frame)
        features = {}
        for name in history.features:
            kind = self.config.encoders[self.config.features[name].encoder].type
            if not length:
                values = []
            elif name == self.config.task.objective.target:
                values = item_tokens(frame[name], self.lookup)
            else:
                values = frame[name].fill_null(0).to_list()
            features[name] = {
                "values": [[value] for value in values],
                "width": 1,
                "dtype": "float32" if kind in {"ple", "numeric"} else "long",
            }
        result = {
            "user_id": int(sample.get("__sample_index", index)),
            "history": {
                "features": features,
                "padding_length": 1,
                "timestamps": (
                    frame.select(
                        (
                            (pl.col("timestamp") - pl.lit(sample["timestamp"]))
                            .dt.total_nanoseconds()
                            .cast(pl.Float64)
                            / 1e9
                        ).alias("seconds")
                    )["seconds"].to_list()
                    if length
                    else []
                ),
            },
        }
        if self.context_features:
            context = sample.get("context") or {}
            result["context"] = {
                "features": {
                    name: {
                        "values": [[context.get(name) or 0]],
                        "width": 1,
                        "dtype": (
                            "float32"
                            if self.config.encoders[
                                self.config.features[name].encoder
                            ].type
                            in {"numeric", "ple"}
                            else "long"
                        ),
                    }
                    for name in self.context_features
                },
                "padding_length": 1,
            }
        if self.train_targets is not None:
            targets = sample["target"] or []
            if not targets or any(value not in self.train_targets for value in targets):
                raise ValueError(
                    "HSTU training sample has no valid catalog-mapped targets"
                )
            result["target"] = [self.train_targets[value] for value in targets]
        elif "target" in sample:
            result["target_items"] = sample["target"]
            result["target_item_labels"] = sample.get("target_labels") or [1.0] * len(
                sample["target"]
            )
        return result

    def _read_events(self, client_id: Any, before) -> pl.DataFrame:
        if self.path is not None:
            if self._reader is None:
                self._reader = EventStore(self.path, backend=self.backend).open_reader(
                    None
                )
            try:
                return self._reader.read(str(client_id), before=before)
            except KeyError:
                return pl.DataFrame()
        if self._by_client is None:
            self._by_client = {
                str(key): values
                for (key,), values in self.events.group_by(
                    "client_id", maintain_order=True
                )
            }
        values = self._by_client.get(str(client_id))
        return (
            pl.DataFrame()
            if values is None
            else values.filter(pl.col("timestamp") < before)
        )


def collate(batch):
    """Pad histories by event existence, including brand-only auxiliary events."""
    output = {
        "user_id": [row["user_id"] for row in batch],
        "history": _pad_input([row["history"] for row in batch]),
    }
    shape = output["history"]["attention_mask"].shape
    timestamps = torch.zeros(shape, dtype=torch.float64)
    for index, row in enumerate(batch):
        values = row["history"]["timestamps"]
        if values:
            timestamps[index, -len(values) :] = torch.tensor(
                values, dtype=torch.float64
            )
    output["history"]["timestamps"] = timestamps
    if "context" in batch[0]:
        output["context"] = _pad_input([row["context"] for row in batch])
    if "target" in batch[0]:
        width = max(len(row["target"]) for row in batch)
        output["target"] = torch.full((len(batch), width), -100, dtype=torch.long)
        for index, row in enumerate(batch):
            output["target"][index, : len(row["target"])] = torch.tensor(row["target"])
    elif "target_items" in batch[0]:
        output["target_items"] = [row["target_items"] for row in batch]
        output["target_item_labels"] = [row["target_item_labels"] for row in batch]
    return output


def loader(data, config):
    return DataLoader(
        data,
        batch_size=config.batch_size,
        shuffle=getattr(config, "shuffle", False),
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
        collate_fn=collate,
    )
