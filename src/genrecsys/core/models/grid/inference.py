"""Reusable GRID inference and constrained top-k decoding."""

from __future__ import annotations

import math
import os
from collections.abc import Iterator
from typing import Any

import polars as pl
import torch

from genrecsys.core.checkpoint import Trained
from genrecsys.core.event_store import EventsReader, EventStore
from genrecsys.core.models.grid import architecture
from genrecsys.core.models.grid.architecture import GRID
from genrecsys.core.models.grid.catalog import (
    _allowed_codes,
    _catalog,
    _generation_catalog,
    _item_creation_times,
    _item_preprocessor,
    _items_by_sid,
)
from genrecsys.core.models.grid.config import (
    generation_constraint,
    generation_renormalization,
)
from genrecsys.core.models.grid.samples import Samples, _loader, _to_device
from genrecsys.core.sample_table import MappedSamples
from genrecsys.core.tasks import retrieval


class InferenceSession:
    """Reuse GRID and the in-process EventStore reader across inference chunks."""

    def __init__(
        self,
        checkpoint: Trained,
        state: Any,
        /,
        *,
        artifacts: retrieval.Artifacts | None = None,
    ) -> None:
        """Load the model on the inference device and prepare generation constraints.

        Optional artifacts replace the checkpoint catalog. Reuse the session across
        batches to avoid reloading model weights and candidate constraints.
        """

        self.checkpoint = checkpoint
        self.item = _item_preprocessor(checkpoint.prepared)
        self.device = _inference_device()
        self.model = architecture.build_model(state.model_config).to(self.device)
        self.model.load_state_dict(state.state_dict)
        self.model.eval()
        # Identity is part of the trained model's input semantics. Legacy states
        # have no explicit settings and keep SHA256(client_id), even with an
        # inference config overlay that selects the new Amazon recipe.
        self.user_id_column = state.model_config.get("user_id_column", "client_id")
        self.user_id_hash = state.model_config.get("user_id_hash", "sha256")
        catalog = (
            _catalog(artifacts, self.item) if artifacts is not None else state.catalog
        )
        eligible = _generation_catalog(catalog)
        self.allowed_codes = _allowed_codes(eligible)
        self.items_by_sid = _items_by_sid(eligible)
        self.item_created_at = _item_creation_times(eligible)
        self.item_id_dtype = catalog.schema["item_id"]
        predict = checkpoint.prepared.config.inference.predict_kwargs
        self.k = int(predict["k"])
        self.beam_size = int(predict["beam_size"])
        self.constraint = generation_constraint(checkpoint.prepared.config)
        self.renormalize_logits = generation_renormalization(checkpoint.prepared.config)
        self._event_store: EventStore | None = None
        self._event_reader: EventsReader | None = None

    def iter_predictions(
        self,
        samples: pl.DataFrame | MappedSamples,
        events: pl.DataFrame | EventStore,
        /,
    ) -> Iterator[pl.DataFrame]:
        """Yield __sample_index/prediction batches without collecting all outputs.

        Materialize event DataFrames into an EventStore if needed. Use inference_mode
        and configured autocast; with no workers, reuse the history reader.
        """

        if not len(samples):
            return
        if not isinstance(events, EventStore) and not events.is_empty():
            events = events.sort(
                ["client_id", "timestamp"], descending=[False, True]
            ).with_columns(pl.int_range(pl.len()).over("client_id").alias("position"))
            event_store = EventStore(
                backend=self.checkpoint.prepared.config.inference.event_store_backend
            )
            with event_store.open_writer() as writer:
                writer.write(events)
            events = event_store
        source = (
            {"event_store": events}
            if isinstance(events, EventStore)
            else {"events": events}
        )
        data = Samples(
            samples,
            self.checkpoint.prepared.config,
            code_length=self.model.code_len,
            user_id_buckets=self.model.user_id_buckets,
            user_id_column=self.user_id_column,
            user_id_hash=self.user_id_hash,
            **source,
        )
        loader_config = self.checkpoint.prepared.config.inference.dataloader
        if isinstance(events, EventStore) and loader_config.num_workers == 0:
            # Keep the index open across chunks; worker processes own their readers.
            if self._event_store is not events:
                self._event_reader = events.open_reader(None)
                self._event_store = events
            data._reader = self._event_reader
        loader = _loader(data, loader_config)
        prediction_dtype = pl.List(
            pl.Struct(
                {
                    "item": self.item_id_dtype,
                    "score": pl.Float32,
                }
            )
        )
        with torch.inference_mode():
            for batch in loader:
                indices = list(batch["user_id"])
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.checkpoint.prepared.config.inference.mixed_precision
                    == "bf16",
                ):
                    rankings = _predict_batch(
                        self.model,
                        _to_device(batch, self.device),
                        self.allowed_codes,
                        self.items_by_sid,
                        k=self.k,
                        beam_size=self.beam_size,
                        item_created_at=self.item_created_at,
                        constraint=self.constraint,
                        renormalize_logits=self.renormalize_logits,
                    )
                yield pl.DataFrame(
                    {
                        "__sample_index": indices,
                        "prediction": pl.Series(
                            "prediction",
                            rankings,
                            dtype=prediction_dtype,
                        ),
                    }
                )


def _inference_device() -> torch.device:
    """Select CPU or the LOCAL_RANK CUDA device and set the active CUDA device."""

    if not torch.cuda.is_available():
        return torch.device("cpu")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


@torch.inference_mode()
def _predict_batch(
    model: GRID,
    batch: dict[str, Any],
    allowed_codes: tuple[tuple[int, ...], ...],
    items_by_sid: dict[tuple[int, ...], list[Any]],
    *,
    k: int,
    beam_size: int,
    item_created_at: dict[Any, float] | None = None,
    constraint: str = "prefix",
    renormalize_logits: bool = False,
) -> list[list[dict[str, Any]]]:
    """Expand generated full SIDs into eligible unique items and filter future items.

    Free search may produce invalid catalog codes. Drop those without filling the
    ranking from another model or changing the configured search budget.
    """

    if not allowed_codes:
        return [[] for _ in batch["user_id"]]
    sequences, sequence_scores = model.generate_codes_with_scores(
        batch=batch,
        num_beams=beam_size,
        code_len=model.code_len,
        allowed_codes=allowed_codes,
        constraint=constraint,
        renormalize_logits=renormalize_logits,
    )
    returned_beams = len(sequences) // len(batch["user_id"])
    codes = sequences[:, 1 : 1 + model.code_len].view(
        len(batch["user_id"]),
        returned_beams,
        model.code_len,
    )
    scores = sequence_scores.view(len(batch["user_id"]), returned_beams)
    output: list[list[dict[str, Any]]] = []
    cutoffs = batch["reference_time"].cpu().tolist() if item_created_at else None
    for index, (user_codes, user_scores) in enumerate(
        zip(
            codes.cpu().tolist(),
            scores.float().cpu().tolist(),
        )
    ):
        ranking: list[dict[str, Any]] = []
        seen: set[Any] = set()
        for code, score in zip(user_codes, user_scores):
            if not math.isfinite(score):
                continue
            for item_id in items_by_sid.get(tuple(code), []):
                if item_id in seen:
                    continue
                if (
                    item_created_at
                    and item_created_at.get(item_id, -math.inf) > cutoffs[index]
                ):
                    continue
                seen.add(item_id)
                ranking.append({"item": item_id, "score": float(score)})
                if len(ranking) == k:
                    break
            if len(ranking) == k:
                break
        output.append(ranking)
    return output
