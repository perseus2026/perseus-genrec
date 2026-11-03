"""Reusable OneRec inference and constrained top-k decoding."""

from __future__ import annotations

import math
import os
from collections.abc import Iterator
from typing import Any

import polars as pl
import torch

from genrecsys.core.checkpoint import Trained
from genrecsys.core.event_store import EventsReader, EventStore
from genrecsys.core.models.onerec.architecture import OneRec
from genrecsys.core.models.onerec.catalog import (
    _allowed_codes,
    _catalog,
    _item_creation_times,
    _item_preprocessor,
    _items_by_sid,
)
from genrecsys.core.models.onerec.samples import Samples, _loader
from genrecsys.core.models.training import _to_device
from genrecsys.core.tasks import retrieval


class InferenceSession:
    """Reuse OneRec and the in-process EventStore reader across inference chunks."""

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
        batches.
        """

        self.checkpoint = checkpoint
        self.item = _item_preprocessor(checkpoint.prepared)
        self.device = _inference_device()
        self.model = OneRec(state.model_config).to(self.device)
        self.model.load_state_dict(state.state_dict)
        self.model.eval()
        catalog = (
            _catalog(artifacts, self.item, self.model.code_len)
            if artifacts is not None
            else state.catalog
        )
        self.allowed_codes = _allowed_codes(catalog)
        self.items_by_sid = _items_by_sid(catalog)
        self.item_created_at = _item_creation_times(catalog)
        self.item_id_dtype = catalog.schema["item_id"]
        predict = checkpoint.prepared.config.inference.predict_kwargs
        self.k = int(predict["k"])
        self.beam_size = int(predict["beam_size"])
        self._event_store: EventStore | None = None
        self._event_reader: EventsReader | None = None

    def iter_predictions(
        self,
        samples: pl.DataFrame,
        events: pl.DataFrame | EventStore,
        /,
    ) -> Iterator[pl.DataFrame]:
        """Yield __sample_index/prediction batches without collecting all outputs.

        Materialize event DataFrames into an EventStore if needed. Use inference_mode
        and configured autocast; with no workers, reuse the history reader.
        """

        if not isinstance(events, EventStore) and not events.is_empty():
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
    model: OneRec,
    batch: dict[str, Any],
    allowed_codes: tuple[tuple[int, ...], ...],
    items_by_sid: dict[tuple[int, ...], list[Any]],
    *,
    k: int,
    beam_size: int,
    item_created_at: dict[Any, float] | None = None,
) -> list[list[dict[str, Any]]]:
    """Predict top-k items with constrained SID beam search.

    Plain OneRec expands SID codes into unique item IDs and filters future items
    against reference_time when creation timestamps are available.
    """

    sequences, sequence_scores = model.generate_codes_with_scores(
        batch=batch,
        num_beams=beam_size,
        code_len=model.code_len,
        allowed_codes=allowed_codes,
    )
    codes = sequences[:, 1 : 1 + model.code_len].view(
        len(batch["user_id"]),
        beam_size,
        model.code_len,
    )
    scores = sequence_scores.view(len(batch["user_id"]), beam_size)
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
