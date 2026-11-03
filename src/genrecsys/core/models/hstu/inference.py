"""Exact item retrieval using the existing partitioned inference lifecycle."""

import os

import polars as pl
import torch

from genrecsys.core.event_store import EventStore
from genrecsys.core.models.hstu.architecture import HSTU
from genrecsys.core.models.hstu.data import (
    Samples,
    catalog,
    loader,
    sid_lookup,
    vocabulary,
)
from genrecsys.core.models.training import _to_device


@torch.inference_mode()
def rankings(model, batch, tokens, items: pl.Series, k: int):
    """Return external item IDs, gathering only the selected catalog rows to Python."""
    scores, indices = model.topk(batch, tokens, k)
    selected = items.gather(indices.cpu().flatten().numpy()).to_list()
    width = indices.shape[1]
    return [
        [
            {"item": selected[row * width + column], "score": score}
            for column, score in enumerate(values)
        ]
        for row, values in enumerate(scores.cpu().tolist())
    ]


class InferenceSession:
    """Keep model, frozen SID lookup and the EventStore reader between chunks."""

    def __init__(self, checkpoint, state, *, artifacts=None):
        self.config = checkpoint.prepared.config
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(rank)
            self.device = torch.device("cuda", rank)
        self.model = HSTU(state.model_config).to(self.device)
        self.model.load_state_dict(state.state_dict)
        self.model.eval()
        vocab = vocabulary(checkpoint.prepared)
        self.lookup = sid_lookup(vocab)
        candidates = (
            catalog(artifacts.items, vocab) if artifacts is not None else state.catalog
        )
        self.items = candidates["item_id"]
        self.tokens = torch.tensor(
            candidates["item_token"].to_numpy(), dtype=torch.long, device=self.device
        )
        self._store = self._reader = None

    def iter_predictions(self, samples, events):
        source = (
            {"event_store": events}
            if isinstance(events, EventStore)
            else {"events": events}
        )
        data = Samples(samples, self.config, self.lookup, **source)
        settings = self.config.inference.dataloader
        if isinstance(events, EventStore) and settings.num_workers == 0:
            if self._store is not events:
                self._reader = events.open_reader(None)
                self._store = events
            data._reader = self._reader
        dtype = pl.List(pl.Struct({"item": self.items.dtype, "score": pl.Float32}))
        with torch.inference_mode():
            for batch in loader(data, settings):
                with torch.autocast(
                    self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.config.inference.mixed_precision == "bf16",
                ):
                    predictions = rankings(
                        self.model,
                        _to_device(batch, self.device),
                        self.tokens,
                        self.items,
                        self.config.inference.predict_kwargs["k"],
                    )
                yield pl.DataFrame(
                    {
                        "__sample_index": batch["user_id"],
                        "prediction": pl.Series("prediction", predictions, dtype=dtype),
                    }
                )
