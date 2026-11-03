"""Python APIs called by the thin GenRecSys lifecycle CLI."""

from genrecsys.api.evaluate_predictions import evaluate_predictions
from genrecsys.api.event_hub import add_events, delete_events, list_events
from genrecsys.api.fit_baseline import fit_baseline
from genrecsys.api.fit_model import fit_model
from genrecsys.api.make_predictions import (
    create_inference_session,
    distribute_samples,
    iter_predictions,
    make_predictions,
)
from genrecsys.api.prepare_dataset import prepare_dataset

__all__ = [
    "add_events",
    "create_inference_session",
    "delete_events",
    "distribute_samples",
    "evaluate_predictions",
    "fit_baseline",
    "fit_model",
    "iter_predictions",
    "list_events",
    "make_predictions",
    "prepare_dataset",
]
