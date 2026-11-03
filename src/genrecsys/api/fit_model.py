"""Native model-fitting API."""

from __future__ import annotations

from genrecsys.core.checkpoint import Prepared, Trained
from genrecsys.core.dataset import Dataset
from genrecsys.core.models import get


def fit_model(checkpoint: Prepared, dataset: Dataset, /) -> Trained:
    """Validate prepared data, fit the configured model, and return Trained.

    Reuse the existing preprocessors and resources; the caller saves the checkpoint.
    """

    checkpoint.config.validate()
    implementation = get(checkpoint.config.model.type)
    if implementation.validate_prepared is not None:
        implementation.validate_prepared(checkpoint, dataset)
    model = implementation.fit_model(checkpoint, dataset)
    return Trained(checkpoint, model)


__all__ = ["fit_model"]
