"""Native OneRec model plugin for GenRecSys."""

from genrecsys.core.models.onerec.implementation import (
    State,
    create_inference_session,
    data_requirements,
    fit_model,
    implementation,
    validate_prepared,
)

__all__ = [
    "State",
    "create_inference_session",
    "data_requirements",
    "fit_model",
    "implementation",
    "validate_prepared",
]
