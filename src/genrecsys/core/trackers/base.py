"""Training metric tracker contract."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Tracker(ABC):
    @abstractmethod
    def log_metrics(
        self,
        metrics: dict[str, dict[str, float]],
        /,
        *,
        iteration: int,
    ) -> None:
        """Log metric -> group -> value mappings at the given step."""
        ...
