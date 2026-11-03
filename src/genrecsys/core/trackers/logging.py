"""Standard-library logging tracker."""

from __future__ import annotations

import logging

from genrecsys.core.trackers.base import Tracker

logger = logging.getLogger(__name__)


class LoggingTracker(Tracker):
    def log_metrics(
        self,
        metrics: dict[str, dict[str, float]],
        /,
        *,
        iteration: int,
    ) -> None:
        """Log grouped metrics and the step through the standard logger."""

        for metric_name, group_metrics in metrics.items():
            for group_name, metric_value in group_metrics.items():
                logger.info(
                    "[iteration=%d] %s/%s = %s",
                    iteration,
                    metric_name,
                    group_name,
                    metric_value,
                )
