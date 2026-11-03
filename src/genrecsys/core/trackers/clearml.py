"""ClearML implementation of the training metric tracker."""

from __future__ import annotations

import clearml

from genrecsys.core.trackers.base import Tracker


class ClearMLTracker(Tracker):
    def __init__(
        self,
        *,
        project: str | None = None,
        name: str | None = None,
        reuse: bool = False,
        tags: list[str] | None = None,
    ) -> None:
        # A subprocess reporter can deadlock or silently stop delivering events
        # when it is forked after torch.distributed/CUDA initialization.
        """Create or reuse a ClearML task without automatic framework tracking.

        Disable subprocess reporting to avoid hangs after CUDA/DDP initialization.
        Task initialization contacts ClearML.
        """

        clearml.Task._report_subprocess_enabled = False
        self._task = clearml.Task.init(
            project_name=project,
            task_name=name,
            reuse_last_task_id=reuse,
            tags=tags,
            auto_connect_arg_parser=False,
            auto_connect_frameworks=False,
        )

    def log_metrics(
        self,
        metrics: dict[str, dict[str, float]],
        /,
        *,
        iteration: int,
    ) -> None:
        """Report grouped scalar metrics and request a nonblocking upload flush."""

        logger = self._task.get_logger()
        for metric_name, group_metrics in metrics.items():
            for group_name, metric_value in group_metrics.items():
                logger.report_scalar(
                    title=metric_name,
                    series=group_name,
                    value=metric_value,
                    iteration=iteration,
                )
        # Ask the asynchronous reporter to send this batch now. Do not wait for
        # uploads here: metric reporting must never stall a training rank.
        self._task.flush(wait_for_uploads=False)
