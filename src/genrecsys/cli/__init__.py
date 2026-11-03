"""Composition root for the GenRecSys command-line interface."""

from __future__ import annotations

import logging
import multiprocessing
import os
from collections.abc import Iterable

import torch
from cyclopts import App

from genrecsys.cli import event_hub, inference, prepare_shards, train


def _logging_level() -> int:
    """Emit application logs only from the global main distributed process."""

    return (
        logging.INFO if int(os.environ.get("RANK", "0")) == 0 else logging.CRITICAL + 1
    )


def run_app(argv: Iterable[str] | None = None) -> None:
    """Run the GenRecSys CLI for events, preparation, training and inference."""

    multiprocessing.set_start_method("spawn", force=True)
    torch.multiprocessing.set_start_method("spawn", force=True)
    logging.basicConfig(
        level=_logging_level(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    app = App()
    app.command(event_hub.app, name="event-hub")
    app.command(train.app, name="train")
    app.command(inference.app, name="inference")
    app.command(prepare_shards.app, name="prepare-shards")
    app(argv)


__all__ = ["run_app"]
