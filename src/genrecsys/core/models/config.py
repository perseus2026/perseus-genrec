"""Execution validation shared by native retrieval models."""

import math
from typing import Any

from genrecsys.core.config import Config, DataLoader


def _exact_keys(value: Any, keys: set[str], path: str) -> None:
    """Require an exact mapping key set and identify the config path on failure."""

    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{path} must contain exactly {sorted(keys)}")


def _number(value: Any, path: str) -> float:
    """Parse a finite nonnegative float, rejecting booleans."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{path} must be a finite non-negative number")
    return float(value)


def validate_execution(config: Config) -> None:
    """Validate execution settings shared by native models."""

    for name, settings in (
        ("training", config.training),
        ("inference", config.inference),
    ):
        if settings.mixed_precision not in {"no", "bf16"}:
            raise ValueError(f"{name}.mixed_precision must be no or bf16")

    training = config.training
    _positive(
        "training.gradient_accumulation_steps", training.gradient_accumulation_steps
    )
    if training.gradient_clip_norm is not None:
        _positive("training.gradient_clip_norm", training.gradient_clip_norm)
    for name in (
        "num_epochs",
        "log_every_n_train_steps",
        "validation_every_n_epochs",
    ):
        _positive(f"training.{name}", getattr(training, name))
    _validate_dataloader(training.dataloader, "training.dataloader")
    _validate_dataloader(
        training.validation_dataloader,
        "training.validation_dataloader",
    )
    if training.optimizer.type not in {"adam", "adamw"}:
        raise ValueError("optimizer must be adam or adamw")
    if "lr" in training.optimizer.params:
        learning_rate = training.optimizer.params["lr"]
        if not isinstance(learning_rate, (int, float)) or isinstance(
            learning_rate, bool
        ):
            raise TypeError("training.optimizer.params.lr must be numeric")
        _positive("training.optimizer.params.lr", learning_rate)
    if training.scheduler.type == "constant":
        if training.scheduler.params:
            raise ValueError("constant scheduler takes no parameters")
    elif training.scheduler.type == "cosine_warmup":
        _exact_keys(
            training.scheduler.params,
            {"warmup_ratio", "min_lr_ratio"},
            "training.scheduler.params",
        )
        for name in ("warmup_ratio", "min_lr_ratio"):
            value = _number(
                training.scheduler.params[name], f"training.scheduler.params.{name}"
            )
            if value > 1 or (name == "warmup_ratio" and value == 1):
                raise ValueError(f"scheduler {name} is outside its valid range")
    else:
        raise ValueError("scheduler must be constant or cosine_warmup")
    if not training.tracker.type:
        raise ValueError("training.tracker.type must not be empty")

    early = training.early_stopping
    if early is not None:
        if early.metric not in config.task.metrics:
            raise ValueError("early-stopping metric must reference task.metrics")
        _positive("training.early_stopping.patience", early.patience)

    inference = config.inference
    _validate_dataloader(inference.dataloader, "inference.dataloader")


def _validate_dataloader(value: DataLoader, field_name: str) -> None:
    """Validate DataLoader batch size and worker count."""

    if not isinstance(value.batch_size, int) or isinstance(value.batch_size, bool):
        raise TypeError(f"{field_name}.batch_size must be an integer")
    if not isinstance(value.num_workers, int) or isinstance(value.num_workers, bool):
        raise TypeError(f"{field_name}.num_workers must be an integer")
    _positive(f"{field_name}.batch_size", value.batch_size)
    if value.num_workers < 0:
        raise ValueError(f"{field_name}.num_workers must be non-negative")


def _positive(name: str, value: float) -> None:
    """Reject a nonpositive value with an error naming the parameter."""

    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
