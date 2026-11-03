"""Shared training loop, optimizer and tensor utilities for native models."""

import logging
import math
import time
from typing import Any

import torch
from torch import nn

from genrecsys.core.config import Config
from genrecsys.utils.progress import log_phase

logger = logging.getLogger(__name__)


def run_training(
    model,
    optimizer,
    scheduler,
    loader,
    config,
    accelerator,
    tracker,
    validate_epoch,
    *,
    to_device,
):
    """Train with accumulation and select weights using held-out validation.

    The loop knows nothing about OneRec pathways or EventHub.
    A TIGER adapter can reuse it with its own model and prediction callback.

    A model may expose loss_normalizer: the global denominator of its scalar
    loss, shared across ranks. Accumulate weighted losses, then divide gradients
    by the full window's denominator before clipping and stepping. This keeps
    target weighting independent of the microbatch/accumulation split.
    """
    early = config.early_stopping
    best_score = -math.inf
    best_state, history, stale_epochs = {}, [], 0
    for epoch in range(1, config.num_epochs + 1):
        epoch_steps = len(loader)
        loader.set_epoch(epoch - 1)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=accelerator.device)
        steps = torch.zeros_like(loss_sum)
        window_loss = torch.zeros_like(loss_sum)
        window_weight = torch.zeros_like(loss_sum)
        accumulation_weight = torch.zeros_like(loss_sum)
        window_steps, window_started = 0, time.perf_counter()
        train_started = time.perf_counter()
        logger.info("Starting training epoch=%s steps=%s", epoch, epoch_steps)
        for step, batch in enumerate(loader, 1):
            batch = to_device(batch, accelerator.device)
            with accelerator.accumulate(model):
                loss, _ = model(batch, labels=batch["target"])
                loss_normalizer = getattr(
                    accelerator.unwrap_model(model), "loss_normalizer", None
                )
                weight = 1 if loss_normalizer is None else loss_normalizer
                accelerator.backward(loss * weight)
                if loss_normalizer is not None:
                    accumulation_weight += weight
                    if accelerator.sync_gradients:
                        # Accelerate divides each backward by accumulation_steps;
                        # DDP averaging is already compensated in the model loss.
                        scale = (
                            accelerator.gradient_accumulation_steps
                            / accumulation_weight
                        )
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                parameter.grad.mul_(scale)
                        accumulation_weight.zero_()
                if accelerator.sync_gradients and config.gradient_clip_norm is not None:
                    accelerator.clip_grad_norm_(
                        model.parameters(), config.gradient_clip_norm
                    )
                optimizer.step()
                if (
                    accelerator.sync_gradients
                    and not accelerator.optimizer_step_was_skipped
                ):
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += loss.detach() * weight
            window_loss += loss.detach() * weight
            steps += weight
            window_weight += weight
            window_steps += 1
            if step % config.log_every_n_train_steps == 0 or step == epoch_steps:
                reduced = accelerator.reduce(
                    torch.stack((window_loss, window_weight)), reduction="sum"
                )
                if accelerator.is_main_process:
                    value = float((reduced[0] / reduced[1]).item())
                    elapsed = time.perf_counter() - window_started
                    logger.info(
                        "epoch=%s step=%s/%s loss=%.6f lr=%.6g eta_min=%.1f",
                        epoch,
                        step,
                        epoch_steps,
                        value,
                        optimizer.param_groups[0]["lr"],
                        elapsed / window_steps * (epoch_steps - step) / 60,
                    )
                    if tracker:
                        tracker.log_metrics(
                            {"train_step_loss": {"window_mean": value}},
                            iteration=(epoch - 1) * epoch_steps + step,
                        )
                window_loss.zero_()
                window_weight.zero_()
                window_steps, window_started = 0, time.perf_counter()
        train_seconds = time.perf_counter() - train_started
        logger.info("Completed training epoch=%s elapsed=%.1fs", epoch, train_seconds)
        totals = accelerator.reduce(torch.stack((loss_sum, steps)), reduction="sum")
        denominator = totals[1].clamp_min(1)
        result = {
            "epoch": epoch,
            "train_loss": float((totals[0] / denominator).item()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_seconds": train_seconds,
        }
        should_validate = (
            epoch % config.validation_every_n_epochs == 0 or epoch == config.num_epochs
        )
        if should_validate:
            validation_started = time.perf_counter()
            with log_phase(logger, f"validation epoch={epoch}"):
                validation = validate_epoch()
                result.update(validation)
            result["validation_seconds"] = time.perf_counter() - validation_started
            if accelerator.is_main_process:
                logger.info(
                    "Validation metrics epoch=%s %s",
                    epoch,
                    " ".join(
                        f"{name}={float(value):.8f}"
                        for name, value in sorted(validation.items())
                    ),
                )
        history.append(result)
        if tracker:
            tracker.log_metrics(
                {
                    name: {"overall": float(value)}
                    for name, value in result.items()
                    if name != "epoch"
                },
                iteration=epoch,
            )
        if not should_validate:
            continue
        score = float(result[early.metric]) if early else 0.0
        if early is None or score > best_score + early.min_delta:
            best_score, stale_epochs = score, 0
            if accelerator.is_main_process:
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in accelerator.unwrap_model(model)
                    .state_dict()
                    .items()
                }
        else:
            stale_epochs += 1
            if stale_epochs >= early.patience:
                break
    if accelerator.is_main_process and not best_state:
        raise RuntimeError(
            "No validation checkpoint selected; check validation metrics"
        )
    return best_state, history, best_score


def _optimizer(model: nn.Module, config: Config) -> torch.optim.Optimizer:
    """Build Adam or AdamW for the model parameters from training settings."""

    params = dict(config.training.optimizer.params)
    if config.training.optimizer.type == "adam":
        return torch.optim.Adam(model.parameters(), **params)
    if config.training.optimizer.type == "adamw":
        return torch.optim.AdamW(model.parameters(), **params)
    raise ValueError(f"unsupported optimizer {config.training.optimizer.type!r}")


def _scheduler(
    optimizer: torch.optim.Optimizer, config: Config, *, steps_per_epoch: int
):
    """Build a schedule over optimizer updates of the already sharded train loader."""
    if config.training.scheduler.type == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    if config.training.scheduler.type != "cosine_warmup":
        raise ValueError("unsupported scheduler")
    total = (
        math.ceil(steps_per_epoch / config.training.gradient_accumulation_steps)
        * config.training.num_epochs
    )
    params = config.training.scheduler.params
    warmup = math.ceil(total * params["warmup_ratio"])
    floor = params["min_lr_ratio"]

    def multiplier(step):
        """Warm up linearly, then decay to the configured fraction of peak LR."""
        if step < warmup:
            return step / warmup
        progress = min((step - warmup) / max(1, total - warmup), 1.0)
        return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _to_device(value: Any, device: torch.device) -> Any:
    """Move nested dict tensors to the device, leaving other values unchanged."""

    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value
