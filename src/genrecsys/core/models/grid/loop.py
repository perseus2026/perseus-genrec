"""Step-based GRID recipe; the ordinary OneRec recipe uses the shared loop.

The paper monitors NDCG for patience and selects weights by Recall. Keeping this
optional loop here avoids changing the shared epoch-based Training contract.
Steps below always mean completed optimizer updates, never decoder target chunks.
"""

import logging
import math
import time

import torch

logger = logging.getLogger(__name__)


def run_training(
    model,
    optimizer,
    scheduler,
    loader,
    config,
    accelerator,
    tracker,
    validate,
    *,
    to_device,
    recipe,
):
    """Train with explicit validation intervals and independent checkpoint selection."""
    early = config.early_stopping
    checkpoint_metric = recipe["checkpoint_metric"] or (early.metric if early else None)
    best_score, best_monitor = -math.inf, -math.inf
    best_state, history = {}, []
    updates, stale, last_validation = 0, 0, -1
    stop = False

    def evaluate(epoch, loss_sum, weight_sum, started):
        nonlocal best_score, best_monitor, best_state, stale, last_validation, stop
        totals = accelerator.reduce(
            torch.stack((loss_sum, weight_sum)), reduction="sum"
        )
        result = {
            "epoch": epoch,
            "optimizer_step": updates,
            "train_loss": float((totals[0] / totals[1].clamp_min(1)).item()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_seconds": time.perf_counter() - started,
        }
        validation_started = time.perf_counter()
        metrics = validate()
        if any(not math.isfinite(float(value)) for value in metrics.values()):
            raise ValueError("GRID validation metrics must be finite")
        result.update(metrics)
        result["validation_seconds"] = time.perf_counter() - validation_started
        history.append(result)
        last_validation = updates
        score = float(metrics[checkpoint_metric]) if checkpoint_metric else 0.0
        # min_delta belongs to the stopping monitor, not checkpoint ranking.
        if checkpoint_metric is None or score > best_score:
            best_score = score
            if accelerator.is_main_process:
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in accelerator.unwrap_model(model)
                    .state_dict()
                    .items()
                }
        if early is not None:
            monitor = float(metrics[early.metric])
            if monitor > best_monitor + early.min_delta:
                best_monitor, stale = monitor, 0
            else:
                stale += 1
                stop = stale >= early.patience
        if tracker:
            tracker.log_metrics(
                {name: {"overall": float(value)} for name, value in result.items()},
                iteration=updates,
            )
        logger.info(
            "GRID validation epoch=%s optimizer_step=%s metrics=%s",
            epoch,
            updates,
            metrics,
        )
        model.train()

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, config.num_epochs + 1):
        loader.set_epoch(epoch - 1)
        model.train()
        loss_sum = torch.zeros((), device=accelerator.device)
        weight_sum = torch.zeros_like(loss_sum)
        accumulation_weight = torch.zeros_like(loss_sum)
        started = time.perf_counter()
        for microstep, batch in enumerate(loader, 1):
            batch = to_device(batch, accelerator.device)
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    loss, _ = model(batch, labels=batch["target"])
                normalizer = getattr(
                    accelerator.unwrap_model(model), "loss_normalizer", None
                )
                weight = 1 if normalizer is None else normalizer
                accelerator.backward(loss * weight)
                if normalizer is not None:
                    accumulation_weight += weight
                    if accelerator.sync_gradients:
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
                updated = (
                    accelerator.sync_gradients
                    and not accelerator.optimizer_step_was_skipped
                )
                if updated:
                    scheduler.step()
                    updates += 1
                optimizer.zero_grad(set_to_none=True)
            loss_sum += loss.detach() * weight
            weight_sum += weight
            if microstep % config.log_every_n_train_steps == 0:
                logger.info(
                    "GRID epoch=%s microstep=%s optimizer_step=%s loss=%.6f",
                    epoch,
                    microstep,
                    updates,
                    loss.item(),
                )
            reached_limit = (
                recipe["max_steps"] is not None and updates >= recipe["max_steps"]
            )
            interval = recipe["validation_every_n_steps"]
            if updated and (
                (interval is not None and updates % interval == 0) or reached_limit
            ):
                evaluate(epoch, loss_sum, weight_sum, started)
            if stop or reached_limit:
                stop = True
                break
        final_epoch = stop or epoch == config.num_epochs
        epoch_validation = (
            recipe["validation_every_n_steps"] is None
            and epoch % config.validation_every_n_epochs == 0
        )
        if last_validation != updates and (epoch_validation or final_epoch):
            evaluate(epoch, loss_sum, weight_sum, started)
        if stop:
            break
    if accelerator.is_main_process and not best_state:
        raise RuntimeError("No GRID validation checkpoint selected")
    return best_state, history, best_score
