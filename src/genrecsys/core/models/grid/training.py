"""GRID fitting and validation over an already prepared dataset."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict
from importlib.metadata import version
from typing import Any

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import GradientAccumulationPlugin, gather_object, set_seed
from torch import nn

from genrecsys.core import trackers
from genrecsys.core.checkpoint import Prepared
from genrecsys.core.config import Config
from genrecsys.core.dataset import Dataset
from genrecsys.core.evaluation import metric_specs, relevance_by_item, retrieval_metric
from genrecsys.core.models.grid import architecture, inference, loop
from genrecsys.core.models.grid import config as grid_config
from genrecsys.core.models.grid.catalog import (
    _allowed_codes,
    _catalog,
    _generation_catalog,
    _item_creation_times,
    _item_preprocessor,
    _items_by_sid,
    _training_targets,
)
from genrecsys.core.models.grid.samples import Samples, _loader, _to_device
from genrecsys.core.models.training import run_training

logger = logging.getLogger(__name__)


def fit_model(checkpoint: Prepared, dataset: Dataset, /) -> dict[str, Any]:
    """Fit with the shared training loop and select weights on held-out validation.

    Architecture choices never change optimizer, LR, precision or global batch.
    Autocast keeps master weights and optimizer state in FP32.
    """

    config = checkpoint.config
    set_seed(config.training.dataloader.seed)
    accelerator = Accelerator(
        mixed_precision=config.training.mixed_precision,
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=config.training.gradient_accumulation_steps,
            sync_with_dataloader=True,
        ),
        dataloader_config=DataLoaderConfiguration(
            split_batches=False,
            dispatch_batches=False,
            even_batches=True,
            use_seedable_sampler=True,
            data_seed=config.training.dataloader.seed,
        ),
    )
    tracker_config = config.training.tracker
    tracker = (
        trackers.registry[tracker_config.type](**tracker_config.params)
        if accelerator.is_main_process
        else None
    )
    item = _item_preprocessor(checkpoint)
    model_config = grid_config._model_config(checkpoint)
    catalog = _catalog(dataset.validation_artifacts, item)
    index_to_sid = _training_targets(dataset, dataset.train_artifacts, item)
    train_data = Samples(
        dataset.train_samples,
        config,
        code_length=model_config["code_len"],
        event_store=dataset.event_store,
        index_to_sid=index_to_sid,
        train=True,
    )
    validation_data = Samples(
        dataset.validation_samples,
        config,
        code_length=model_config["code_len"],
        event_store=dataset.event_store,
    )
    recipe = grid_config.training_parameters(config)
    train_loader = accelerator.prepare_data_loader(
        _loader(train_data, config.training.dataloader), device_placement=False
    )
    validation_loader = accelerator.prepare_data_loader(
        _loader(validation_data, config.training.validation_dataloader),
        device_placement=False,
    )
    model = architecture.build_model(model_config)
    if config.training.gradient_checkpointing:
        model.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
                "preserve_rng_state": True,
            }
        )
    optimizer = _optimizer(model, config)
    scheduler = _scheduler(optimizer, config, steps_per_epoch=len(train_loader))
    # Step the raw scheduler once per rank-local optimizer update, as in OneRec.
    model, optimizer = accelerator.prepare(model, optimizer)
    nominal_global_batch = (
        config.training.dataloader.batch_size
        * accelerator.num_processes
        * config.training.gradient_accumulation_steps
    )
    if accelerator.is_main_process:
        logger.info(
            "training setup: device=%s ranks=%s precision=%s "
            "train_rows=%s validation_rows=%s train_steps_per_rank=%s "
            "batch_per_rank=%s nominal_global_batch=%s accumulation=%s "
            "epochs=%s gradient_checkpointing=%s",
            accelerator.device,
            accelerator.num_processes,
            accelerator.mixed_precision,
            len(train_data),
            len(validation_data),
            len(train_loader),
            config.training.dataloader.batch_size,
            nominal_global_batch,
            config.training.gradient_accumulation_steps,
            config.training.num_epochs,
            config.training.gradient_checkpointing,
        )
        logger.info(
            "GRID consumes SID feature=%s history_input=%s; unused inputs=%s features=%s",
            config.task.objective.target,
            grid_config.history_input(config),
            sorted(set(config.inputs) - {grid_config.history_input(config)}),
            sorted(set(config.features) - {config.task.objective.target}),
        )
    specs = metric_specs(config)
    eligible = _generation_catalog(catalog)
    allowed_codes = _allowed_codes(eligible)
    items_by_sid = _items_by_sid(eligible)
    item_created_at = _item_creation_times(eligible)

    def validate_epoch():
        """Generate with the same decoding policy as final inference."""

        with accelerator.autocast():
            metrics = _evaluate(
                accelerator.unwrap_model(model),
                validation_loader,
                allowed_codes,
                items_by_sid,
                specs,
                accelerator.device,
                accelerator=accelerator,
                beam_size=int(config.inference.predict_kwargs["beam_size"]),
                item_created_at=item_created_at,
                candidate_count=len(catalog),
                constraint=grid_config.generation_constraint(config),
                renormalize_logits=grid_config.generation_renormalization(config),
            )
        if accelerator.is_main_process:
            logger.info(
                "validation metrics: %s",
                " ".join(f"{name}={value:.6f}" for name, value in metrics.items()),
            )
        return metrics

    step_recipe = any(value is not None for value in recipe.values())
    train_loop = loop.run_training if step_recipe else run_training
    best_state, history, best_score = train_loop(
        model,
        optimizer,
        scheduler,
        train_loader,
        config.training,
        accelerator,
        tracker,
        validate_epoch,
        to_device=_to_device,
        **({"recipe": recipe} if step_recipe else {}),
    )
    accelerator.wait_for_everyone()
    return {
        "state_dict": best_state,
        "catalog": catalog,
        "training": {
            "history": history,
            "best_score": best_score,
            "data_statistics": dataset.statistics,
            "world_size": accelerator.num_processes,
            "nominal_global_batch": nominal_global_batch,
            "nominal_global_batch_unit": "source_rows",
            "training_augmentation": grid_config.augmentation_parameters(config),
            "recipe": recipe,
            "settings": asdict(config.training),
            "optimizer_defaults": dict(optimizer.defaults),
            "software": {
                package: version(package)
                for package in ("torch", "transformers", "accelerate", "polars")
            },
            "decoding_policy": {
                **config.inference.predict_kwargs,
                "score": "sum_token_log_probabilities",
                "length_penalty": 0.0,
                "future_items": "filter_after_search",
            },
            "data_contract_sha256": checkpoint.data_contract.sha256,
            "dataset_identity": checkpoint.dataset_identity,
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "history_input": grid_config.history_input(config),
            "consumed_features": [config.task.objective.target],
            "ignored_features": sorted(
                set(config.features) - {config.task.objective.target}
            ),
            "ignored_inputs": sorted(
                set(config.inputs) - {grid_config.history_input(config)}
            ),
            "history_selector": asdict(config.input(grid_config.history_input(config))),
            "max_history_items": model_config["max_history_items"],
            "mixed_precision": accelerator.mixed_precision,
        },
        "model_config": model_config,
        "is_main_process": accelerator.is_main_process,
    }


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
        raise ValueError("unsupported GRID scheduler")
    total = (
        math.ceil(steps_per_epoch / config.training.gradient_accumulation_steps)
        * config.training.num_epochs
    )
    max_steps = grid_config.training_parameters(config)["max_steps"]
    if max_steps is not None:
        total = min(total, max_steps)
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


@torch.inference_mode()
def _evaluate(
    model,
    loader,
    allowed_codes,
    items_by_sid,
    specs,
    device,
    *,
    accelerator,
    beam_size,
    candidate_count,
    item_created_at=None,
    constraint="prefix",
    renormalize_logits=False,
) -> dict[str, float]:
    """Evaluate retrieval metrics, gathering only per-sample metric scalars."""
    model.eval()
    sample_specs = tuple(spec for spec in specs if spec.metric_type != "coverage_at_k")
    coverage_specs = tuple(
        spec for spec in specs if spec.metric_type == "coverage_at_k"
    )
    names = [spec.name for spec in sample_specs]
    totals = torch.zeros(len(names), dtype=torch.float64, device=device)
    sample_count = 0
    coverage_items = {spec.name: set() for spec in coverage_specs}
    started = time.perf_counter()
    progress_every = max(1, math.ceil(len(loader) / 4))
    for batch_index, batch in enumerate(loader, 1):
        predictions = inference._predict_batch(
            model,
            _to_device(batch, device),
            allowed_codes,
            items_by_sid,
            k=max(spec.k for spec in specs),
            beam_size=beam_size,
            item_created_at=item_created_at,
            constraint=constraint,
            renormalize_logits=renormalize_logits,
        )
        batch_metrics = []
        for ranking, items, labels in zip(
            predictions, batch["target_items"], batch["target_item_labels"], strict=True
        ):
            relevance = relevance_by_item(items, labels)
            predicted_items = [value["item"] for value in ranking]
            row = [
                retrieval_metric(spec, [value["item"] for value in ranking], relevance)
                for spec in sample_specs
            ]
            for spec in coverage_specs:
                coverage_items[spec.name].update(predicted_items[: spec.k])
            batch_metrics.append(row)
        values = accelerator.gather_for_metrics(
            torch.tensor(batch_metrics, dtype=torch.float64, device=device)
        )
        totals += values.sum(0)
        sample_count += len(values)
        if accelerator.is_main_process and (
            batch_index % progress_every == 0 or batch_index == len(loader)
        ):
            elapsed = time.perf_counter() - started
            logger.info(
                "validation progress=%s%% elapsed_min=%.1f eta_min=%.1f",
                round(100 * batch_index / len(loader)),
                elapsed / 60,
                elapsed * (len(loader) / batch_index - 1) / 60,
            )
    if sample_count != len(loader.dataset):
        raise RuntimeError(
            f"validation coverage mismatch: {sample_count} != {len(loader.dataset)}"
        )
    metrics = {
        name: float(totals[index].item()) / max(sample_count, 1)
        for index, name in enumerate(names)
    }
    for spec in coverage_specs:
        items = gather_object(list(coverage_items[spec.name]))
        metrics[spec.name] = len(set(items)) / max(candidate_count, 1)
    return metrics
