"""Fit HSTU with the shared Accelerate loop and exact split-specific retrieval."""

import logging
import math
import time

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import gather_object, set_seed

from genrecsys.core import trackers
from genrecsys.core.evaluation import metric_specs, relevance_by_item, retrieval_metric
from genrecsys.core.models.hstu.architecture import HSTU
from genrecsys.core.models.hstu.config import model_config
from genrecsys.core.models.hstu.data import (
    Samples,
    catalog,
    loader,
    sid_lookup,
    vocabulary,
)
from genrecsys.core.models.hstu.inference import rankings
from genrecsys.core.models.training import (
    _optimizer,
    _scheduler,
    _to_device,
    run_training,
)

logger = logging.getLogger(__name__)


def fit_model(prepared, dataset):
    config = prepared.config
    set_seed(config.training.dataloader.seed)
    accelerator = Accelerator(
        mixed_precision=config.training.mixed_precision,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        dataloader_config=DataLoaderConfiguration(
            split_batches=False,
            dispatch_batches=False,
            even_batches=True,
            use_seedable_sampler=True,
            data_seed=config.training.dataloader.seed,
        ),
    )
    vocab = vocabulary(prepared)
    lookup = sid_lookup(vocab)
    train_catalog = catalog(dataset.train_artifacts.items, vocab)
    validation_catalog = catalog(dataset.validation_artifacts.items, vocab)
    targets = dataset.task_preprocessor.items.select("index", "item_id").join(
        train_catalog.select("item_id").with_row_index("candidate"),
        on="item_id",
        how="left",
        validate="1:1",
    )
    target_indices = dict(targets.select("index", "candidate").iter_rows())
    train_data = Samples(
        dataset.train_samples,
        config,
        lookup,
        event_store=dataset.event_store,
        train_targets=target_indices,
    )
    validation_data = Samples(
        dataset.validation_samples,
        config,
        lookup,
        event_store=dataset.event_store,
    )
    train_loader = accelerator.prepare_data_loader(
        loader(train_data, config.training.dataloader),
        device_placement=False,
    )
    validation_loader = accelerator.prepare_data_loader(
        loader(validation_data, config.training.validation_dataloader),
        device_placement=False,
    )
    description = model_config(prepared)
    model = HSTU(description)
    model.candidate_tokens = torch.tensor(
        train_catalog["item_token"].to_numpy(), dtype=torch.long
    )
    validation_tokens = torch.tensor(
        validation_catalog["item_token"].to_numpy(),
        dtype=torch.long,
        device=accelerator.device,
    )
    optimizer = _optimizer(model, config)
    scheduler = _scheduler(optimizer, config, steps_per_epoch=len(train_loader))
    model, optimizer = accelerator.prepare(model, optimizer)
    tracker = (
        trackers.registry[config.training.tracker.type](
            **config.training.tracker.params
        )
        if accelerator.is_main_process
        else None
    )
    if accelerator.is_main_process:
        count = sum(value.numel() for value in model.parameters())
        global_batch = (
            config.training.dataloader.batch_size
            * accelerator.num_processes
            * config.training.gradient_accumulation_steps
        )
        logger.info(
            "HSTU: lifelong_max=%s item_vocab=%s candidates=%s embedding_dim=%s "
            "parameters=%s fp32_weights_grad_adam_gib=%.2f "
            "full_catalog_weight_workspace_gib=%.2f ranks=%s global_batch=%s "
            "train_rows=%s validation_rows=%s loss=%s",
            description["max_len"],
            len(vocab),
            len(train_catalog),
            description["embedding_dim"],
            count,
            16 * count / 2**30,
            8 * len(train_catalog) * description["embedding_dim"] / 2**30,
            accelerator.num_processes,
            global_batch,
            len(train_data),
            len(validation_data),
            config.task.objective.loss,
        )
        logger.warning(
            "HSTU validation/test scan every candidate. Bounded logits do not bound "
            "total time; full evaluation can take longer than training. "
            "Memory estimates exclude activations and optimizer temporaries."
        )
    specs = metric_specs(config)

    def validate_epoch():
        with accelerator.autocast():
            return evaluate(
                accelerator.unwrap_model(model),
                validation_loader,
                validation_tokens,
                validation_catalog["item_id"],
                specs,
                accelerator,
            )

    weights, history, score = run_training(
        model,
        optimizer,
        scheduler,
        train_loader,
        config.training,
        accelerator,
        tracker,
        validate_epoch,
        to_device=_to_device,
    )
    accelerator.wait_for_everyone()
    return {
        "state_dict": weights,
        "catalog": validation_catalog,
        "model_config": description,
        "is_main_process": accelerator.is_main_process,
        "training": {
            "history": history,
            "best_score": score,
            "objective": "item_cross_entropy",
            "input": "lifelong",
            "candidate_count": len(train_catalog),
            "item_vocab_size": len(vocab),
            "context_features": [
                feature["name"] for feature in description["context_features"]
            ],
            "loss": dict(config.task.objective.loss),
        },
    }


@torch.inference_mode()
def evaluate(model, batches, candidates, items, specs, accelerator):
    """Gather per-query metrics, removing distributed final-batch duplicates."""
    model.eval()
    sample_specs = tuple(spec for spec in specs if spec.metric_type != "coverage_at_k")
    coverage_specs = tuple(
        spec for spec in specs if spec.metric_type == "coverage_at_k"
    )
    totals = torch.zeros(
        len(sample_specs), dtype=torch.float64, device=accelerator.device
    )
    coverage = {spec.name: set() for spec in coverage_specs}
    count, started = 0, time.perf_counter()
    every = max(1, math.ceil(len(batches) / 4))
    for step, batch in enumerate(batches, 1):
        predictions = rankings(
            model,
            _to_device(batch, accelerator.device),
            candidates,
            items,
            max(spec.k for spec in specs),
        )
        rows = []
        for ranking, targets, labels in zip(
            predictions, batch["target_items"], batch["target_item_labels"]
        ):
            predicted = [value["item"] for value in ranking]
            relevance = relevance_by_item(targets, labels)
            rows.append(
                [retrieval_metric(spec, predicted, relevance) for spec in sample_specs]
            )
            for spec in coverage_specs:
                coverage[spec.name].update(predicted[: spec.k])
        values = accelerator.gather_for_metrics(
            torch.tensor(rows, dtype=torch.float64, device=accelerator.device)
        )
        totals += values.sum(0)
        count += len(values)
        if accelerator.is_main_process and (step % every == 0 or step == len(batches)):
            elapsed = time.perf_counter() - started
            logger.info(
                "HSTU validation %s/%s elapsed_min=%.1f eta_min=%.1f",
                step,
                len(batches),
                elapsed / 60,
                elapsed * (len(batches) / step - 1) / 60,
            )
    if count != len(batches.dataset):
        raise RuntimeError(
            f"HSTU validation coverage mismatch: {count} != {len(batches.dataset)}"
        )
    result = {
        spec.name: float(totals[index].item()) / count
        for index, spec in enumerate(sample_specs)
    }
    for spec in coverage_specs:
        result[spec.name] = len(set(gather_object(list(coverage[spec.name])))) / len(
            items
        )
    if accelerator.is_main_process:
        logger.info(
            "HSTU validation metrics: %s",
            " ".join(f"{name}={value:.10f}" for name, value in result.items()),
        )
    return result
