"""OneRec fitting and validation over an already prepared dataset."""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import GradientAccumulationPlugin, gather_object, set_seed
from torch import nn

from genrecsys.core import trackers
from genrecsys.core.checkpoint import Prepared
from genrecsys.core.dataset import Dataset
from genrecsys.core.evaluation import metric_specs, relevance_by_item, retrieval_metric
from genrecsys.core.models.onerec import config as onerec_config
from genrecsys.core.models.onerec import inference
from genrecsys.core.models.onerec.architecture import OneRec
from genrecsys.core.models.onerec.catalog import (
    _allowed_codes,
    _catalog,
    _item_creation_times,
    _item_preprocessor,
    _items_by_sid,
    _training_targets,
)
from genrecsys.core.models.onerec.samples import Samples, _loader
from genrecsys.core.models.training import _optimizer, _scheduler, _to_device
from genrecsys.core.tasks import retrieval

logger = logging.getLogger(__name__)


def initialize_feature_embeddings(model: nn.Module, checkpoint: Prepared) -> None:
    """Load frozen feature lookups once, after ordinary model initialization."""
    for name, encoder in checkpoint.config.encoders.items():
        artifact = encoder.embedder.get("artifact")
        if encoder.type != "id" or artifact is None:
            continue
        matrix = np.load(checkpoint.files[artifact], allow_pickle=False)
        for event_encoder in model.event_encoders.values():
            if name not in event_encoder.embeddings:
                continue
            weight = event_encoder.embeddings[name].weight
            if tuple(matrix.shape) != tuple(weight.shape):
                raise ValueError(
                    f"feature embedding {name} shape differs from its frozen vocabulary"
                )
            with torch.no_grad():
                weight.copy_(torch.from_numpy(matrix))


def fit_model(
    checkpoint: Prepared,
    dataset: Dataset,
    /,
) -> dict[str, Any]:
    """Fit OneRec with Accelerate and return the selected checkpoint state.

    Use gradient accumulation, validation, and configured early stopping. The main
    rank retains selected CPU weights, metric history, and the validation catalog;
    the caller writes the checkpoint.

    Validation generates recommendations and may cost more than training.
    """

    config = checkpoint.config
    set_seed(config.training.dataloader.seed)
    # Autocast keeps master weights and optimizer state in FP32.
    accelerator = Accelerator(
        mixed_precision=config.training.mixed_precision,
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=config.training.gradient_accumulation_steps,
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
    if not len(dataset.train_samples) or not len(dataset.validation_samples):
        raise ValueError("training and validation splits must not be empty")
    if not isinstance(dataset.train_artifacts, retrieval.Artifacts) or not isinstance(
        dataset.validation_artifacts, retrieval.Artifacts
    ):
        raise TypeError("multi-path OneRec requires retrieval.Artifacts")
    item = _item_preprocessor(checkpoint)
    model_config = onerec_config._model_config(checkpoint)
    catalog = _catalog(dataset.validation_artifacts, item, model_config["code_len"])
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
    train_loader = _loader(train_data, config.training.dataloader)
    validation_loader = _loader(
        validation_data,
        config.training.validation_dataloader,
    )
    # Keep dynamic, variable-length collation on each rank. No global GPU batch.
    train_loader = accelerator.prepare_data_loader(train_loader, device_placement=False)
    validation_loader = accelerator.prepare_data_loader(
        validation_loader, device_placement=False
    )
    model = OneRec(model_config)
    initialize_feature_embeddings(model, checkpoint)
    if config.training.gradient_checkpointing:
        model.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
                "preserve_rng_state": True,
            }
        )
    optimizer = _optimizer(model, config)
    scheduler = _scheduler(optimizer, config, steps_per_epoch=len(train_loader))
    # Step the raw scheduler once per rank-local optimizer update. Accelerate's
    # scheduler wrapper otherwise multiplies steps by world size with split_batches=False.
    model, optimizer = accelerator.prepare(
        model,
        optimizer,
    )
    device = accelerator.device
    if accelerator.is_main_process:
        nominal_global_batch = (
            config.training.dataloader.batch_size
            * accelerator.num_processes
            * config.training.gradient_accumulation_steps
        )
        logger.info(
            "training setup: device=%s ranks=%s precision=%s "
            "train_rows=%s validation_rows=%s "
            "train_steps_per_rank=%s validation_steps_per_rank=%s "
            "train_batch_per_rank=%s nominal_global_batch=%s "
            "gradient_accumulation_steps=%s epochs=%s gradient_checkpointing=%s",
            device,
            accelerator.num_processes,
            accelerator.mixed_precision,
            len(train_data),
            len(validation_data),
            len(train_loader),
            len(validation_loader),
            config.training.dataloader.batch_size,
            nominal_global_batch,
            config.training.gradient_accumulation_steps,
            config.training.num_epochs,
            config.training.gradient_checkpointing,
        )
    specs = metric_specs(config)
    maximum_k = max(spec.k for spec in specs)
    validation_beam_size = int(config.inference.predict_kwargs["beam_size"])
    if maximum_k >= 100:
        logger.warning(
            "Multi-path validation runs constrained autoregressive generation at "
            "k=%s and can take longer than training; use a deliberately small "
            "validation basis for framework bring-up",
            maximum_k,
        )
    allowed_codes = _allowed_codes(catalog)
    items_by_sid = _items_by_sid(catalog)
    item_created_at = _item_creation_times(catalog)
    from genrecsys.core.models.training import run_training

    def validate_epoch():
        """Evaluate with the validation candidate catalog."""
        unwrapped = accelerator.unwrap_model(model)
        # Generation bypasses the prepared forward, so enter autocast explicitly.
        with accelerator.autocast():
            metrics = _evaluate(
                unwrapped,
                validation_loader,
                allowed_codes,
                items_by_sid,
                specs,
                device,
                accelerator=accelerator,
                beam_size=validation_beam_size,
                item_created_at=item_created_at,
                candidate_count=len(catalog),
            )
        if accelerator.is_main_process:
            logger.info(
                "validation metrics: %s",
                " ".join(f"{name}={value:.6f}" for name, value in metrics.items()),
            )
        return metrics

    best_state, history, best_score = run_training(
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
        "state_dict": best_state,
        "catalog": catalog,
        "training": {
            "history": history,
            "best_score": best_score,
            "data_statistics": dataset.statistics,
        },
        "model_config": model_config,
        "is_main_process": accelerator.is_main_process,
    }


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
) -> dict[str, float]:
    """Evaluate retrieval, gathering only per-sample metric scalars."""

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
        )
        batch_metrics = []
        for ranking, items, labels in zip(
            predictions, batch["target_items"], batch["target_item_labels"]
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
