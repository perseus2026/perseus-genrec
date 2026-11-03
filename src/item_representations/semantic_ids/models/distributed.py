"""Accelerate execution for single-process and distributed SID training."""

from __future__ import annotations

import torch
from accelerate import Accelerator, DataLoaderConfiguration


def training_accelerator(device: str | torch.device | None) -> Accelerator:
    """Use FP32 Accelerate training; the launcher selects CUDA devices."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return Accelerator(
        cpu=device.type == "cpu",
        mixed_precision="no",
        gradient_accumulation_steps=1,
        dataloader_config=DataLoaderConfiguration(
            split_batches=False,
            dispatch_batches=False,
            even_batches=True,
            use_seedable_sampler=False,
        ),
    )


def check_loss(loss: torch.Tensor, accelerator: Accelerator) -> None:
    finite = torch.isfinite(loss).to(dtype=torch.int32)
    finite = accelerator.reduce(finite, reduction="sum")
    if finite.item() != accelerator.num_processes:
        raise FloatingPointError("Non-finite SID training loss on at least one process")


def distributed_config(accelerator: Accelerator, batch_size: int) -> dict:
    return {
        "distributed": {
            "world_size": accelerator.num_processes,
            "batch_size_per_device": batch_size,
            "nominal_global_batch_size": batch_size * accelerator.num_processes,
            "even_batches": True,
            "gradient_reduction": "mean",
            "precision": "float32",
        }
    }
