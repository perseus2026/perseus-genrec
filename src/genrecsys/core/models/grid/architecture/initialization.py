"""Explicit initialization strategies for GRID architecture components."""

import torch
from torch import nn


def initialize_grid_weights(model: nn.Module, strategy: str) -> None:
    """Keep HF defaults, reset the GRID backbone, or apply Xavier uniform."""
    if strategy == "default":
        return
    if strategy == "grid":
        # Upstream resets its encoder/decoder before replacing their FFNs and
        # removes embed_tokens first. Preserve the shared SID embedding here.
        for stack in (model.model.encoder, model.model.decoder):
            for name, child in stack.named_children():
                if name != "embed_tokens":
                    _reset_parameters(child)
        return
    if strategy != "xavier_uniform":
        raise ValueError(f"Unknown weight initialization: {strategy!r}")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.ndim >= 2:
                nn.init.xavier_uniform_(parameter)
            elif name.endswith("weight"):
                nn.init.ones_(parameter)
            else:
                nn.init.zeros_(parameter)
        for module in model.modules():
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                module.weight[module.padding_idx].zero_()


def _reset_parameters(module: nn.Module) -> None:
    """Follow upstream GRID's recursive PyTorch reset_parameters traversal."""
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
    else:
        for child in module.children():
            _reset_parameters(child)
