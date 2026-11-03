"""Initialization recipes for the supported OneRec backbones."""

import torch
from torch import nn


def initialize_onerec_weights(model: nn.Module, strategy: str) -> None:
    """Keep module defaults or apply Xavier uniform with identity norm scales."""
    if strategy == "default":
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
