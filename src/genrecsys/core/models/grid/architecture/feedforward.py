"""Deeper ReLU feed-forward blocks for the GRID encoder and decoder."""

import torch
from torch import nn
from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5LayerNorm


class FeedForward(nn.Module):
    """Apply pre-norm, a multi-layer ReLU MLP and a residual connection.

    Match GRID's sequence of hidden linear layers, ReLU and dropout, including
    biases and dropout after the output projection. Native one-hidden-layer T5
    blocks are retained separately so their initialization and math stay intact.
    """

    def __init__(self, config: T5Config, num_hidden_layers: int):
        super().__init__()
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.hidden = nn.ModuleList(
            nn.Linear(config.d_model if index == 0 else config.d_ff, config.d_ff)
            for index in range(num_hidden_layers)
        )
        self.output = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return the input plus its normalized feed-forward transformation."""

        transformed = self.layer_norm(hidden_states)
        for index, layer in enumerate(self.hidden):
            transformed = layer(transformed)
            if index:
                transformed = self.dropout(transformed)
            transformed = transformed.relu()
        transformed = self.dropout(self.output(transformed))
        return hidden_states + self.dropout(transformed)
