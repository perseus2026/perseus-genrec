"""Reusable feature projection and QFormer layers; no dataset-specific ports."""

import torch
from torch import nn

from genrecsys.core.encoders.ple import encode as ple_encode


class PiecewiseLinearEncoding(nn.Module):
    """Dynamically expand one scalar using frozen train-quantile thresholds."""

    def __init__(self, thresholds: list[float]):
        super().__init__()
        self.register_buffer(
            "thresholds",
            torch.tensor(thresholds, dtype=torch.float32),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Return one coordinate per interval without materializing it on disk."""

        return ple_encode(values, self.thresholds)


class EventEncoder(nn.Module):
    """Compose one event token or one token per SID coordinate.

    Concatenate non-additive features and project to d_model, then add feature
    embeddings of width d_model. Project additive numeric/PLE features separately.
    """

    def __init__(self, description: dict, d_model: int, dropout: float):
        super().__init__()
        self.features = description["features"]
        self.token_layout = description.get("token_layout", "event")
        if self.token_layout not in {"event", "sid_sequence"}:
            raise ValueError("token_layout must be event or sid_sequence")
        self.tokens_per_event = 1
        if self.token_layout == "sid_sequence":
            sid = [f for f in self.features if f["type"] == "semantic_id"]
            if description["type"] != "sequence" or len(sid) != 1:
                raise ValueError(
                    "sid_sequence requires a sequence with one SID feature"
                )
            self.tokens_per_event = sid[0]["prefix_length"]
        self.additive = set(description["additive_features"])
        self.embeddings = nn.ModuleDict()
        self.ple_encoders = nn.ModuleDict()
        self.additive_projections = nn.ModuleDict()
        self.feature_projections = nn.ModuleDict()
        width = 0
        identity = description["event_encoder_type"] == "identity"
        for feature in self.features:
            kind = feature["type"]
            dim = (
                d_model
                if identity or feature["name"] in self.additive
                else min(64, d_model)
            )
            if "embedding_dim" in feature:
                dim = int(feature["embedding_dim"])
            if kind == "id" and feature["encoder"] not in self.embeddings:
                self.embeddings[feature["encoder"]] = nn.Embedding(
                    feature["vocab_size"], dim, padding_idx=0
                )
                if feature.get("frozen_embedding", False):
                    self.embeddings[feature["encoder"]].weight.requires_grad_(False)
            if kind == "ple" and feature["encoder"] not in self.ple_encoders:
                self.ple_encoders[feature["encoder"]] = PiecewiseLinearEncoding(
                    feature["thresholds"]
                )
            if kind == "id":
                feature_width = dim
            elif kind == "semantic_id":
                feature_width = (
                    d_model
                    if self.token_layout == "sid_sequence"
                    else feature["prefix_length"] * d_model
                )
            elif kind == "ple":
                feature_width = len(feature["thresholds"]) - 1
            else:
                feature_width = 1
            if (
                kind == "ple"
                and feature["name"] not in self.additive
                and "embedding_dim" in feature
            ):
                self.feature_projections[feature["name"]] = nn.Linear(
                    feature_width, dim, bias=False
                )
                feature_width = dim
            if feature["name"] in self.additive and kind in {"ple", "numeric"}:
                self.additive_projections[feature["name"]] = nn.Linear(
                    feature_width, d_model, bias=False
                )
            if feature["name"] not in self.additive:
                width += feature_width
        self.projection = (
            nn.Identity()
            if identity
            else nn.Sequential(
                nn.Linear(width, d_model),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
        )

    def forward(self, values: dict, sid_embeddings: nn.Embedding) -> torch.Tensor:
        """Embed/project concatenated features, then sum additive feature embeddings."""
        parts, additive = [], []
        for feature in self.features:
            value = values[feature["name"]]
            if feature["type"] == "semantic_id":
                codes = value[..., : feature["prefix_length"]]
                embedded = sid_embeddings(codes)
                if feature.get("optional", False):
                    embedded = embedded.masked_fill(codes.eq(0).unsqueeze(-1), 0)
                if self.token_layout == "event":
                    embedded = embedded.flatten(2)
            elif feature["type"] == "id":
                embedded = self.embeddings[feature["encoder"]](value.squeeze(-1))
            elif feature["type"] == "ple":
                embedded = self.ple_encoders[feature["encoder"]](value)
            else:
                embedded = value
            if feature["name"] in self.feature_projections:
                embedded = self.feature_projections[feature["name"]](embedded)
            if feature["name"] in self.additive_projections:
                embedded = self.additive_projections[feature["name"]](embedded)
            if self.token_layout == "sid_sequence" and feature["type"] != "semantic_id":
                embedded = embedded.unsqueeze(2).expand(
                    -1, -1, self.tokens_per_event, -1
                )
            (additive if feature["name"] in self.additive else parts).append(embedded)
        tokens = self.projection(torch.cat(parts, dim=-1))
        for embedded in additive:
            tokens = tokens + embedded
        return tokens.flatten(1, 2) if self.token_layout == "sid_sequence" else tokens


class QFormerBlock(nn.Module):
    """Apply query cross-attention to a masked history.

    Queries and outputs have shape (B, Q, D); values have shape (B, S, D).
    The value mask (B, S) is 1 for events and 0 for padding.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        """Build pre-normalized cross-attention and a residual feed-forward block."""

        super().__init__()
        self.query_norm = nn.RMSNorm(d_model)
        self.kv_norm = nn.RMSNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(
        self,
        queries: torch.Tensor,
        values: torch.Tensor,
        value_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Update queries (B, Q, D) from values (B, S, D), ignoring padded values."""

        key_padding_mask = None
        if value_mask is not None:
            # PyTorch MultiheadAttention expects True at positions that should
            # be ignored, while the rest of the repo uses 1 for visible tokens.
            key_padding_mask = value_mask == 0

        normed_values = self.kv_norm(values)
        attn_out, _ = self.cross_attn(
            query=self.query_norm(queries),
            key=normed_values,
            value=normed_values,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        queries = queries + self.dropout(attn_out)
        queries = queries + self.dropout(self.ffn(self.ffn_norm(queries)))
        return queries


class QFormerCompressor(nn.Module):
    """Compress long user history into a fixed number of query tokens.

    See OneRec Technical Report, section 2.2.1, equations (17)-(18):
    https://arxiv.org/html/2506.13695v1#S2.SS2.SSS1
    This is a compact query cross-attention/FFN variant, not a complete BLIP-2
    Q-Former. Its output length is always ``num_queries``, independent of history.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_queries: int,
        num_layers: int,
        dropout: float,
    ):
        """Initialize learned query tokens and the QFormer layer stack."""

        super().__init__()
        self.query_tokens = nn.Parameter(torch.empty(num_queries, d_model))
        self.layers = nn.ModuleList(
            [
                QFormerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        nn.init.normal_(self.query_tokens, mean=0.0, std=d_model**-0.5)

    def forward(
        self,
        values: torch.Tensor,
        value_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compress history into (batch, num_queries, dim), regardless of input length."""

        batch_size = values.size(0)
        # Learned query tokens are shared across users and expanded per batch.
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        for layer in self.layers:
            queries = layer(queries, values, value_mask)
        return queries
