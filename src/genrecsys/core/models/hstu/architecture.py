"""Bidirectional HSTU retrieval with a query readout and bounded catalog logits."""

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from genrecsys.core.models.sequence import EventEncoder, PiecewiseLinearEncoding


class HSTUBlock(nn.Module):
    """Apply the Perseus SiLU attention normalization and gated residual."""

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.input_norm = nn.LayerNorm(dim)
        self.uvqk = nn.Linear(dim, 4 * dim)
        self.output_norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, bias, scale):
        batch, length, dim = x.shape
        parts = F.silu(self.uvqk(self.input_norm(x))).chunk(4, -1)
        u, v, q, k = (
            value.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
            for value in parts
        )
        scores = q @ k.transpose(-2, -1) + bias
        weights = (F.silu(scores) / scale).masked_fill(mask, 0)
        values = (weights @ v).transpose(1, 2).reshape(batch, length, dim)
        u = u.transpose(1, 2).reshape(batch, length, dim)
        gated = self.output_norm(values) * u
        return x + self.output(self.dropout(gated))


class HSTU(nn.Module):
    """Encode lifelong and score exact items with a tied item embedding table."""

    def __init__(self, description: dict):
        super().__init__()
        self.description = description
        dim, embedding_dim = description["dim"], description["embedding_dim"]
        if description["feature_fusion"] == "concat_projection":
            self.event_encoder = EventEncoder(description, dim, description["dropout"])
        self.embeddings = nn.ModuleDict()
        self.feature_projections = nn.ModuleDict()
        features = {
            feature["name"]: feature
            for feature in (
                description["context_features"]
                if description["feature_fusion"] == "concat_projection"
                else [*description["features"], *description["context_features"]]
            )
        }
        for feature in features.values():
            if feature["type"] == "id":
                key = feature["encoder"]
                if key not in self.embeddings:
                    self.embeddings[key] = nn.Embedding(
                        feature["vocab_size"], embedding_dim, padding_idx=0
                    )
            elif feature["type"] == "ple":
                self.feature_projections[feature["name"]] = nn.Sequential(
                    PiecewiseLinearEncoding(feature["thresholds"]),
                    nn.Linear(
                        len(feature["thresholds"]) - 1, embedding_dim, bias=False
                    ),
                )
            else:
                self.feature_projections[feature["name"]] = nn.Linear(
                    1, embedding_dim, bias=False
                )
        if description["feature_fusion"] == "sum" or description["context_features"]:
            self.input_projection = nn.Linear(embedding_dim, dim, bias=False)
        self.output_projection = nn.Linear(dim, embedding_dim, bias=False)
        self.readout = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.readout, std=0.02)
        self.position_bias = nn.Embedding(
            2 * description["max_len"] + 1, 1, padding_idx=0
        )
        self.time_bias = nn.Embedding(
            2 * description["num_time_buckets"] + 1, 1, padding_idx=0
        )
        nn.init.normal_(self.position_bias.weight, std=0.02)
        nn.init.normal_(self.time_bias.weight, std=0.02)
        with torch.no_grad():
            self.position_bias.weight[0].zero_()
            self.time_bias.weight[0].zero_()
        self.layers = nn.ModuleList(
            HSTUBlock(dim, description["num_heads"], description["dropout"])
            for _ in range(description["num_layers"])
        )
        self.register_buffer(
            "candidate_tokens", torch.empty(0, dtype=torch.long), persistent=False
        )
        self.loss_normalizer = None
        if description["weight_initialization"] == "perseus":
            with torch.random.fork_rng(devices=[]):
                self.apply(_initialize_perseus_weights)

    @property
    def item_embeddings(self):
        if self.description["feature_fusion"] == "concat_projection":
            return self.event_encoder.embeddings[self.description["target_encoder"]]
        return self.embeddings[self.description["target_encoder"]]

    def _event_embeddings(self, features, descriptions=None):
        """Sum aligned categorical and projected numeric features at embedding width."""
        values = []
        descriptions = (
            self.description["features"] if descriptions is None else descriptions
        )
        for feature in descriptions:
            value = features[feature["name"]]
            if feature["type"] == "id":
                value = self.embeddings[feature["encoder"]](value.squeeze(-1))
            else:
                value = self.feature_projections[feature["name"]](value)
            values.append(value)
        return torch.stack(values).sum(0)

    def _attention(self, valid, timestamps, dtype):
        """Build signed position/time biases; history cannot attend to query readout."""
        length = valid.shape[1]
        positions = valid.long().cumsum(-1).masked_fill(~valid, 0)
        positions[:, -1] = 0  # Dedicated query position, as in Perseus.
        differences = positions[:, :, None] - positions[:, None, :]
        position_buckets = torch.where(
            differences >= 0,
            differences + 1,
            -differences + self.description["max_len"],
        )
        time_differences = timestamps[:, :, None] - timestamps[:, None, :]
        buckets = (
            (time_differences.abs().log1p() / 0.301)
            .long()
            .clamp_max(self.description["num_time_buckets"] - 1)
        )
        time_buckets = (
            buckets + 1 + time_differences.lt(0) * self.description["num_time_buckets"]
        )
        mask = ~valid[:, None, :].expand(-1, length, -1).clone()
        mask[:, :-1, -1] = True
        position_buckets = position_buckets.masked_fill(mask, 0)
        time_buckets = time_buckets.masked_fill(mask, 0)
        bias = (
            self.position_bias(position_buckets) + self.time_bias(time_buckets)
        ).squeeze(-1)
        scale = valid.sum(-1).to(dtype).view(-1, 1, 1, 1)
        return mask[:, None], bias[:, None].to(dtype), scale

    def encode(self, batch):
        """Encode past events bidirectionally and return the dedicated query readout."""
        history = batch["history"]
        valid = history["attention_mask"].bool()
        if self.description["feature_fusion"] == "concat_projection":
            x = self.event_encoder(history["features"], self.item_embeddings)
        else:
            x = self.input_projection(self._event_embeddings(history["features"]))
        readout = self.readout.to(x.dtype).expand(len(x), 1, -1)
        if self.description["context_features"]:
            context = self._event_embeddings(
                batch["context"]["features"], self.description["context_features"]
            )
            readout = readout + self.input_projection(context)
        x = torch.cat([x, readout], dim=1)
        valid = F.pad(valid, (0, 1), value=True)
        timestamps = F.pad(history["timestamps"], (0, 1), value=0)
        mask, bias, scale = self._attention(valid, timestamps, x.dtype)
        for layer in self.layers:
            if self.description["gradient_checkpointing"] and self.training:
                x = checkpoint(layer, x, mask, bias, scale, use_reentrant=False)
            else:
                x = layer(x, mask, bias, scale)
        return self.output_projection(x[:, -1])

    def _catalog_statistics(self, query, weights, labels, start):
        scores = (query @ weights.T).float()
        local = labels - start
        valid = local.ge(0) & local.lt(len(weights)) & labels.ne(-100)
        positives = scores.gather(-1, local.clamp(0, len(weights) - 1))
        return scores.logsumexp(-1), positives.masked_fill(~valid, 0).sum(-1)

    def _cross_entropy_sums(self, query, weights, labels):
        """Sum independent target losses per query without retaining full logits."""
        log_partition = torch.full((len(query),), -torch.inf, device=query.device)
        positive_sum = torch.zeros_like(log_partition)
        width = self.description["catalog_chunk_size"]
        for index, block in enumerate(weights.split(width)):
            partition, positives = checkpoint(
                self._catalog_statistics,
                query,
                block,
                labels,
                index * width,
                use_reentrant=False,
            )
            log_partition = torch.logaddexp(log_partition, partition)
            positive_sum = positive_sum + positives
        return labels.ne(-100).sum(-1) * log_partition - positive_sum

    def forward(self, batch, labels):
        """Full or uniformly sampled CE on independent native targets."""
        query = self.encode(batch)
        candidates = self.candidate_tokens
        loss_type = self.description["loss_type"]
        if loss_type == "sampled_cross_entropy":
            # Uniform draws with replacement plus all microbatch positives.
            # Do not allocate randperm(catalog_size) for million-item catalogs.
            positive_mask = labels.ne(-100)
            indices = torch.cat(
                [
                    labels[positive_mask],
                    torch.randint(
                        len(candidates),
                        (self.description["num_negatives"],),
                        device=query.device,
                    ),
                ]
            ).unique(sorted=True)
            candidates = candidates[indices]
            labels = labels.clone()
            labels[positive_mask] = torch.searchsorted(indices, labels[positive_mask])
        # Gather once, then split: one embedding backward for the catalog,
        # rather than a vocabulary-sized gradient allocation per block.
        losses = self._cross_entropy_sums(
            query, self.item_embeddings(candidates), labels
        )
        valid = labels.ne(-100)
        counts = valid.sum(-1)
        if self.description["reduction"] == "mean_targets_global":
            numerator = losses.sum()
            denominator = counts.sum().float()
        else:
            numerator = (losses / counts).sum()
            denominator = query.new_tensor(len(query), dtype=torch.float32)
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
            dist.all_reduce(denominator)
        self.loss_normalizer = denominator
        return numerator * world_size / denominator, query

    @torch.inference_mode()
    def topk(self, batch, candidates: torch.Tensor, k: int):
        """Compute exact top-k over the supplied split catalog with bounded logits."""
        query = self.encode(batch)
        scores = torch.empty(len(query), 0, device=query.device)
        indices = torch.empty(len(query), 0, dtype=torch.long, device=query.device)
        width = self.description["catalog_chunk_size"]
        for start in range(0, len(candidates), width):
            tokens = candidates[start : start + width]
            block = (query @ self.item_embeddings(tokens).T).float()
            block_scores, block_indices = block.topk(min(k, len(tokens)), -1)
            scores = torch.cat((scores, block_scores), -1)
            indices = torch.cat((indices, block_indices + start), -1)
            scores, order = scores.topk(min(k, scores.shape[1]), -1)
            indices = indices.gather(-1, order)
        return scores, indices


@torch.no_grad()
def _initialize_perseus_weights(module):
    """Use Perseus module initialization; keep padding and norm identities."""
    if isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.padding_idx is not None:
            module.weight[module.padding_idx].zero_()
    elif isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
