"""One-history SID encoder-decoder with explicit GRID architecture components."""

from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import T5Config, T5ForConditionalGeneration

from .feedforward import FeedForward
from .initialization import initialize_grid_weights


class GRID(nn.Module):
    """Encode one full-SID history and autoregressively predict target SID tokens.

    Batch input_ids are already globally namespaced SID tokens, left-padded with
    PAD in complete SID groups. user_bucket, when enabled, is a precomputed hash
    of the external client ID, independent of the sample-row user_id metadata.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.code_len = config["code_len"]
        self.sid_token_vocab_size = config["vocab_size"]
        self.user_id_buckets = config["user_id_buckets"]
        if (
            not isinstance(self.user_id_buckets, int)
            or isinstance(self.user_id_buckets, bool)
            or self.user_id_buckets < 0
        ):
            raise ValueError("GRID user_id_buckets must be a non-negative integer")
        if not isinstance(config["item_separator"], bool):
            raise TypeError("GRID item_separator must be a boolean")
        if config["pad_token_id"] != 0 or config["eos_token_id"] != 1:
            raise ValueError("GRID frozen SID vocabulary reserves PAD=0 and EOS=1")
        self.vocab_size = self.sid_token_vocab_size + self.user_id_buckets
        self.output_head = config["output_head"]
        self.token_loss_reduction = config.get("token_loss_reduction", "mean")
        self.multi_target_decoder_chunk_size = config.get(
            "multi_target_decoder_chunk_size", 128
        )
        self.decoder_activation_checkpointing = config.get(
            "decoder_activation_checkpointing", True
        )
        if not isinstance(self.decoder_activation_checkpointing, bool):
            raise TypeError("GRID decoder_activation_checkpointing must be a boolean")
        self.multi_target_loss_reduction = config.get(
            "multi_target_loss_reduction", "mean_tokens_then_targets_per_sample"
        )
        self.loss_normalizer: torch.Tensor | None = None
        if self.token_loss_reduction not in {"mean", "sum"}:
            raise ValueError("GRID token_loss_reduction must be mean or sum")
        if self.multi_target_loss_reduction not in {
            "mean_tokens_then_targets_per_sample",
            "mean_tokens_then_targets_global",
        }:
            raise ValueError("unsupported GRID multi_target_loss_reduction")
        if self.output_head not in {"shared", "per_level"}:
            raise ValueError("GRID output_head must be shared or per_level")
        if config["bos_token"] not in {"pad", "learned"}:
            raise ValueError("GRID bos_token must be pad or learned")
        if (
            not isinstance(self.multi_target_decoder_chunk_size, int)
            or isinstance(self.multi_target_decoder_chunk_size, bool)
            or self.multi_target_decoder_chunk_size <= 0
        ):
            raise ValueError("GRID decoder chunk size must be positive")
        self._register_sid_vocabulary(config["allowed_tokens_by_level"])
        self.model = self._build_hf_backbone(config)
        if config["weight_initialization"] == "grid":
            initialize_grid_weights(self, "grid")
        depth = config["num_feedforward_hidden_layers"]
        if not isinstance(depth, int) or isinstance(depth, bool) or depth <= 0:
            raise ValueError("GRID feed-forward depth must be a positive integer")
        feed_forward_type = config.get(
            "feed_forward_type", "t5" if depth == 1 else "mlp"
        )
        if feed_forward_type not in {"t5", "mlp"}:
            raise ValueError("GRID feed_forward_type must be t5 or mlp")
        if feed_forward_type == "t5" and depth != 1:
            raise ValueError("native T5 feed-forward blocks require one hidden layer")
        if feed_forward_type == "mlp":
            if config["feed_forward_proj"] != "relu":
                raise ValueError("GRID MLP feed-forward blocks require relu")
            for stack in (self.model.encoder, self.model.decoder):
                for block in stack.block:
                    block.layer[-1] = FeedForward(stack.config, depth)
        self.bos_token = (
            nn.Parameter(torch.randn(1, config["d_model"]))
            if config["bos_token"] == "learned"
            else None
        )
        self.sep_token = (
            nn.Parameter(torch.randn(1, config["d_model"]))
            if config["item_separator"]
            else None
        )
        self.output_heads = (
            nn.ModuleList(
                nn.Linear(config["d_model"], size, bias=False)
                for size in self.level_sizes
            )
            if self.output_head == "per_level"
            else nn.ModuleList()
        )
        self._generation_constraints = {}
        if config["weight_initialization"] != "grid":
            initialize_grid_weights(self, config["weight_initialization"])

    @staticmethod
    def _build_hf_backbone(config: dict[str, Any]) -> nn.Module:
        """Build native T5 with its relative position bias and shared SID embeddings."""

        params = {
            key: config[key]
            for key in (
                "num_decoder_layers",
                "d_model",
                "d_ff",
                "num_heads",
                "d_kv",
                "dropout_rate",
                "vocab_size",
                "pad_token_id",
                "eos_token_id",
            )
        }
        params["vocab_size"] += config["user_id_buckets"]
        return T5ForConditionalGeneration(
            T5Config(
                **params,
                num_layers=config["num_encoder_layers"],
                decoder_start_token_id=config["pad_token_id"],
                feed_forward_proj=config["feed_forward_proj"],
            )
        )

    def _register_sid_vocabulary(self, levels: Any) -> None:
        """Preserve the frozen global token IDs and each level's declared class space.

        Token lists must describe the complete level vocabulary, including unused
        codewords when declared by the SID artifact; never infer them from a batch.
        No remapping or codebook fitting is performed by the model.
        """

        if (
            not isinstance(self.code_len, int)
            or isinstance(self.code_len, bool)
            or self.code_len <= 0
        ):
            raise ValueError("GRID code_len must be a positive integer")
        if len(levels) != self.code_len or any(not level for level in levels):
            raise ValueError("GRID requires a nonempty token vocabulary per SID level")
        for level in levels:
            if len(set(level)) != len(level) or any(
                type(token) is not int or not 2 <= token < self.sid_token_vocab_size
                for token in level
            ):
                raise ValueError(
                    "SID level tokens must be unique integers in the frozen vocabulary"
                )
        self.level_sizes = tuple(len(level) for level in levels)
        tokens = torch.full(
            (self.code_len, max(self.level_sizes)), -1, dtype=torch.long
        )
        for index, level in enumerate(levels):
            tokens[index, : len(level)] = torch.tensor(level)
        self.register_buffer("tokens_by_level", tokens)

    def build_encoder_inputs(
        self, batch: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed full SID groups, optionally add separators and prepend a user token."""

        ids, mask = batch["input_ids"], batch["attention_mask"]
        if (
            ids.ndim != 2
            or not ids.size(0)
            or not ids.size(1)
            or mask.shape != ids.shape
        ):
            raise ValueError("GRID input_ids and attention_mask must have shape (B, S)")
        if ids.dtype != torch.long or ids.size(1) % self.code_len:
            raise ValueError(
                "GRID input_ids must be long tensors padded to full SID groups"
            )
        if bool(((mask != 0) & (mask != 1)).any()) or bool(
            (mask[:, 1:] < mask[:, :-1]).any()
        ):
            raise ValueError(
                "GRID attention_mask must be binary with padding on the left"
            )
        grouped_mask = mask.reshape(ids.size(0), -1, self.code_len)
        if bool((grouped_mask != grouped_mask[:, :, :1]).any()):
            raise ValueError("GRID history must not contain partially masked SIDs")
        if bool(((ids < 0) | (ids >= self.sid_token_vocab_size)).any()) or bool(
            ((ids < 2) & mask.bool()).any()
        ):
            raise ValueError(
                "GRID history tokens are outside the frozen SID vocabulary"
            )
        grouped_ids = ids.reshape(ids.size(0), -1, self.code_len)
        visible = grouped_mask[:, :, 0].bool()
        for level, size in enumerate(self.level_sizes):
            if not bool(
                torch.isin(
                    grouped_ids[:, :, level][visible],
                    self.tokens_by_level[level, :size],
                ).all()
            ):
                raise ValueError(f"GRID history tokens are outside SID level {level}")
        embeds = self.model.shared(ids)
        mask = mask.clone()
        if self.sep_token is not None:
            grouped = embeds.reshape(ids.size(0), -1, self.code_len, embeds.size(-1))
            separators = self.sep_token.reshape(1, 1, 1, -1).expand(
                grouped.size(0), grouped.size(1), 1, -1
            )
            embeds = torch.cat((grouped, separators), dim=2).flatten(1, 2)
            mask = torch.cat((grouped_mask, grouped_mask[:, :, -1:]), dim=2).flatten(
                1, 2
            )
        if self.user_id_buckets:
            buckets = batch.get("user_bucket")
            if (
                not isinstance(buckets, torch.Tensor)
                or buckets.dtype != torch.long
                or buckets.shape != (ids.size(0),)
                or bool(((buckets < 0) | (buckets >= self.user_id_buckets)).any())
            ):
                raise ValueError("GRID requires one valid user_bucket per sample")
            user_embeds = self.model.shared(buckets + self.sid_token_vocab_size)
            positions = mask.size(1) - mask.sum(dim=1).long()
            embeds = F.pad(embeds, (0, 0, 1, 0)).scatter(
                1,
                positions[:, None, None].expand(-1, 1, embeds.size(-1)),
                user_embeds[:, None],
            )
            mask = F.pad(mask, (1, 0)).scatter(1, positions[:, None], 1)
        # Preserve cold users: one visible PAD embedding keeps attention defined.
        mask[~mask.bool().any(1), 0] = 1
        return embeds, mask

    def encode_users(self, batch: Mapping[str, Any]) -> tuple[Any, torch.Tensor]:
        """Encode inputs once and return encoder outputs and a mask for decoder reuse."""
        encoder_embeds, attention_mask = self.build_encoder_inputs(batch)
        encoder_outputs = self.model.get_encoder()(
            inputs_embeds=encoder_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return encoder_outputs, attention_mask

    def _validate_target_labels(self, labels: torch.Tensor) -> None:
        """Require one complete SID per active target in the frozen level vocabularies."""

        if labels.dtype != torch.long:
            raise TypeError("GRID target labels must have torch.long dtype")
        if labels.ndim != 2 or not labels.size(0) or labels.size(1) != self.code_len:
            raise ValueError("GRID target labels must have shape (B, code_len)")
        for level, size in enumerate(self.level_sizes):
            if not bool(
                torch.isin(labels[:, level], self.tokens_by_level[level, :size]).all()
            ):
                raise ValueError(f"GRID target tokens are outside SID level {level}")

    def _decoder_logits(
        self,
        encoder_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> torch.Tensor:
        """Teacher-force a complete SID, or compute the first step from BOS only."""

        decoder_ids = (
            self.model._shift_right(labels)
            if labels is not None
            else torch.full(
                (encoder_hidden.size(0), 1),
                self.model.config.decoder_start_token_id,
                dtype=torch.long,
                device=encoder_hidden.device,
            )
        )
        decoder_embeds = self.model.shared(decoder_ids)
        if self.bos_token is not None:
            decoder_embeds = torch.cat(
                (
                    self.bos_token[None].expand(decoder_ids.size(0), -1, -1),
                    decoder_embeds[:, 1:],
                ),
                1,
            )
        outputs = self.model.get_decoder()(
            inputs_embeds=decoder_embeds,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return self._project_decoder_logits(outputs.last_hidden_state)

    def _project_decoder_logits(
        self, hidden: torch.Tensor, first_level: int = 0
    ) -> torch.Tensor:
        """Use the same head and token probabilities during teacher forcing and generation."""

        if self.output_head == "shared":
            # Native T5 rescales hidden states before its tied vocabulary projection.
            return self.model.lm_head(hidden * self.model.model_dim**-0.5)
        logits = []
        for position in range(hidden.size(1)):
            level = first_level + position
            values = self.output_heads[level](hidden[:, position])
            tokens = self.tokens_by_level[level, : self.level_sizes[level]]
            logits.append(
                values.new_full((hidden.size(0), self.vocab_size), -torch.inf).scatter(
                    1, tokens[None].expand(hidden.size(0), -1), values
                )
            )
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def generate_codes_with_scores(
        self,
        batch: Mapping[str, Any] | None = None,
        num_beams: int = 20,
        code_len: int | None = None,
        allowed_codes: tuple[tuple[int, ...], ...] | None = None,
        encoder_outputs: Any | None = None,
        attention_mask: torch.Tensor | None = None,
        *,
        constraint: str = "prefix",
        renormalize_logits: bool = False,
        use_cache: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate full SIDs and their summed token log-probabilities in eval mode.

        Output sequences include one decoder-start placeholder. Prefix constraints
        prune invalid paths, optionally renormalizing their token probabilities.
        Free search can yield unmapped SIDs, filtered by the inference layer.
        """

        from .generation import generate_codes_with_scores

        return generate_codes_with_scores(
            self,
            batch,
            num_beams,
            code_len,
            allowed_codes,
            encoder_outputs,
            attention_mask,
            constraint=constraint,
            renormalize_logits=renormalize_logits,
            use_cache=use_cache,
        )

    def _target_losses(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Return per-target CE with an explicit mean or sum over full-SID tokens."""

        losses = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="none",
        ).view(labels.shape)
        return losses.mean(1) if self.token_loss_reduction == "mean" else losses.sum(1)

    def _independent_target_chunk_loss(
        self,
        encoder_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        user_indices: torch.Tensor,
        labels: torch.Tensor,
        pair_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Decode one bounded chunk of independent positive target SIDs."""

        selected_hidden = encoder_hidden.index_select(0, user_indices)
        selected_attention_mask = attention_mask.index_select(0, user_indices)
        logits = self._decoder_logits(selected_hidden, selected_attention_mask, labels)
        return (self._target_losses(logits, labels) * pair_weights).sum()

    def _forward_independent_targets(
        self,
        batch: Mapping[str, Any],
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Average independent target losses with one encoder pass and bounded decoding."""

        if labels.ndim != 3 or labels.size(2) != self.code_len:
            raise ValueError(
                "Independent target labels must have shape (B, T, code_len)"
            )
        target_mask = batch.get("multi_target_mask")
        if target_mask is None or target_mask.shape != labels.shape[:2]:
            raise ValueError(
                "Independent target labels require a matching multi_target_mask"
            )
        target_mask = target_mask.to(device=labels.device, dtype=torch.bool)
        targets_per_user = target_mask.sum(dim=1)
        if not bool((targets_per_user > 0).all()):
            raise ValueError("Every training sample must have an active target SID")
        valid_labels = labels[target_mask]
        self._validate_target_labels(valid_labels)
        encoder_outputs, attention_mask = self.encode_users(batch)
        encoder_hidden = encoder_outputs.last_hidden_state
        batch_size = labels.size(0)
        if batch_size != encoder_hidden.size(0):
            raise ValueError("GRID labels and histories must have the same batch size")
        user_grid = torch.arange(batch_size, device=labels.device).unsqueeze(1)
        user_indices = user_grid.expand_as(target_mask)[target_mask]
        if batch.get("augmented_windows", False):
            total_windows = torch.tensor(
                batch_size, device=labels.device, dtype=torch.float32
            )
            world_size = 1
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(total_windows, op=dist.ReduceOp.SUM)
                world_size = dist.get_world_size()
            # DDP averages ranks. Average windows globally within a microbatch;
            # upstream accumulation then averages these microbatch means.
            pair_weights = (
                targets_per_user.to(dtype=torch.float32)
                .reciprocal()
                .index_select(0, user_indices)
                * world_size
                / total_windows
            )
        elif self.multi_target_loss_reduction == "mean_tokens_then_targets_global":
            target_count = torch.tensor(
                valid_labels.size(0), dtype=torch.long, device=labels.device
            )
            world_size = 1
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(target_count, op=dist.ReduceOp.SUM)
                world_size = dist.get_world_size()
            self.loss_normalizer = target_count.to(dtype=torch.float32)
            pair_weights = torch.ones(
                valid_labels.size(0), dtype=torch.float32, device=labels.device
            ) * (world_size / target_count.float())
        else:
            pair_weights = (
                targets_per_user.to(dtype=torch.float32)
                .reciprocal()
                .index_select(0, user_indices)
                / batch_size
            )
        loss = encoder_hidden.new_zeros(())
        chunk_size = self.multi_target_decoder_chunk_size
        use_checkpoint = (
            self.decoder_activation_checkpointing
            and self.training
            and torch.is_grad_enabled()
        )
        for start in range(0, valid_labels.size(0), chunk_size):
            stop = min(start + chunk_size, valid_labels.size(0))
            chunk_args = (
                encoder_hidden,
                attention_mask,
                user_indices[start:stop],
                valid_labels[start:stop],
                pair_weights[start:stop],
            )
            if use_checkpoint:
                chunk_loss = checkpoint(
                    self._independent_target_chunk_loss,
                    *chunk_args,
                    use_reentrant=False,
                )
            else:
                chunk_loss = self._independent_target_chunk_loss(*chunk_args)
            loss = loss + chunk_loss
        return loss, encoder_hidden.new_empty((batch_size, 0, self.vocab_size))

    def forward(
        self,
        batch: Mapping[str, Any],
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Return loss and logits using the same forward interface as OneRec.

        Labels (B, L) yield full-vocabulary logits (B, L, V). Independent targets
        (B, T, L) use their explicit mask and return empty (B, 0, V) logits. Without
        labels, only the first decoder step is evaluated; beam search is separate.
        """

        self.loss_normalizer = None
        if labels is not None:
            if labels.ndim == 3:
                return self._forward_independent_targets(batch, labels)
            self._validate_target_labels(labels)
        encoder_outputs, attention_mask = self.encode_users(batch)
        encoder_hidden = encoder_outputs.last_hidden_state
        if labels is not None and labels.size(0) != encoder_hidden.size(0):
            raise ValueError("GRID labels and histories must have the same batch size")
        logits = self._decoder_logits(encoder_hidden, attention_mask, labels)
        loss = (
            self._target_losses(logits, labels).mean() if labels is not None else None
        )
        return loss, logits
