"""OneRec retrieval architecture assembled from the ordered inputs in config."""

from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import (
    LogitsProcessorList,
    SwitchTransformersConfig,
    SwitchTransformersForConditionalGeneration,
    T5Config,
    T5ForConditionalGeneration,
)
from transformers.modeling_outputs import BaseModelOutput

from genrecsys.core.models.position import (
    encoder_position_ids,
    qformer_recency_position_ids,
)
from genrecsys.core.models.semantic_generation import (
    CodebookLevelLogitsProcessor,
    CodebookPrefixLogitsProcessor,
)
from genrecsys.core.models.sequence import EventEncoder, QFormerCompressor

from .initialization import initialize_onerec_weights


class OneRec(nn.Module):
    """Train a dense HF encoder-decoder on SID targets with configurable input streams."""

    def __init__(self, config: dict[str, Any]):
        """Create the backbone, shared event projections and optional per-input QFormers."""
        super().__init__()
        self.code_len = int(config["code_len"])
        self.sid_token_vocab_size = int(config["vocab_size"])
        self.d_model = int(config["d_model"])
        # Independent targets otherwise multiply decoder activations by targets/user.
        self.multi_target_decoder_chunk_size = int(
            config.get("multi_target_decoder_chunk_size", 128)
        )
        self.decoder_activation_checkpointing = config.get(
            "decoder_activation_checkpointing", True
        )
        self.multi_target_loss_reduction = config.get(
            "multi_target_loss_reduction",
            "mean_tokens_then_targets_per_sample",
        )
        # Global target count of the current forward, used to normalize the
        # complete optimizer update rather than each accumulation microbatch.
        self.loss_normalizer: torch.Tensor | None = None
        if self.multi_target_loss_reduction not in {
            "mean_tokens_then_targets_per_sample",
            "mean_tokens_then_targets_global",
        }:
            raise ValueError(
                "multi_target_loss_reduction must be "
                "mean_tokens_then_targets_per_sample or "
                "mean_tokens_then_targets_global"
            )
        self.inputs = config["inputs"]
        self.positions = set(
            config.get("position_embeddings", ["absolute", "qformer_recency"])
        )
        self.model = self._build_hf_backbone(config)
        self.event_encoders = nn.ModuleDict()
        self.compressors = nn.ModuleDict()
        self.empty_tokens = nn.ParameterDict()
        self.input_positions = nn.ModuleDict()
        self.input_types = (
            nn.Embedding(len(self.inputs), self.d_model)
            if config.get("input_type_embeddings", True)
            else None
        )
        self.decoder_level_embedding = (
            nn.Embedding(self.code_len, self.d_model)
            if config.get("decoder_level_embeddings", False)
            else None
        )
        self.max_encoder_positions = sum(
            value["compression"]["num_queries"]
            if value["compression"]
            else value["max_len"]
            * (
                next(
                    f["prefix_length"]
                    for f in value["features"]
                    if f["type"] == "semantic_id"
                )
                if value.get("token_layout", "event") == "sid_sequence"
                else 1
            )
            for value in self.inputs
        )
        self.position_embedding = (
            nn.Embedding(self.max_encoder_positions, self.d_model)
            if "absolute" in self.positions
            else None
        )
        self.daily_recency_position = (
            nn.Embedding(
                int(config["daily_recency_vocab_size"]),
                self.d_model,
                padding_idx=0,
            )
            if "daily_recency" in self.positions
            else None
        )
        for value in self.inputs:
            key = value["event_encoder"]
            if key not in self.event_encoders:
                self.event_encoders[key] = EventEncoder(
                    value, self.d_model, config["dropout_rate"]
                )
            if value["compression"] is None:
                continue
            name, compression = value["name"], value["compression"]
            self.compressors[name] = QFormerCompressor(
                d_model=self.d_model,
                num_heads=compression["num_heads"],
                d_ff=compression["dim_feedforward"],
                num_queries=compression["num_queries"],
                num_layers=compression["num_layers"],
                dropout=config["dropout_rate"],
            )
            self.empty_tokens[name] = nn.Parameter(
                torch.empty(compression["num_queries"], self.d_model)
            )
            nn.init.normal_(self.empty_tokens[name], std=self.d_model**-0.5)
            if "qformer_recency" in self.positions:
                self.input_positions[name] = nn.Embedding(
                    value["max_len"] + 1, self.d_model, padding_idx=0
                )
        self._logits_processor_cache = {}
        self.weight_initialization = config.get("weight_initialization", "default")
        initialize_onerec_weights(self, self.weight_initialization)
        if self.decoder_level_embedding is not None:
            self.model.register_forward_pre_hook(
                self._inject_decoder_level_embeddings,
                with_kwargs=True,
            )

    @staticmethod
    def _build_hf_backbone(config: dict[str, Any]) -> nn.Module:
        """Construct T5/Switch with independently configured encoder/decoder relative bias."""
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
        params.update(
            num_layers=config["num_encoder_layers"],
            decoder_start_token_id=config["pad_token_id"],
        )
        kind = config.get("backbone", "t5")
        if kind == "t5":
            backbone = T5ForConditionalGeneration(
                T5Config(**params, feed_forward_proj="gated-gelu")
            )
        elif kind == "switch_dense":
            backbone = SwitchTransformersForConditionalGeneration(
                SwitchTransformersConfig(
                    **params,
                    dense_act_fn="gelu_new",
                    num_experts=1,
                    expert_capacity=64,
                    num_sparse_encoder_layers=0,
                    num_sparse_decoder_layers=0,
                    router_bias=False,
                    router_jitter_noise=0.0,
                    router_dtype="float32",
                    router_ignore_padding_tokens=True,
                    router_aux_loss_coef=0.0,
                    router_z_loss_coef=0.0,
                    add_router_probs=False,
                )
            )
        else:
            raise ValueError(f"backbone must be one of: t5, switch_dense; got {kind!r}")
        for stack, keep in (
            (backbone.encoder, "relative" in config.get("position_embeddings", [])),
            (backbone.decoder, config.get("decoder_relative_attention", False)),
        ):
            if not keep:
                for module in stack.modules():
                    if getattr(module, "has_relative_attention_bias", False):
                        module.has_relative_attention_bias = False
                        del module.relative_attention_bias
        return backbone

    def _inject_decoder_level_embeddings(
        self,
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Add the current semantic-ID level to decoder token embeddings."""

        input_ids = kwargs.get("decoder_input_ids")
        if input_ids is None or kwargs.get("decoder_inputs_embeds") is not None:
            return args, kwargs
        cache_position = kwargs.get("cache_position")
        if cache_position is None or cache_position.numel() != input_ids.shape[1]:
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        else:
            positions = cache_position.to(device=input_ids.device)
        positions = positions.remainder(self.code_len)
        kwargs["decoder_inputs_embeds"] = (
            module.get_input_embeddings()(input_ids)
            + self.decoder_level_embedding(positions)[None]
        )
        kwargs["decoder_input_ids"] = None
        return args, kwargs

    def build_encoder_inputs(
        self, batch: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project inputs in declaration order and compress only streams requesting QFormer."""
        parts, masks = [], []
        for index, description in enumerate(self.inputs):
            name = description["name"]
            values = batch["inputs"][name]
            mask = values["attention_mask"]
            event_mask = mask
            encoder = self.event_encoders[description["event_encoder"]]
            tokens = encoder(values["features"], self.model.get_input_embeddings())
            width = encoder.tokens_per_event
            if width != 1:
                mask = mask.repeat_interleave(width, dim=1)
            if self.daily_recency_position is not None:
                recency = self.daily_recency_position(values["daily_recency"])
                if width != 1:
                    recency = recency.repeat_interleave(width, dim=1)
                tokens = tokens + recency
            if name in self.compressors:
                if name in self.input_positions:
                    recency = self.input_positions[name](
                        qformer_recency_position_ids(event_mask)
                    )
                    if width != 1:
                        recency = recency.repeat_interleave(width, dim=1)
                    tokens = tokens + recency
                empty = ~mask.bool().any(1)
                safe_mask = mask.clone()
                safe_mask[empty, 0] = 1
                tokens = self.compressors[name](tokens, safe_mask)
                tokens = torch.where(
                    empty[:, None, None], self.empty_tokens[name][None], tokens
                )
                mask = torch.ones(
                    tokens.shape[:2], dtype=torch.long, device=tokens.device
                )
            if self.input_types is not None:
                tokens = tokens + self.input_types.weight[index]
            parts.append(tokens)
            masks.append(mask)
        encoder_inputs, attention_mask = torch.cat(parts, 1), torch.cat(masks, 1)
        if self.position_embedding is not None:
            ids = encoder_position_ids(
                encoder_inputs.shape[1], device=encoder_inputs.device
            )
            encoder_inputs = encoder_inputs + self.position_embedding(ids)
        # A completely cold user needs a visible token for attention/pooling.
        attention_mask = attention_mask.clone()
        attention_mask[~attention_mask.bool().any(1), 0] = 1
        return encoder_inputs, attention_mask

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
        decoder_input_ids = self.model._shift_right(labels)
        outputs = self.model(
            encoder_outputs=BaseModelOutput(last_hidden_state=selected_hidden),
            attention_mask=selected_attention_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            return_dict=True,
        )
        token_losses = F.cross_entropy(
            outputs.logits.float().reshape(-1, outputs.logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(labels.shape)
        valid_tokens = labels.ne(-100)
        pair_losses = (token_losses * valid_tokens).sum(dim=1) / valid_tokens.sum(
            dim=1
        ).clamp_min(1)
        return (pair_losses * pair_weights).sum()

    def _forward_independent_targets(
        self,
        batch: Mapping[str, Any],
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Average independent target losses with one encoder pass.

        Chunk decoder work by multi_target_decoder_chunk_size. During training, use
        activation checkpointing to avoid retaining decoder activations for all chunks.
        Global target reduction compensates for DDP's subsequent rank-wise averaging.
        """

        if labels.ndim != 3:
            raise ValueError(
                "Independent multi-target labels must have shape "
                "(batch, targets, code_len)"
            )
        if labels.size(2) != self.code_len:
            raise ValueError(
                "Independent target SID width differs from model code_len: "
                f"{labels.size(2)} != {self.code_len}"
            )
        target_mask = batch.get("multi_target_mask")
        if target_mask is None:
            raise ValueError(
                "Independent multi-target labels require multi_target_mask"
            )
        target_mask = target_mask.to(device=labels.device, dtype=torch.bool)
        if target_mask.shape != labels.shape[:2]:
            raise ValueError(
                "multi_target_mask shape differs from target prefix shape: "
                f"{tuple(target_mask.shape)} != {tuple(labels.shape[:2])}"
            )
        targets_per_user = target_mask.sum(dim=1)
        encoder_embeds, attention_mask = self.build_encoder_inputs(batch)
        encoder_outputs = self.model.get_encoder()(
            inputs_embeds=encoder_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        encoder_hidden = encoder_outputs.last_hidden_state
        batch_size = labels.size(0)
        user_grid = torch.arange(batch_size, device=labels.device).unsqueeze(1)
        user_grid = user_grid.expand_as(target_mask)
        user_indices = user_grid[target_mask]
        valid_labels = labels[target_mask]
        if self.multi_target_loss_reduction == "mean_tokens_then_targets_global":
            target_count = torch.tensor(
                valid_labels.size(0),
                dtype=torch.long,
                device=labels.device,
            )
            world_size = 1
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(target_count, op=dist.ReduceOp.SUM)
                world_size = dist.get_world_size()
            self.loss_normalizer = target_count.to(dtype=torch.float32)
            pair_weights = torch.ones(
                valid_labels.size(0),
                dtype=torch.float32,
                device=labels.device,
            ) * (world_size / target_count.float())
        else:
            pair_weights = (
                targets_per_user.to(dtype=encoder_hidden.dtype)
                .reciprocal()
                .index_select(0, user_indices)
                / batch_size
            )

        chunk_size = self.multi_target_decoder_chunk_size or batch_size
        chunk_size = max(1, chunk_size)
        loss = encoder_hidden.new_zeros(())
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

        empty_logits = encoder_hidden.new_empty(
            (batch_size, 0, self.sid_token_vocab_size)
        )
        return loss, empty_logits

    def forward(
        self,
        batch: Mapping[str, Any],
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Return SID cross-entropy loss and decoder logits.

        Labels (B, L) produce logits (B, L, vocab). Independent targets (B, T, L)
        use the configured target reduction and return empty (B, 0, vocab) logits.
        Global target reduction exposes its denominator as loss_normalizer.
        """
        self.loss_normalizer = None
        if labels is not None and labels.ndim == 3:
            return self._forward_independent_targets(batch, labels)

        encoder_embeds, attention_mask = self.build_encoder_inputs(batch)
        decoder = {"labels": labels}
        if labels is not None and self.decoder_level_embedding is not None:
            decoder["decoder_input_ids"] = self.model._shift_right(labels)
        outputs = self.model(
            inputs_embeds=encoder_embeds,
            attention_mask=attention_mask,
            **decoder,
            return_dict=True,
        )
        return outputs.loss, outputs.logits

    def _generation_logits_processor(
        self,
        allowed_tokens_by_level: tuple[tuple[int, ...], ...] | None,
        allowed_codes: tuple[tuple[int, ...], ...] | None,
        code_len: int,
    ) -> LogitsProcessorList | None:
        """Reuse the generation constraint processor for the catalog or SID levels.

        A full-SID whitelist takes precedence over level masks. Return None when no
        constraints are configured; cache prefix tables between calls.
        """

        processors = []
        if allowed_codes is not None:
            cache_key = ("prefix", id(allowed_codes), code_len)
            if cache_key not in self._logits_processor_cache:
                self._logits_processor_cache[cache_key] = LogitsProcessorList(
                    [
                        CodebookPrefixLogitsProcessor(
                            allowed_codes=allowed_codes,
                            code_len=code_len,
                        )
                    ]
                )
            processors.extend(self._logits_processor_cache[cache_key])
        elif allowed_tokens_by_level is not None:
            cache_key = ("level", id(allowed_tokens_by_level), code_len)
            if cache_key not in self._logits_processor_cache:
                self._logits_processor_cache[cache_key] = LogitsProcessorList(
                    [
                        CodebookLevelLogitsProcessor(
                            allowed_tokens_by_level=allowed_tokens_by_level,
                            code_len=code_len,
                        )
                    ]
                )
            processors.extend(self._logits_processor_cache[cache_key])
        return LogitsProcessorList(processors) if processors else None

    @torch.no_grad()
    def generate_codes_with_scores(
        self,
        batch: Mapping[str, Any],
        num_beams: int = 20,
        code_len: int = 3,
        allowed_tokens_by_level: tuple[tuple[int, ...], ...] | None = None,
        allowed_codes: tuple[tuple[int, ...], ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate one item SID per beam and its normalized sequence log-probability."""

        encoder_embeds, attention_mask = self.build_encoder_inputs(batch)
        generated = self.model.generate(
            inputs_embeds=encoder_embeds,
            attention_mask=attention_mask,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            do_sample=False,
            max_new_tokens=code_len,
            min_new_tokens=code_len,
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.pad_token_id,
            logits_processor=self._generation_logits_processor(
                allowed_tokens_by_level,
                allowed_codes,
                code_len,
            ),
            return_dict_in_generate=True,
            output_scores=True,
        )
        token_scores = self.model.compute_transition_scores(
            generated.sequences,
            generated.scores,
            generated.beam_indices,
            normalize_logits=True,
        )
        sequence_scores = token_scores[:, :code_len].sum(dim=1)
        return generated.sequences, sequence_scores
