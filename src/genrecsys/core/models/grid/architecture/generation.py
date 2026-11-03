"""Batched fixed-width SID beam search with T5 KV caching and optional prefix pruning."""

import math

import torch

from genrecsys.core.models.semantic_generation import CodebookPrefixLogitsProcessor


def _prefix_constraint(model, codes):
    """Cache an immutable candidate trie and the number of distinct prefixes per level."""

    if not codes:
        raise ValueError("prefix generation requires nonempty allowed_codes")
    key = id(codes)
    if key not in model._generation_constraints:
        namespaces = [
            set(model.tokens_by_level[level, :size].tolist())
            for level, size in enumerate(model.level_sizes)
        ]
        if len(set(codes)) != len(codes) or any(
            len(code) != model.code_len
            or any(
                type(token) is not int or token not in namespaces[level]
                for level, token in enumerate(code)
            )
            for code in codes
        ):
            raise ValueError(
                "allowed_codes must contain unique complete SIDs from the frozen vocabulary"
            )
        if model.vocab_size ** (model.code_len - 1) > 2**63:
            raise ValueError("GRID compact SID prefix encoding exceeds int64")
        processor = CodebookPrefixLogitsProcessor(codes, model.code_len)
        widths = [
            len({code[: level + 1] for code in codes})
            for level in range(model.code_len)
        ]
        # processor retains codes, keeping id(codes) valid for the cache lifetime.
        model._generation_constraints[key] = (processor, widths)
    return model._generation_constraints[key]


def generate_codes_with_scores(
    model,
    batch,
    num_beams,
    code_len,
    allowed_codes,
    encoder_outputs,
    attention_mask,
    *,
    constraint,
    renormalize_logits,
    use_cache,
):
    """Search with raw or conditionally normalized constrained token probabilities."""

    if model.training:
        raise ValueError("GRID generation requires model.eval()")
    if not isinstance(num_beams, int) or isinstance(num_beams, bool) or num_beams <= 0:
        raise ValueError("GRID num_beams must be a positive integer")
    if code_len is not None and code_len != model.code_len:
        raise ValueError("GRID generation must use the full frozen SID width")
    if constraint not in {"none", "prefix"}:
        raise ValueError("GRID generation constraint must be none or prefix")
    if not isinstance(use_cache, bool):
        raise TypeError("GRID use_cache must be a boolean")
    if not isinstance(renormalize_logits, bool):
        raise TypeError("GRID renormalize_logits must be a boolean")
    processor = None
    if constraint == "prefix":
        processor, widths = _prefix_constraint(model, allowed_codes)
    elif model.output_head == "per_level":
        widths = [
            math.prod(model.level_sizes[: level + 1]) for level in range(model.code_len)
        ]
    else:
        widths = [model.vocab_size ** (level + 1) for level in range(model.code_len)]
    if encoder_outputs is None:
        if batch is None:
            raise ValueError("GRID generation requires batch or encoder_outputs")
        encoder_outputs, attention_mask = model.encode_users(batch)
    if attention_mask is None:
        raise ValueError("attention_mask is required with encoder_outputs")
    hidden = encoder_outputs.last_hidden_state
    batch_size = hidden.size(0)
    if not batch_size or attention_mask.shape != hidden.shape[:2]:
        raise ValueError(
            "GRID encoder outputs and attention_mask must have shape (B, S)"
        )
    start = model.model.config.decoder_start_token_id
    sequences = torch.full(
        (batch_size, 1), start, dtype=torch.long, device=hidden.device
    )
    scores = torch.zeros(batch_size, dtype=torch.float32, device=hidden.device)
    cache, beam_width = None, 1
    for level in range(model.code_len):
        decoder_ids = sequences[:, -1:] if cache is not None else sequences
        embeddings = model.model.shared(decoder_ids)
        if cache is None and model.bos_token is not None:
            embeddings = torch.cat(
                (
                    model.bos_token[None].expand(len(sequences), -1, -1),
                    embeddings[:, 1:],
                ),
                dim=1,
            )
        outputs = model.model.get_decoder()(
            inputs_embeds=embeddings,
            encoder_hidden_states=hidden,
            encoder_attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=use_cache,
            return_dict=True,
        )
        logits = model._project_decoder_logits(
            outputs.last_hidden_state[:, -1:], first_level=level
        )[:, 0]
        log_probs = logits.float().log_softmax(dim=-1)
        if bool(torch.isnan(log_probs).any()):
            raise ValueError("GRID generation returned NaN token probabilities")
        if processor is not None:
            log_probs = processor(sequences, log_probs)
            log_probs.masked_fill_(
                log_probs == torch.finfo(log_probs.dtype).min, -torch.inf
            )
            if renormalize_logits:
                normalizer = log_probs.logsumexp(dim=-1, keepdim=True)
                # Dead beam placeholders have no valid children. Keep their -inf
                # scores rather than producing NaN by subtracting -inf from -inf.
                log_probs = log_probs - normalizer.masked_fill(
                    ~torch.isfinite(normalizer), 0.0
                )
        cumulative = log_probs + scores[:, None]
        next_width = min(num_beams, widths[level])
        scores, selected = cumulative.reshape(batch_size, -1).topk(next_width, dim=-1)
        parents = selected // model.vocab_size
        tokens = selected % model.vocab_size
        rows = (
            torch.arange(batch_size, device=hidden.device)[:, None] * beam_width
            + parents
        ).reshape(-1)
        sequences = torch.cat(
            (sequences.index_select(0, rows), tokens.reshape(-1, 1)), dim=1
        )
        scores = scores.reshape(-1)
        if use_cache:
            cache = outputs.past_key_values
            if hasattr(cache, "reorder_cache"):
                cache.reorder_cache(rows)
            else:
                cache = tuple(
                    tuple(value.index_select(0, rows) for value in layer)
                    for layer in cache
                )
        hidden = hidden.index_select(0, rows)
        attention_mask = attention_mask.index_select(0, rows)
        beam_width = next_width
    # Different users may retain different numbers of live trie paths after pruning.
    # Keep rectangular batches with -inf-scored placeholders; inference drops them.
    if not bool(torch.isfinite(scores.reshape(batch_size, -1)).any(dim=1).all()):
        raise ValueError("GRID beam search returned no valid path")
    return sequences, scores
