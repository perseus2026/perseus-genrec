"""Shared semantic-ID constraints for HuggingFace beam generation."""

import logging

import torch
from transformers import LogitsProcessor

logger = logging.getLogger(__name__)


class CodebookLevelLogitsProcessor(LogitsProcessor):
    """Mask generation logits so each SID position uses its own codebook level."""

    def __init__(
        self,
        allowed_tokens_by_level: tuple[tuple[int, ...], ...],
        code_len: int,
    ):
        """Store allowed tokens per SID level and initialize the vocabulary-mask cache."""

        self.allowed_tokens_by_level = allowed_tokens_by_level
        self.code_len = code_len
        self._mask_cache: dict[tuple[int, str, int], torch.Tensor] = {}

        if len(allowed_tokens_by_level) != code_len:
            raise ValueError(
                "allowed_tokens_by_level must contain one token list per SID level: "
                f"{len(allowed_tokens_by_level)} != {code_len}"
            )
        for level, tokens in enumerate(allowed_tokens_by_level):
            if not tokens:
                raise ValueError(f"SID level {level + 1} has no allowed tokens")

    def _mask_for(
        self,
        level: int,
        device: torch.device,
        vocab_size: int,
    ) -> torch.Tensor:
        """Build or reuse the level's vocabulary mask on the target device."""

        key = (level, str(device), vocab_size)
        if key not in self._mask_cache:
            mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
            tokens = torch.tensor(
                self.allowed_tokens_by_level[level],
                dtype=torch.long,
                device=device,
            )
            tokens = tokens[(tokens >= 0) & (tokens < vocab_size)]
            if tokens.numel() == 0:
                raise ValueError(
                    f"SID level {level + 1} has no tokens within vocab_size={vocab_size}"
                )
            mask[tokens] = True
            self._mask_cache[key] = mask
        return self._mask_cache[key]

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Mask logits for tokens disallowed at the current SID level."""

        generated_len = max(0, input_ids.size(1) - 1)
        level = generated_len % self.code_len
        mask = self._mask_for(level, scores.device, scores.size(-1))
        return scores.masked_fill(~mask.unsqueeze(0), torch.finfo(scores.dtype).min)


class CodebookPrefixLogitsProcessor(LogitsProcessor):
    """Constrain generation to allowed SID codes using compact prefix tables.

    Encode prefixes as int64 keys and store children in indexed tensors. Build
    tables lazily on the inference device and release temporary CPU arrays.
    """

    def __init__(
        self,
        allowed_codes: tuple[tuple[int, ...], ...],
        code_len: int,
    ):
        """Store the full-SID whitelist and initialize compact prefix-table caches."""

        self.allowed_codes = allowed_codes
        self.code_len = code_len
        if not allowed_codes:
            raise ValueError("allowed_codes must contain at least one SID")
        if code_len < 1:
            raise ValueError("code_len must be positive")
        self._level_cache: dict[tuple[str, int], list] = {}

    def _level_tables(
        self,
        device: torch.device,
        vocab_size: int,
    ) -> list:
        """Build and cache compact CSR-like trie tables on one device."""
        cache_key = (str(device), int(vocab_size))
        if cache_key in self._level_cache:
            return self._level_cache[cache_key]

        base = int(vocab_size)
        if base < 1:
            raise ValueError("vocab_size must be positive")
        # Only parent prefixes need integer keys; the final token is a child.
        if base ** (self.code_len - 1) > 2**63:
            raise ValueError(
                "Compact SID prefix encoding exceeds int64: "
                f"vocab_size={base}, code_len={self.code_len}"
            )
        try:
            codes = torch.tensor(self.allowed_codes, dtype=torch.int32)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(
                "allowed_codes must be a rectangular integer matrix"
            ) from exc
        if codes.ndim != 2 or codes.size(1) != self.code_len:
            observed = tuple(codes.shape) if codes.ndim else ()
            raise ValueError(
                "Every allowed code must have length "
                f"{self.code_len}, observed tensor shape={observed}"
            )
        if bool(((codes < 0) | (codes >= base)).any().item()):
            raise ValueError(f"allowed_codes contain tokens outside [0, {base})")

        path_keys = torch.zeros(codes.size(0), dtype=torch.int64)
        tables: list = []
        for level in range(self.code_len):
            # Keep (parent, child) separate so a full SID never overflows int64.
            edges = torch.unique(
                torch.stack((path_keys, codes[:, level]), dim=1),
                sorted=True,
                dim=0,
            )
            parent_for_edge = edges[:, 0]
            children = edges[:, 1].to(torch.int32)
            parent_keys, counts = torch.unique_consecutive(
                parent_for_edge,
                return_counts=True,
            )
            offsets = torch.empty(parent_keys.numel() + 1, dtype=torch.int64)
            offsets[0] = 0
            offsets[1:] = counts.cumsum(dim=0)
            tables.append(
                (
                    parent_keys.to(device),
                    offsets.to(device),
                    children.to(device),
                )
            )
            logger.info(
                "Built compact SID constraint level %d/%d: parents=%s edges=%s "
                "max_children=%s",
                level + 1,
                self.code_len,
                f"{parent_keys.numel():,}",
                f"{children.numel():,}",
                int(counts.max().item()),
            )
            if level + 1 < self.code_len:
                path_keys.mul_(base).add_(codes[:, level])
            del edges, parent_for_edge, children, parent_keys, counts, offsets

        del codes, path_keys
        self._level_cache[cache_key] = tables
        return tables

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Mask scores in place to allow only valid SID-prefix continuations.

        Process parent groups separately to avoid a dense batch * beams * max_children
        allocation for high-fanout buckets.
        """

        generated_len = max(0, input_ids.size(1) - 1)
        level = generated_len % self.code_len

        tables = self._level_tables(scores.device, scores.size(-1))
        parent_keys, offsets, children = tables[level]
        rows = scores.size(0)
        vocab_size = scores.size(-1)

        row_keys = scores.new_zeros(rows, dtype=torch.long)
        # ``-0`` is ``0`` in Python: slicing with ``[:, -level:]`` at the
        # root would accidentally encode the complete decoder input (including
        # decoder_start_token_id) instead of selecting the root node.
        if level:
            for prefix_token in input_ids[:, -level:].to(torch.long).unbind(dim=1):
                row_keys.mul_(vocab_size).add_(prefix_token)

        pos = torch.searchsorted(parent_keys, row_keys).clamp_(
            max=parent_keys.numel() - 1
        )
        matched = parent_keys[pos] == row_keys

        # Do not pad every row to the largest fan-out. With 256x1 SIDs the
        # collision token makes vocab_size ~78k and one prefix can have ~65k
        # children. A dense [batch*beams, max_children] int64 matrix consumed
        # several GiB and OOMed H100 evaluation. Rows sharing a parent are few
        # (at most one beam per prefix and user at this level), so preserve the
        # allowed values one parent group at a time and mask logits in place.
        min_score = torch.finfo(scores.dtype).min
        if bool((~matched).any().item()):
            scores[~matched] = min_score

        matched_parent_positions = torch.unique(pos[matched])
        if matched_parent_positions.numel():
            parent_bounds = (
                torch.stack(
                    (
                        offsets[matched_parent_positions],
                        offsets[matched_parent_positions + 1],
                    ),
                    dim=1,
                )
                .cpu()
                .tolist()
            )
            parent_position_values = matched_parent_positions.cpu().tolist()
            for parent_position, (start, stop) in zip(
                parent_position_values,
                parent_bounds,
            ):
                row_indices = torch.nonzero(
                    matched & (pos == parent_position),
                    as_tuple=False,
                ).flatten()
                child_tokens = children[start:stop].to(torch.long)
                allowed_values = scores[
                    row_indices.unsqueeze(1),
                    child_tokens.unsqueeze(0),
                ]
                scores.index_fill_(0, row_indices, min_score)
                scores[
                    row_indices.unsqueeze(1),
                    child_tokens.unsqueeze(0),
                ] = allowed_values

        return scores
