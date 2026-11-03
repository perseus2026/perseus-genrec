"""Shared learned-position indexing for sequential recommendation models."""

import torch


def encoder_position_ids(seq_len: int, *, device: torch.device) -> torch.Tensor:
    """Return left-to-right absolute positions for the concatenated encoder input."""
    return torch.arange(seq_len, device=device).unsqueeze(0)


def qformer_recency_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return batch-independent recency IDs for left-padded, oldest-first histories.

    Assign padding 0, the newest event 1, and progressively older events 2, 3, etc.
    These are indices for OneRec's trainable position embeddings, not learned values.
    """
    if attention_mask.ndim != 2:
        raise ValueError(
            "QFormer attention mask must have shape (batch, sequence), got "
            f"{tuple(attention_mask.shape)}"
        )
    visible = attention_mask.to(dtype=torch.long)
    reverse_counts = torch.flip(
        torch.cumsum(torch.flip(visible, dims=(1,)), dim=1),
        dims=(1,),
    )
    return reverse_counts * visible
