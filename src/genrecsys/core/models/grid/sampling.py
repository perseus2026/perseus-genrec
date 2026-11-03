"""User-first contiguous-window sampling, matching upstream GRID's collator."""

import math
from bisect import bisect_right
from itertools import accumulate

import torch


def contiguous_windows(batch, *, max_windows_per_batch, max_history_items=None):
    """Sample upstream GRID's contiguous subsequences without enumerating O(L²) rows.

    Enumerate by user, end, start, then sample with replacement and deduplicate,
    exactly as the released collator does. The final item is the sole target.
    The cap applies to each rank's user microbatch, before gradient accumulation.
    """
    counts = []
    for row in batch:
        width = row["code_length"]
        sequence = row["training_sequence"]
        if len(sequence) % width:
            raise ValueError("augmentation requires complete SIDs")
        length = len(sequence) // width
        counts.append(length * (length - 1) // 2)
    total = sum(counts)
    if not total:
        raise ValueError(
            "contiguous augmentation requires at least two items in a training sequence"
        )
    selected = (
        torch.randint(total, (max_windows_per_batch,)).unique(sorted=True).tolist()
        if total > max_windows_per_batch
        else range(total)
    )
    boundaries = list(accumulate(counts))
    output = []
    for index in selected:
        user = bisect_right(boundaries, index)
        local = index - (boundaries[user - 1] if user else 0)
        end = (1 + math.isqrt(1 + 8 * local)) // 2 + 1
        start = local - (end - 2) * (end - 1) // 2
        row = batch[user]
        width = row["code_length"]
        if max_history_items is not None:
            start = max(start, end - 1 - max_history_items)
        output.append(
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key != "training_sequence"
                },
                "input_ids": row["training_sequence"][
                    start * width : (end - 1) * width
                ],
                "independent_targets": [
                    row["training_sequence"][(end - 1) * width : end * width]
                ],
            }
        )
    return output
