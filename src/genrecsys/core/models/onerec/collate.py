"""One padding path for all declared inputs; target padding remains independent."""

import torch
from torch.nn.utils.rnn import pad_sequence

from genrecsys.core.inputs.collate import _pad_input


def _pad_independent_targets(
    values: list[list[list[int]]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad independent per-user target SIDs to (B, T, code_len).

    Keep each target as a separate sequence; mark padded target rows with -100
    so cross-entropy ignores them.
    """

    if not values or any(not user_targets for user_targets in values):
        raise ValueError("Every independent-target sample must contain a target")
    code_lengths = {len(target) for user_targets in values for target in user_targets}
    if len(code_lengths) != 1:
        raise ValueError(
            "Independent targets must have one uniform SID width, got "
            f"{sorted(code_lengths)}"
        )
    code_len = next(iter(code_lengths))
    if code_len <= 0:
        raise ValueError("Independent target SID width must be positive")

    max_targets = max(len(user_targets) for user_targets in values)
    targets = torch.full(
        (len(values), max_targets, code_len),
        -100,
        dtype=torch.long,
    )
    mask = torch.zeros((len(values), max_targets), dtype=torch.bool)
    for user_index, user_targets in enumerate(values):
        count = len(user_targets)
        targets[user_index, :count] = torch.as_tensor(
            user_targets,
            dtype=torch.long,
        )
        mask[user_index, :count] = True
    return targets, mask


def onerec_collate_fn(batch: list[dict]) -> dict:
    """Collate named inputs, train SID labels or raw evaluation targets."""
    output = {
        "user_id": [row["user_id"] for row in batch],
        "inputs": {
            name: _pad_input([row["inputs"][name] for row in batch])
            for name in batch[0]["inputs"]
        },
    }
    if "reference_time" in batch[0]:
        output["reference_time"] = torch.tensor(
            [row["reference_time"] for row in batch], dtype=torch.float64
        )
    if "independent_targets" in batch[0]:
        output["target"], output["multi_target_mask"] = _pad_independent_targets(
            [row["independent_targets"] for row in batch]
        )
    elif "target" in batch[0]:
        output["target"] = pad_sequence(
            [torch.tensor(row["target"], dtype=torch.long) for row in batch],
            batch_first=True,
            padding_value=-100,
        )
    elif "target_items" in batch[0]:
        output["target_items"] = [row["target_items"] for row in batch]
        output["target_item_labels"] = [row["target_item_labels"] for row in batch]
    return output
