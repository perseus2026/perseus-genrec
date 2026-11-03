"""Pad one GRID SID history in whole-item groups and keep targets independent."""

import torch
from torch.nn.utils.rnn import pad_sequence

from genrecsys.core.models.grid.sampling import contiguous_windows


def _pad_input(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Left-pad to a complete SID boundary, including batches containing only cold users."""

    lengths = {row["code_length"] for row in batch}
    if len(lengths) != 1 or next(iter(lengths)) <= 0:
        raise ValueError("GRID batch requires one positive SID width")
    code_length = next(iter(lengths))
    values = [row["input_ids"] for row in batch]
    if any(len(value) % code_length for value in values):
        raise ValueError("GRID histories must contain complete SIDs")
    width = max(code_length, max(len(value) for value in values))
    ids = torch.zeros((len(values), width), dtype=torch.long)
    mask = torch.zeros_like(ids)
    for index, value in enumerate(values):
        if value:
            ids[index, -len(value) :] = torch.as_tensor(value, dtype=torch.long)
            mask[index, -len(value) :] = 1
    return {"input_ids": ids, "attention_mask": mask}


def grid_collate_fn(
    batch: list[dict],
    *,
    augmentation: dict | None = None,
    max_history_items: int | None = None,
) -> dict:
    """Collate train SID labels, raw evaluation targets or prediction-only samples."""

    if not batch:
        raise ValueError("GRID cannot collate an empty batch")
    if augmentation is not None:
        batch = contiguous_windows(
            batch, max_history_items=max_history_items, **augmentation
        )
    elif any("training_sequence" in row for row in batch):
        raise ValueError("training_sequence requires explicit augmentation settings")
    modes = {"independent_targets", "target", "target_items"}
    mode = modes.intersection(batch[0])
    if len(mode) > 1 or any(modes.intersection(row) != mode for row in batch):
        raise ValueError("GRID batch must use one consistent target mode")
    output = {"user_id": [row["user_id"] for row in batch], **_pad_input(batch)}
    if augmentation is not None:
        output["augmented_windows"] = True
    if any("user_bucket" in row for row in batch):
        if not all("user_bucket" in row for row in batch):
            raise ValueError("GRID batch must use user tokens for every row or none")
        output["user_bucket"] = torch.tensor(
            [row["user_bucket"] for row in batch], dtype=torch.long
        )
    if "reference_time" in batch[0]:
        output["reference_time"] = torch.tensor(
            [row["reference_time"] for row in batch], dtype=torch.float64
        )
    if "independent_targets" in mode:
        output["target"], output["multi_target_mask"] = _pad_independent_targets(
            [row["independent_targets"] for row in batch]
        )
    elif "target" in mode:
        output["target"] = pad_sequence(
            [torch.tensor(row["target"], dtype=torch.long) for row in batch],
            batch_first=True,
            padding_value=-100,
        )
    elif "target_items" in mode:
        output["target_items"] = [row["target_items"] for row in batch]
        output["target_item_labels"] = [row["target_item_labels"] for row in batch]
    return output


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
