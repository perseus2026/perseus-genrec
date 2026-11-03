"""Aligned left padding for native event sequence inputs."""

import torch


def _pad_input(values: list[dict]) -> dict[str, torch.Tensor]:
    """Left-pad each aligned feature to the input's batch width or fixed source cap."""
    features = values[0]["features"]
    first = next(iter(features))
    lengths = [len(value["features"][first]["values"]) for value in values]
    width = max(max(lengths), values[0]["padding_length"])
    mask = torch.zeros((len(values), width), dtype=torch.long)
    daily_recency = torch.zeros((len(values), width), dtype=torch.long)
    output = {}
    for name, description in features.items():
        dtype = getattr(torch, description["dtype"])
        tensor = torch.zeros((len(values), width, description["width"]), dtype=dtype)
        for index, length in enumerate(lengths):
            if length:
                tensor[index, -length:] = torch.as_tensor(
                    values[index]["features"][name]["values"], dtype=dtype
                )
                mask[index, -length:] = 1
                daily_recency[index, -length:] = torch.as_tensor(
                    values[index].get("daily_recency", [0] * length),
                    dtype=torch.long,
                )
        output[name] = tensor
    return {
        "features": output,
        "attention_mask": mask,
        "daily_recency": daily_recency,
    }
