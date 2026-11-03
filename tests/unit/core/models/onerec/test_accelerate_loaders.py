"""Exercise Accelerate's real batch sharder locally, without distributed sockets."""

import pytest
import torch
from accelerate.data_loader import prepare_data_loader
from torch.utils.data import DataLoader


@pytest.mark.parametrize("rows", [1, 3, 5, 8, 9])
@pytest.mark.parametrize("world_size", [2, 8])
@pytest.mark.parametrize("shuffle", [False, True])
def test_one_owner_per_row_before_tail_padding(rows, world_size, shuffle):
    loaders = [
        prepare_data_loader(
            DataLoader(range(rows), batch_size=2, shuffle=shuffle),
            num_processes=world_size,
            process_index=rank,
            split_batches=False,
            dispatch_batches=False,
            even_batches=True,
            use_seedable_sampler=True,
            data_seed=7,
            rng_types=[],
        )
        for rank in range(world_size)
    ]
    for epoch in range(2):
        rank_batches = []
        for loader in loaders:
            loader.set_epoch(epoch)
            rank_batches.append(list(loader))
        assert len({len(batches) for batches in rank_batches}) == 1
        merged = [torch.cat(batch).tolist() for batch in zip(*rank_batches)]
        assert all(len(batch) == world_size * 2 for batch in merged)
        # gather_for_metrics truncates only the final global batch to this size.
        if remainder := rows % (world_size * 2):
            merged[-1] = merged[-1][:remainder]
        assert sorted(index for batch in merged for index in batch) == list(range(rows))
