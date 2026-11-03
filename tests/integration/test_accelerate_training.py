"""Opt-in two-process CPU test of the real OneRec training CLI and validation."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from genrecsys import api, core


@pytest.mark.skipif(
    os.environ.get("RUN_DISTRIBUTED_TESTS") != "1",
    reason="requires local distributed sockets; set RUN_DISTRIBUTED_TESTS=1",
)
@pytest.mark.parametrize("cosine_checkpointing", [False, True])
def test_two_process_training(tmp_path, monkeypatch, cosine_checkpointing):
    from test_lifecycle import (
        _synthetic_basis,
        _synthetic_config,
        _write_synthetic_eventhub,
    )

    hub = tmp_path / "hub"
    _write_synthetic_eventhub(hub)
    monkeypatch.setenv("INTERNAL_STORAGE_EVENT_HUB", str(hub))
    config = _synthetic_config()
    config.training.num_epochs = 2
    config.training.dataloader.num_workers = 1
    if cosine_checkpointing:
        from genrecsys.core.config import Scheduler

        config.training.gradient_accumulation_steps = 2
        config.training.gradient_checkpointing = True
        config.training.scheduler = Scheduler(
            "cosine_warmup", {"warmup_ratio": 0.1, "min_lr_ratio": 0.1}
        )
        config.validate()
    train, validation, train_artifacts, val_artifacts, resources, mappings = (
        _synthetic_basis()
    )
    prepared, dataset = api.prepare_dataset(
        config,
        train,
        validation,
        train_artifacts=train_artifacts,
        validation_artifacts=val_artifacts,
        resources=resources,
        event_feature_mappings=mappings,
    )
    dataset.save(tmp_path / "dataset")
    prepared.save(tmp_path / "checkpoint")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--rdzv-backend=c10d",
            "--rdzv-endpoint=127.0.0.1:0",
            "--local-addr=127.0.0.1",
            "--nproc-per-node=2",
            str(Path(__file__).resolve()),
            str(tmp_path),
        ],
        env={
            **os.environ,
            "ACCELERATE_USE_CPU": "true",
            "OMP_NUM_THREADS": "1",
            "POLARS_MAX_THREADS": "1",
            "GLOO_SOCKET_IFNAME": "lo0" if sys.platform == "darwin" else "lo",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "rank-0-ok").exists()
    assert (tmp_path / "rank-1-ok").exists()
    trained = core.checkpoint.Trained.load(tmp_path / "checkpoint")
    assert len(trained.model.training["history"]) == 2
    if cosine_checkpointing:
        # One rank-local update per epoch: peak after warmup, then the floor.
        # A world-size-multiplied scheduler would reach the floor in epoch one.
        assert trained.model.training["history"][0]["learning_rate"] == pytest.approx(
            config.training.optimizer.params["lr"]
        )
        assert trained.model.training["history"][-1]["learning_rate"] == pytest.approx(
            config.training.optimizer.params["lr"] * 0.1
        )


def _collate_validation(indices):
    return {
        "user_id": indices,
        "target_items": [[index] for index in indices],
        "target_item_labels": [[1.0] for _ in indices],
    }


def _worker(workdir):
    from accelerate import Accelerator, DataLoaderConfiguration
    from accelerate.state import PartialState
    from torch.utils.data import DataLoader

    from genrecsys.cli.train import fit_model
    from genrecsys.core.evaluation import MetricSpec
    from genrecsys.core.models.onerec import inference, training

    torch.multiprocessing.set_start_method("spawn", force=True)
    accelerator = Accelerator(
        cpu=True,
        dataloader_config=DataLoaderConfiguration(
            split_batches=False,
            dispatch_batches=False,
            even_batches=True,
            use_seedable_sampler=True,
            data_seed=0,
        ),
    )
    # This worker is CPU-only. Accelerate passes device_ids even for Gloo,
    # which makes PyTorch 2.12 dispatch the barrier to unsupported MPS on macOS.
    # Use the real CPU barrier here; production/CUDA synchronization is untouched.
    PartialState.wait_for_everyone = lambda self: torch.distributed.barrier()
    original = inference._predict_batch
    inference._predict_batch = lambda model, batch, *args, **kwargs: [
        [{"item": 0, "score": 1.0}] for _ in batch["user_id"]
    ]
    try:
        # Includes fewer rows than ranks and uneven/padded final global batches.
        for count in (1, 3, 5, 8, 9):
            loader = accelerator.prepare_data_loader(
                DataLoader(
                    range(count),
                    batch_size=2,
                    collate_fn=_collate_validation,
                ),
                device_placement=False,
            )
            metrics = training._evaluate(
                torch.nn.Identity(),
                loader,
                (),
                {},
                (MetricSpec("hit@1", "hit_rate_at_k", 1),),
                accelerator.device,
                accelerator=accelerator,
                beam_size=1,
                candidate_count=1,
            )
            assert metrics["hit@1"] == pytest.approx(1 / count)
        # Synchronized shuffling covers every row exactly once before padding;
        # changing the epoch changes order, but never sample ownership coverage.
        loader = accelerator.prepare_data_loader(
            DataLoader(range(8), batch_size=2, shuffle=True)
        )
        orders = []
        for epoch in range(2):
            loader.set_epoch(epoch)
            indices = torch.cat(
                [accelerator.gather_for_metrics(batch) for batch in loader]
            ).tolist()
            assert sorted(indices) == list(range(8))
            orders.append(indices)
        assert orders[0] != orders[1]
    finally:
        inference._predict_batch = original
    fit_model(workdir=workdir)
    (workdir / f"rank-{accelerator.process_index}-ok").touch()


if __name__ == "__main__":
    _worker(Path(sys.argv[1]))
