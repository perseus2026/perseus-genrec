"""Opt-in local Gloo coverage of HSTU checkpointing, mmap workers and validation."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("RUN_DISTRIBUTED_TESTS") != "1",
    reason="requires local distributed sockets; set RUN_DISTRIBUTED_TESTS=1",
)
@pytest.mark.parametrize(
    "loss_type,fusion",
    [("cross_entropy", "sum"), ("sampled_cross_entropy", "concat_projection")],
)
def test_two_process_hstu_training(tmp_path, monkeypatch, loss_type, fusion):
    from test_hstu_lifecycle import hstu_config, prepared_fixture

    from genrecsys import core

    prepared, dataset = prepared_fixture(tmp_path, monkeypatch, with_context=True)
    config = hstu_config(
        prepared.config, "mean_targets_global", True, loss_type, fusion
    )
    config.model.params["weight_initialization"] = "perseus"
    config.training.dataloader.num_workers = 1
    prepared.config = config
    prepared.save(tmp_path / "checkpoint")
    dataset.save(tmp_path / "dataset")
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
    trained = core.Trained.load(tmp_path / "checkpoint")
    assert trained.model.training["history"][0]["Recall@2"] == 1
    assert trained.model.training["objective"] == "item_cross_entropy"
    assert trained.model.training["context_features"] == ["region"]


def _worker(workdir):
    import torch
    from accelerate import Accelerator
    from accelerate.state import PartialState

    from genrecsys.cli.train import fit_model

    torch.set_num_threads(1)
    torch.multiprocessing.set_start_method("spawn", force=True)
    # Initialize the shared PartialState explicitly on CPU before the CLI's
    # local_main_process_first context (macOS otherwise detects MPS here).
    Accelerator(cpu=True)
    # PyTorch on macOS can route a Gloo barrier with device_ids through MPS.
    # Match the existing CPU distributed harness; CUDA code is unaffected.
    PartialState.wait_for_everyone = lambda self: torch.distributed.barrier()
    fit_model(workdir=workdir)
    (workdir / f"rank-{torch.distributed.get_rank()}-ok").touch()


if __name__ == "__main__":
    _worker(Path(sys.argv[1]))
