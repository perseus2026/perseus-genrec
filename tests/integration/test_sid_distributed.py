"""Real torchrun fits on CPU; CUDA-only SyncBatchNorm checks are opt-in by hardware."""

import json
import math
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from accelerate.data_loader import prepare_data_loader
from test_sid_single_accelerator import plain_training
from torch.utils.data import DataLoader, TensorDataset

from item_representations.semantic_ids.mapping import fit_sid_mapper
from item_representations.semantic_ids.models import RQVAE, RVQ, BaseSIDSGen
from item_representations.semantic_ids.models.distributed import (
    check_loss,
    training_accelerator,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("world_size", [None, "1"])
def test_single_process_uses_accelerator_without_process_group(monkeypatch, world_size):
    if world_size is None:
        monkeypatch.delenv("WORLD_SIZE", raising=False)
    else:
        monkeypatch.setenv("WORLD_SIZE", world_size)

    accelerator = training_accelerator("cpu")
    assert accelerator.num_processes == 1
    assert accelerator.mixed_precision == "no"
    assert not torch.distributed.is_initialized()
    accelerator.wait_for_everyone()


@pytest.mark.parametrize("count,batch,world", [(23, 7, 2), (3, 8, 4), (24, 6, 2)])
def test_even_batches_and_single_process_tail(count, batch, world):
    loader = DataLoader(TensorDataset(torch.arange(count)), batch_size=batch)
    single = training_accelerator("cpu").prepare_data_loader(
        loader, device_placement=False
    )
    assert [i for (rows,) in single for i in rows.tolist()] == list(range(count))
    partitions = [
        [
            rows.tolist()
            for (rows,) in prepare_data_loader(
                loader,
                num_processes=world,
                process_index=rank,
                split_batches=False,
                even_batches=True,
            )
        ]
        for rank in range(world)
    ]
    expected_steps = math.ceil(count / (batch * world))
    assert all(len(batches) == expected_steps for batches in partitions)
    assert all(len(rows) == batch for batches in partitions for rows in batches)
    combined = [i for step in zip(*partitions) for rows in step for i in rows]
    assert combined[:count] == list(range(count))
    assert len(combined) == expected_steps * batch * world


def launch(tmp_path, mode, device="cpu"):
    if os.environ.get("RUN_DISTRIBUTED_TESTS") != "1":
        pytest.skip("requires local distributed sockets; set RUN_DISTRIBUTED_TESTS=1")
    env = {
        **os.environ,
        "PYTHONPATH": "src",
        "OMP_NUM_THREADS": "1",
        "POLARS_MAX_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    # Avoid accidentally inheriting an unrelated Accelerate launch configuration.
    for key in list(env):
        if key.startswith("ACCELERATE_"):
            del env[key]
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    # Static localhost rendezvous also works on laptops without resolvable hostnames.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    result = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            "-B",
            "-m",
            "torch.distributed.run",
            "--master-addr=127.0.0.1",
            f"--master-port={port}",
            "--nproc-per-node=2",
            str(Path(__file__).resolve()),
            str(tmp_path),
            mode,
            device,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-7000:]


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                torch.cuda.device_count() < 2, reason="requires two CUDA GPUs"
            ),
        ),
    ],
)
def test_two_process_fits_publish_portable_mappers(tmp_path, device):
    launch(tmp_path, "fit", device)
    for name in ("rvq-sum", "rvq-mean", "rq_vae"):
        a = torch.load(tmp_path / f"{name}-rank0.pt", weights_only=False)
        b = torch.load(tmp_path / f"{name}-rank1.pt", weights_only=False)
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        output = tmp_path / name
        meta = json.loads((output / "sid_mapper.json").read_text())
        method = "rvq" if name.startswith("rvq") else "rq_vae"
        assert meta[method]["distributed"]["world_size"] == 2
        assert meta[method]["distributed"]["nominal_global_batch_size"] == 14
        model = BaseSIDSGen.load(output / "checkpoint.pkl")
        assert isinstance(model, (RVQ, RQVAE))
        assert not any(
            isinstance(m, torch.nn.parallel.DistributedDataParallel)
            for m in model.modules()
        )
        for row in model.fit_diagnostics:
            assert row["optimizer_updates"] == 4
            assert row["examples_seen"] == 56  # 23 rows padded to 28, two epochs.
        x = np.load(tmp_path / "x.npy")
        codes = model.transform(x)
        assert codes.shape == (23, 2)
        assert len(pl.read_parquet(output / "item_to_sid.parquet")) == 23


def test_two_process_gradients_match_averaged_local_loss_reference(tmp_path):
    launch(tmp_path, "reference")


def test_nonfinite_loss_fails_on_all_ranks(tmp_path):
    launch(tmp_path, "nonfinite")


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires two CUDA GPUs for SyncBatchNorm"
)
def test_cuda_sync_batchnorm_and_cpu_checkpoint(tmp_path):
    launch(tmp_path, "normalized", "cuda")
    model = BaseSIDSGen.load(tmp_path / "normalized/checkpoint.pkl")
    model.device = torch.device("cpu")
    model.cpu()
    x = np.load(tmp_path / "x.npy")
    assert model.transform(x).shape == (23, 2)
    a = torch.load(
        tmp_path / "normalized-rank0.pt", map_location="cpu", weights_only=False
    )
    b = torch.load(
        tmp_path / "normalized-rank1.pt", map_location="cpu", weights_only=False
    )
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    # Explicit full-catalog initialization, then four global training forwards.
    assert model.encoder.normalization[0].num_batches_tracked.item() == 5


def worker(root, mode, device):
    accelerator = training_accelerator(device)
    rank = accelerator.process_index
    if mode == "nonfinite":
        loss = torch.tensor(float("nan") if rank else 1.0, device=accelerator.device)
        try:
            check_loss(loss, accelerator)
        except FloatingPointError:
            return
        raise AssertionError("All ranks must reject the failed loss")
    if mode == "reference":
        x = np.random.default_rng(71).normal(size=(24, 5)).astype("float32")
        for reduction in ("sum", "mean"):
            options = {
                "n_levels": 2,
                "codebook_size": (3, 3),
                "init_buffer_size": 8,
                "device": "cpu",
                "random_state": 42,
                "loss_reduction": reduction,
            }
            model = RVQ(**options, loss_scale=0.125).fit(
                x,
                epochs_per_layer=2,
                batch_size=6,
                accelerator=accelerator,
            )
            # Evaluate the same global batches without launching distributed execution.
            reference = RVQ(
                **options, loss_scale=0.125 / (2 if reduction == "sum" else 1)
            ).fit(
                x,
                epochs_per_layer=2,
                batch_size=12,
                accelerator=plain_training("cpu"),
            )
            for a, b in zip(model.codebooks, reference.codebooks):
                torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-6)
        options = {
            "input_dim": 5,
            "hidden_dims": (4, 2),
            "codebook_size": (3, 3),
            "device": "cpu",
            "random_state": 42,
            "codebook_kmeans_init": False,
            "gradient_clip": None,
        }
        model = RQVAE(**options).fit(x, epochs=2, batch_size=6, accelerator=accelerator)
        reference = RQVAE(**options).fit(
            x, epochs=2, batch_size=12, accelerator=plain_training("cpu")
        )
        for key, value in model.state_dict().items():
            torch.testing.assert_close(
                value, reference.state_dict()[key], rtol=2e-6, atol=2e-6
            )
        return
    x = np.random.default_rng(17).normal(size=(23, 8)).astype("float32")
    if accelerator.is_main_process:
        np.save(root / "x.npy", x)
        pl.DataFrame({"item_id": [f"{i:03}" for i in range(23)]}).write_parquet(
            root / "items.pq"
        )
    torch.distributed.barrier()
    x = np.load(root / "x.npy", mmap_mode="r")
    names = (
        ("normalized",) if mode == "normalized" else ("rvq-sum", "rvq-mean", "rq_vae")
    )
    for name in names:
        method = "rvq" if name.startswith("rvq") else "rq_vae"
        torch.manual_seed(
            100 + rank
        )  # Initialization must not depend on ambient rank RNG.
        options = {
            "codebook_size": (3, 3),
            "random_state": 42,
            "device": str(accelerator.device),
        }
        if method == "rvq":
            model = RVQ(
                **options,
                n_levels=2,
                init_buffer_size=8,
                loss_reduction=name.split("-")[1],
            )
            model.fit(x, epochs_per_layer=2, batch_size=7, accelerator=accelerator)
        else:
            model = RQVAE(
                8,
                hidden_dims=(4, 2),
                **options,
                joint_normalization=mode == "normalized",
            )
            model.fit(
                x,
                epochs=2,
                batch_size=7,
                accelerator=accelerator,
                kmeans_init_batch_size=32 if mode == "normalized" else None,
            )
        torch.save(
            {k: v.detach().cpu() for k, v in model.state_dict().items()},
            root / f"{name}-rank{rank}.pt",
        )
        fit_sid_mapper(
            root / "items.pq",
            root / "x.npy",
            root / name,
            method=method,
            **options,
            hidden_dim_factors=(2, 4),
            init_buffer_size=8,
            epochs=2,
            epochs_per_layer=2,
            training_batch_size=7,
            eval_batch_size=8,
            loss_reduction="mean" if name == "rvq-mean" else "sum",
            joint_normalization=mode == "normalized",
            kmeans_init_batch_size=32 if mode == "normalized" else None,
        )
        torch.distributed.barrier()


if __name__ == "__main__":
    worker(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
