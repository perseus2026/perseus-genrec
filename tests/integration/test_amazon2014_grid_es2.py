"""Fresh ES2 inputs and author configuration contracts; no remote access/CUDA fit."""

import importlib.util
import json
import pickle
import shutil
import sys

import polars as pl
import pytest
import torch
import yaml
from test_amazon2014_grid_embeddings import _inputs

from model_datasets.amazon2014_grid.basis import build_basis
from model_datasets.amazon2014_grid.prepare import check_source_inputs


def helper(name):
    spec = importlib.util.spec_from_file_location(
        name.replace("-", "_"),
        f"scripts/amazon2014-grid/{'' if name == 'grid-quantizers' else 'es2/'}{name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def grid(tmp_path, mock_grid_download):
    spec = importlib.util.spec_from_file_location(
        "restore_grid", "scripts/amazon2014-grid/restore-grid.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mock_grid_download(module, upstream=True)
    return module.restore(tmp_path / "GRID-main")


@pytest.mark.parametrize("method,devices", [("rkmeans", 1), ("rvq", 1), ("rqvae", 1)])
def test_native_hydra_composition(grid, method, devices):
    hydra = pytest.importorskip("hydra")
    from omegaconf import OmegaConf

    common = [
        "hydra/job_logging=default",
        "hydra/hydra_logging=default",
        "seed=44",
    ]
    with hydra.initialize_config_dir(
        config_dir=str(grid / "configs"), version_base="1.3"
    ):
        train = hydra.compose(
            config_name="train",
            overrides=[f"experiment=es2/{method}_train", *common],
        )
        infer = hydra.compose(
            config_name="inference",
            overrides=[f"experiment=es2/{method}_inference", *common],
        )
    for config in (train, infer):
        assert (
            config.embedding_dim,
            config.num_hierarchies,
            config.codebook_width,
        ) == (2048, 3, 256)
    assert train.trainer.devices == list(range(devices))
    assert train.trainer.sync_batchnorm == (devices == 8)
    assert (
        train.data_loading.datamodule.train_dataloader_config.batch_size_per_device
        == 16384 // devices
    )
    assert train.trainer.max_steps == (15000 if method == "rqvae" else 3000)
    assert train.trainer.precision == "bf16-mixed"
    assert train.trainer.accumulate_grad_batches == 1
    assert train.trainer.strategy == "ddp_find_unused_parameters_true"
    assert train.trainer.max_epochs == -1
    assert train.should_skip_retry and train.callbacks.early_stopping is None
    assert train.callbacks.model_checkpoint.save_last
    assert infer.trainer.devices == [0] and infer.trainer.strategy == "ddp"
    assert not infer.trainer.sync_batchnorm
    assert infer.data_loading.datamodule.predict_dataloader_config.num_workers == 0
    assert train.model.normalize_residuals == (method != "rqvae")
    expected = {"rkmeans": "SGD", "rvq": "Adam", "rqvae": "Adagrad"}[method]
    assert train.optim.optimizer._target_ == f"torch.optim.{expected}"
    assert train.model.scheduler is None
    original = OmegaConf.to_container(train.model, resolve=False)
    prediction = OmegaConf.to_container(infer.model, resolve=False)
    prediction.pop("loss_function", None)
    prediction.pop("evaluator", None)
    assert prediction == original


def checkpoint(output, method, seed, devices):
    state = {
        "global_step": 15000 if method == "rqvae" else 3000,
        "layers_initialized": [True] * 3,
        "current_layer": 0 if method == "rqvae" else 2,
    }
    cfg = {
        "seed": seed,
        "num_hierarchies": 3,
        "trainer": {
            "max_steps": state["global_step"],
            "devices": 8 if devices == 8 else [0],
            "sync_batchnorm": devices == 8,
            "precision": "bf16-mixed",
            "accumulate_grad_batches": 1,
        },
        "data_loading": {
            "datamodule": {
                "train_dataloader_config": {"batch_size_per_device": 16384 // devices}
            }
        },
        "model": {"normalize_residuals": method != "rqvae"},
    }
    (output / "train/checkpoints").mkdir(parents=True)
    (output / "train/.hydra").mkdir()
    torch.save(state, output / "train/checkpoints/last.ckpt")
    (output / "train/.hydra/config.yaml").write_text(yaml.safe_dump(cfg))
    return state, cfg


@pytest.mark.parametrize(
    "experiment,method,seed,available",
    [
        ("es1", "rkmeans", 42, 8),
        ("es1", "rkmeans", 42, 1),
        ("es2", "rkmeans", 44, 8),
        ("es2", "rvq", 44, 8),
        ("es2", "rqvae", 44, 8),
    ],
)
def test_fresh_data_to_import_and_shared_preparation(
    tmp_path, grid, monkeypatch, experiment, method, seed, available
):
    api = helper("grid-quantizers")
    data = tmp_path / "data"
    _inputs(data)  # synthetic raw TFRecords and text-encoder output
    (data / "source").rename(data / "prepared")
    check_source_inputs(
        prepared_source_dir=data / "prepared",
        source_dir=data / "raw",
        output_path=data / "stage01/runtime/source-inputs.json",
        scope="items",
    )
    build_basis(
        prepared_source_dir=data / "prepared",
        stage01_dir=data / "stage01",
        output_dir=data / "basis",
        category="beauty",
        augmentation="none",
        max_history=20,
    )
    output = data / "grid-sids" if experiment == "es1" else tmp_path / "new-quantizer"
    devices = 8 if experiment == "es1" and available >= 8 else 1
    calls = []

    def fake_author(command, **kwargs):
        # Only external GRID calls are simulated; import/materialize/prepare are real.
        calls.append(command)
        assert kwargs["env"]["PYTHONPATH"] == str(grid)
        if "src.train" in command:
            assert (
                "experiment=reproduce_rkmeans_train"
                if experiment == "es1"
                else f"experiment=es2/{method}_train"
            ) in command
            if experiment == "es1":
                assert f"trainer.devices={devices}" in command
                assert (
                    "data_loading.datamodule.train_dataloader_config."
                    f"batch_size_per_device={16384 // devices}"
                ) in command
            else:
                assert not any(arg.startswith("trainer.devices=") for arg in command)
            checkpoint(output, method, seed, devices)
        elif "src.inference" in command:
            assert not any(arg.startswith("trainer.devices=") for arg in command)
            root = output / "inference/pickle"
            root.mkdir(parents=True)
            codes = torch.tensor([[i, i % 5, i % 3, 0] for i in range(27)])
            torch.save(codes.T, root / "merged_predictions_tensor.pt")
            with (root / "merged_predictions.pkl").open("wb") as stream:
                pickle.dump(
                    [{"item_id": i, "cluster_ids": codes[i, :3]} for i in range(27)],
                    stream,
                )

    with monkeypatch.context() as patch:
        patch.setattr(api.subprocess, "run", fake_author)

        def probe(*args, **kwargs):
            assert experiment == "es1"
            return str(available)

        patch.setattr(api.subprocess, "check_output", probe)
        # The venv interpreter symlink must not resolve to the system Python.
        interpreter = tmp_path / "grid-venv/bin/python"
        interpreter.parent.mkdir(parents=True)
        interpreter.symlink_to(sys.executable)
        api.fit(
            experiment,
            data / "raw",
            data,
            output,
            grid,
            interpreter,
            "beauty",
            method,
            seed,
        )
        assert [call[2] for call in calls] == ["src.train", "src.inference"]
        assert all(call[0] == str(interpreter) for call in calls)
        with pytest.raises(FileExistsError):
            api.fit(
                experiment,
                data / "raw",
                data,
                output,
                grid,
                interpreter,
                "beauty",
                method,
                seed,
            )
    if experiment == "es1":
        assert (data / "sids/item_to_sid.pq").is_file()
        assert not (output / "imported").exists()
    record = json.loads((output / "runtime/sid-validation.json").read_text())
    assert record["train_devices"] == devices
    assert record["batch_size_per_device"] * devices == 16384
    assert record["method"] == method and record["seed"] == seed
    assert not (output / "recipe.json").exists()
    prepared = tmp_path / "prepared-dataset"
    if experiment == "es1":
        from test_amazon2014_grid_lifecycle import command

        command(
            "bash",
            "scripts/amazon2014-grid/prepare.sh",
            str(data),
            str(prepared),
            "configs/amazon2014-grid/beauty.yaml",
        )
    else:
        from test_amazon2014_grid_lifecycle import command

        command(
            "bash",
            "scripts/amazon2014-grid/prepare.sh",
            str(data),
            str(prepared),
            "configs/amazon2014-grid/beauty.yaml",
            str(output / "imported/item_to_sid.pq"),
            str(output / "imported/sid_import_report.json"),
        )
    assert (prepared / "checkpoint/data_contract.json").is_file()
    assert (prepared / "resources/sid_import_report.json").is_file()
    assert (prepared / "dataset/events").is_dir()
    for split in ("train", "validation", "test"):
        samples = pl.read_parquet(prepared / f"basis/{split}/samples.pq")
        assert len(samples) == 5 and samples["history_length"].to_list() == [20] * 5
    if experiment == "es2" and method == "rqvae":
        # Native fits use the same preparation, without a GRID import report.
        native = tmp_path / "native-fit"
        (native / "inference").mkdir(parents=True)
        shutil.copy2(
            output / "imported/item_to_sid.pq", native / "inference/item_to_sid.parquet"
        )
        native_prepared = tmp_path / "native-prepared"
        command(
            "bash",
            "scripts/amazon2014-grid/prepare.sh",
            str(data),
            str(native_prepared),
            "configs/amazon2014-grid/beauty.yaml",
            str(native / "inference/item_to_sid.parquet"),
            "",
        )
        assert (native_prepared / "dataset/events").is_dir()
        assert not (native_prepared / "resources/sid_import_report.json").exists()
        assert (native_prepared / "checkpoint/data_contract.json").read_bytes() == (
            prepared / "checkpoint/data_contract.json"
        ).read_bytes()


@pytest.mark.parametrize("failed_phase", ["src.train", "src.inference"])
def test_author_failure_stops_before_import(tmp_path, monkeypatch, failed_phase):
    import subprocess

    api = helper("grid-quantizers")
    data = tmp_path / "data"
    (data / "stage01/runtime").mkdir(parents=True)
    (data / "basis/representations").mkdir(parents=True)
    (data / "stage01/runtime/source-inputs.json").write_text('{"category": "beauty"}')
    (data / "basis/representations/metadata.json").write_text("{}")
    calls = []

    def fail(command, **kwargs):
        assert kwargs["check"] is True
        calls.append(command[2])
        if command[2] == failed_phase:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(api.subprocess, "run", fail)
    output = tmp_path / "fit"
    with pytest.raises(subprocess.CalledProcessError):
        api.fit(
            "es2",
            tmp_path / "raw",
            data,
            output,
            tmp_path / "grid",
            tmp_path / "python",
            "beauty",
            "rvq",
            42,
        )
    assert calls == (
        ["src.train"] if failed_phase == "src.train" else ["src.train", "src.inference"]
    )
    assert not (output / "imported").exists()
    assert not (output / "runtime/sid-validation.json").exists()


@pytest.mark.parametrize("available", [0, 1, 2, 4, 7, 8, 16])
def test_es1_resources_preserve_batch_and_hydra_recipe(grid, monkeypatch, available):
    api = helper("grid-quantizers")
    env = {"CUDA_VISIBLE_DEVICES": "test-mask"}

    def probe(command, **kwargs):
        assert command[0] == "/author/bin/python"
        assert "torch.cuda.device_count()" in command[-1]
        assert kwargs["env"] is env
        return f"{available}\n"

    monkeypatch.setattr(api.subprocess, "check_output", probe)
    if available == 0:
        with pytest.raises(RuntimeError, match="at least one visible CUDA GPU"):
            api.es1_resource_overrides("/author/bin/python", env)
        return
    overrides = api.es1_resource_overrides("/author/bin/python", env)
    hydra = pytest.importorskip("hydra")
    with hydra.initialize_config_dir(
        config_dir=str(grid / "configs"), version_base="1.3"
    ):
        cfg = hydra.compose(
            config_name="train",
            overrides=[
                "experiment=reproduce_rkmeans_train",
                "hydra/job_logging=default",
                "hydra/hydra_logging=default",
                *overrides,
            ],
        )
    assert cfg.trainer.devices == (8 if available >= 8 else 1)
    batch = cfg.data_loading.datamodule.train_dataloader_config.batch_size_per_device
    assert batch * cfg.trainer.devices == 16384
    assert cfg.trainer.accumulate_grad_batches == 1
    assert cfg.trainer.max_steps == 3000
    assert cfg.seed == 42 and cfg.model.normalize_residuals

    loader = cfg.data_loading.datamodule.train_dataloader_config
    if cfg.trainer.devices == 1:
        assert loader.num_workers == 0 and loader.timeout == 0
        assert not loader.pin_memory and not loader.persistent_workers
    else:
        assert loader.num_workers == 12 and loader.timeout == 60
        assert loader.pin_memory and loader.persistent_workers
