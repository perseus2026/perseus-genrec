"""Author TIGER configuration and shell orchestration, offline and without CUDA."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "category,steps", [("beauty", 34942), ("toys", 30331), ("sports", 55622)]
)
def test_author_tiger_noaug_training_and_evaluation_recipe(
    tmp_path, mock_grid_download, category, steps
):
    hydra = pytest.importorskip("hydra")
    from omegaconf import OmegaConf

    spec = importlib.util.spec_from_file_location(
        "restore", ROOT / "scripts/amazon2014-grid/restore-grid.py"
    )
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    mock_grid_download(api, upstream=True)
    grid = api.restore(tmp_path / "grid")
    with hydra.initialize_config_dir(
        config_dir=str(grid / "configs"), version_base="1.3"
    ):
        config = hydra.compose(
            config_name="train",
            overrides=[
                "experiment=reproduce_tiger_train",
                f"category={category}",
                "hydra/job_logging=default",
                "hydra/hydra_logging=default",
                "seed=44",
            ],
        )
    assert config.train and config.test
    assert config.seed == 44
    assert config.trainer.devices == 1
    assert config.trainer.accumulate_grad_batches == 1
    assert config.trainer.max_steps == steps
    assert config.trainer.val_check_interval == 100
    assert config.trainer.check_val_every_n_epoch is None
    assert config.optim.optimizer.lr == 0.0005
    assert config.optim.optimizer.weight_decay == 0.000001
    assert config.num_hierarchies == 4
    assert config.model.num_user_bins == 2000
    assert config.model.mlp_layers == 1
    assert config.model.should_add_sep_token
    assert config.model.should_check_prefix
    assert config.model.top_k_for_generation == 10
    assert config.model.huggingface_model.config.dropout_rate == 0.15
    assert config.model.decoder.config.dropout_rate == 0.15
    assert config.callbacks.early_stopping.monitor == "val/ndcg@10"
    assert config.callbacks.early_stopping.patience == 10
    assert config.callbacks.model_checkpoint.monitor == "val/recall@10"
    loader = config.data_loading.train_dataloader_config.dataloader
    assert loader.batch_size_per_device == 256
    # A last-target collator with no augmentation-only arguments.
    assert OmegaConf.to_container(loader.collate_fn, resolve=True) == {
        "_target_": "src.data.loading.components.collate_functions.collate_fn_train",
        "_partial_": True,
        "sequence_length": 84,
        "padding_token": -1,
    }


@pytest.mark.parametrize(
    "mode", ["success", "failure", "missing-metrics", "missing-checkpoint"]
)
def test_author_tiger_shell_exports_metrics_and_propagates_failures(tmp_path, mode):
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    grid = tmp_path / "grid"
    config = grid / "configs/experiment/reproduce_tiger_train.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("# model execution is stubbed\n")
    sids = tmp_path / "sid.pt"
    sids.write_bytes(b"unchanged SID artifact")
    executor = tmp_path / "grid-python"
    executor.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\nfrom pathlib import Path\n"
        "Path('args.json').write_text(json.dumps(sys.argv[1:]))\n"
        "assert os.environ['PYTHONPATH'] == str(Path.cwd())\n"
        "mode = os.environ['TEST_MODE']\n"
        "if mode == 'failure': sys.exit(7)\n"
        "root = Path(next(x.split('=', 1)[1] for x in sys.argv if x.startswith('hydra.run.dir=')))\n"
        "(root / 'checkpoints').mkdir(parents=True)\n"
        "if mode != 'missing-checkpoint': (root / 'checkpoints/best_000100.ckpt').write_bytes(b'weights')\n"
        "if mode != 'missing-metrics':\n"
        "    log = root / 'csv/version_0/metrics.csv'\n"
        "    log.parent.mkdir(parents=True)\n"
        "    log.write_text('step,val/recall@10,test/recall@5,test/recall@10,test/ndcg@5,test/ndcg@10\\n100,0.9,,,,\\n100,,0.02,0.04,0.01,0.03\\n')\n"
    )
    executor.chmod(0o755)
    work = tmp_path / "runs/seed44"
    command = [
        "bash",
        str(ROOT / "scripts/amazon2014-grid/es1/run-tiger.sh"),
        str(tmp_path / "raw/beauty"),
        str(sids),
        str(work),
        "beauty",
        "44",
        str(grid),
        str(executor),
    ]
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env={**os.environ, "TEST_MODE": mode},
        capture_output=True,
        text=True,
        check=False,
    )
    assert sids.read_bytes() == b"unchanged SID artifact"
    assert json.loads((grid / "args.json").read_text()) == [
        "-m",
        "src.train",
        "experiment=reproduce_tiger_train",
        "category=beauty",
        f"data_dir={tmp_path}/raw/beauty",
        f"semantic_id_path={sids}",
        "seed=44",
        f"hydra.run.dir={work}/train",
    ]
    if mode != "success":
        assert result.returncode == 7 if mode == "failure" else result.returncode != 0
        assert not (work / "metrics.json").exists()
        return
    assert result.returncode == 0, result.stderr
    assert json.loads((work / "metrics.json").read_text()) == {
        "metrics": {
            "Recall@5": 0.02,
            "Recall@10": 0.04,
            "NDCG@5": 0.01,
            "NDCG@10": 0.03,
        },
        "best_checkpoint": "train/checkpoints/best_000100.ckpt",
    }
    before = (work / "metrics.json").read_bytes()
    retry = subprocess.run(
        command,
        cwd=tmp_path,
        env={**os.environ, "TEST_MODE": "failure"},
        capture_output=True,
        check=False,
    )
    assert retry.returncode != 0
    assert (work / "metrics.json").read_bytes() == before
