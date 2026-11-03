"""Offline checks of GRID source and historical configuration equivalence."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "entry,original,new,overrides",
    [
        (
            "inference",
            "sem_embeds_inference_flat",
            "reproduce_embeddings",
            ["trainer.devices=1", "+trainer.strategy=ddp"],
        ),
        (
            "train",
            "rkmeans_train_flat",
            "reproduce_rkmeans_train",
            [
                "embedding_dim=2048",
                "num_hierarchies=3",
                "codebook_width=256",
                "seed=42",
                "model.normalize_residuals=true",
                "trainer.devices=8",
                "trainer.max_steps=3000",
                "trainer.max_epochs=-1",
                "data_loading.datamodule.train_dataloader_config.batch_size_per_device=2048",
                "callbacks.early_stopping=null",
                "callbacks.model_checkpoint.save_last=true",
                "+should_skip_retry=true",
            ],
        ),
        (
            "inference",
            "rkmeans_inference_flat",
            "reproduce_rkmeans_inference",
            [
                "embedding_dim=2048",
                "num_hierarchies=3",
                "codebook_width=256",
                "seed=42",
                "model.normalize_residuals=true",
                "callbacks.bq_writer=null",
                "trainer.devices=1",
                "+trainer.strategy=ddp",
            ],
        ),
    ],
)
def test_grid_configs_match_historical_overrides(
    tmp_path, mock_grid_download, entry, original, new, overrides
):
    hydra = pytest.importorskip(
        "hydra"
    )  # GRID dependency, optional for OneRec-only installs
    from omegaconf import OmegaConf

    spec = importlib.util.spec_from_file_location(
        "restore_grid", ROOT / "scripts/amazon2014-grid/restore-grid.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mock_grid_download(module, upstream=True)
    config_dir = module.restore(tmp_path / "GRID-main") / "configs"
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        # Only local composition: colorlog plugin is unnecessary, no GRID imports/GPU.
        logs = ["hydra/job_logging=default", "hydra/hydra_logging=default"]
        old = hydra.compose(
            config_name=entry, overrides=[f"experiment={original}", *overrides, *logs]
        )
        new_config = hydra.compose(
            config_name=entry, overrides=[f"experiment={new}", *logs]
        )
        assert OmegaConf.to_container(old, resolve=False) == OmegaConf.to_container(
            new_config, resolve=False
        )
