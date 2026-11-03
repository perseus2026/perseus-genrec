"""Native ES2 recipes use standard CLI fits and shared dataset preparation."""

import json
import re
import shutil
import subprocess
import sys

import numpy as np
import polars as pl
import pytest
from test_amazon2014_grid_lifecycle import ROOT, command, frozen_inputs

from genrecsys.core import Config, Prepared
from genrecsys.utils.files import file_sha256
from item_representations.semantic_ids.cli import app


@pytest.mark.parametrize("category", ["beauty", "toys", "sports"])
@pytest.mark.parametrize("method", ["rq_vae", "rvq", "rq_kmeans_full"])
def test_final_shell_recipes_use_standard_cli(tmp_path, category, method):
    # Capture commands without fitting a full production recipe.
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    log = tmp_path / "calls.jsonl"
    python.write_text(
        f"#!{sys.executable}\nimport json, sys\n"
        f"with open({str(log)!r}, 'a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    python.chmod(0o755)
    output = tmp_path / "fit"
    subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/amazon2014-grid/es2/sids.sh"),
            "representations",
            str(output),
            category,
            method,
            "44",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 2
    fit, apply = calls
    assert fit[:3] == ["-m", "item_representations.semantic_ids", "fit-sid-mapper"]
    assert apply[:3] == ["-m", "item_representations.semantic_ids", "apply-sid-mapper"]
    _, bound, _ = app.parse_args(fit[2:])
    bound.apply_defaults()
    options = bound.arguments
    assert options["method"] == method and options["random_state"] == 44
    assert options["codebook_size"] == (256, 256, 256)
    assert options["device"] == "cuda" and not options["mapper_only"]
    if method in ("rq_vae", "rvq"):
        assert options["training_batch_size"] == (
            18357 if category == "sports" else 16384
        )
        assert options["learning_rate"] == 0.001
    if method == "rq_vae":
        assert (
            options["epochs"]
            == {"beauty": 20310, "toys": 20611, "sports": 13388}[category]
        )
        assert options["optimizer_name"] == "adagrad" and options["gradient_clip"] == 0
        assert (
            not options["latent_l2"]
            and options["joint_normalization"]
            and options["mlp_bias"]
        )
        assert options["hidden_dim_factors"] == (8 / 3, 8, 16, 32)
    elif method == "rvq":
        assert (
            options["epochs_per_layer"]
            == {"beauty": 1354, "toys": 1375, "sports": 893}[category]
        )
        assert options["normalize_residuals"] and options["loss_scale"] == 0.125
    else:
        assert options["max_iter"] == 100 and options["n_init"] == 3
        assert options["max_gpu_memory_fraction"] == 0.8
    _, applied, _ = app.parse_args(apply[2:])
    assert applied.arguments["checkpoint"] == output / "train/checkpoint.pkl"
    assert applied.arguments["eval_batch_size"] == options["eval_batch_size"]


@pytest.mark.parametrize("method", ["rq_vae", "rvq", "rq_kmeans_full"])
def test_fit_apply_prepare_and_no_overwrite(tmp_path, method):
    data, _ = frozen_inputs(tmp_path, "sports")
    representations = tmp_path / "representations"
    train = representations / "train"
    train.mkdir(parents=True)
    pl.DataFrame({"item_id": [str(i) for i in range(27)]}).write_parquet(
        train / "items.pq"
    )
    np.save(
        train / "embeddings.npy",
        np.random.default_rng(19).normal(size=(27, 8)).astype("float32"),
    )
    shutil.copytree(train, representations / "inference")
    metadata = {
        "category": "sports",
        "catalog_items": 27,
        "normalize_embeddings": False,
        "train_vectors": "exact_copy_of_full_inference_matrix",
        "source_report_sha256": file_sha256(data / "prepared/report.json"),
    }
    (representations / "metadata.json").write_text(json.dumps(metadata))

    # Run the real shell and CLI with a tiny copy of the recipe; no production
    # parameter override or test-only switch is introduced into public scripts.
    worker = tmp_path / "worker"
    worker.mkdir()
    for name in (".venv", "src"):
        (worker / name).symlink_to(ROOT / name, target_is_directory=True)
    scripts = worker / "scripts/amazon2014-grid/es2"
    scripts.mkdir(parents=True)
    shell = (ROOT / "scripts/amazon2014-grid/es2/sids.sh").read_text()
    for name in ("batch",):
        shell = re.sub(rf"\b{name}=\d+", f"{name}=27", shell)
    for name in ("epochs", "layer_epochs"):
        shell = re.sub(rf"\b{name}=\d+", f"{name}=2", shell)
    shell = shell.replace("--codebook-size=256", "--codebook-size=3")
    shell = shell.replace("--init-buffer-size=3072", "--init-buffer-size=8")
    # Keep hidden widths non-zero for eight-dimensional synthetic vectors.
    shell = re.sub(r"--hidden-dim-factors=[0-9.]+", "--hidden-dim-factors=2", shell)
    (scripts / "sids.sh").write_text(shell)
    sids = tmp_path / "sids"
    args = [
        "bash",
        str(scripts / "sids.sh"),
        str(representations),
        str(sids),
        "sports",
        method,
        "43",
        "cpu",
    ]
    result = subprocess.run(
        args, cwd=worker, capture_output=True, text=True, timeout=180, check=False
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-6000:]
    record = json.loads((sids / "train/sid_mapper.json").read_text())
    assert record["random_state"] == 43 and record["method"] == method
    fitted = pl.read_parquet(sids / "train/item_to_sid.parquet").sort("item_id")
    applied = pl.read_parquet(sids / "inference/item_to_sid.parquet").sort("item_id")
    assert fitted.equals(applied)
    assert not (sids / "recipe.json").exists()
    prepared_dir = tmp_path / "prepared"
    command(
        "bash",
        "scripts/amazon2014-grid/prepare.sh",
        str(data),
        str(prepared_dir),
        "configs/amazon2014-grid/sports.yaml",
        str(sids / "inference/item_to_sid.parquet"),
        "",
    )
    prepared = Prepared.load(
        prepared_dir / "checkpoint",
        config=Config.load_data("configs/amazon2014-grid/sports.yaml"),
    )
    assert prepared.dataset_identity
    assert (prepared_dir / "dataset/events").is_dir()
    assert not (prepared_dir / "checkpoint/model").exists()
    for split in ("train", "validation", "test"):
        samples = pl.read_parquet(prepared_dir / f"basis/{split}/samples.pq")
        assert len(samples) == 5 and samples["history_length"].to_list() == [20] * 5
    before = (sids / "train/sid_mapper.json").read_bytes()
    rerun = subprocess.run(
        args, cwd=worker, capture_output=True, text=True, timeout=60, check=False
    )
    assert (
        rerun.returncode != 0
        and (sids / "train/sid_mapper.json").read_bytes() == before
    )
