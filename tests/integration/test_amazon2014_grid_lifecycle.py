"""Synthetic Amazon GRID inputs through the current shared lifecycle; no network/GPU."""

import copy
import json
import os
import pickle
import shutil
import subprocess
from pathlib import Path

import polars as pl
import pytest
import torch
from test_amazon2014_grid_import import _source

from data_sources.event_hub import Reader, distribute_to_partitions
from genrecsys.core import Config
from genrecsys.utils.files import file_sha256
from model_datasets.amazon2014_grid.basis import build_basis
from model_datasets.amazon2014_grid.grid_sids import TOKEN_OFFSETS

ROOT = Path(__file__).resolve().parents[2]


def command(*args):
    env = {
        **os.environ,
        "PYTHONPATH": "src",
        "OMP_NUM_THREADS": "1",
        "POLARS_MAX_THREADS": "4",
        "ACCELERATE_USE_CPU": "true",
        "ACCELERATE_TORCH_DEVICE": "cpu",
    }
    result = subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-6000:]


def initialize_without_training(tmp_path, prepared, run, category, seed, config):
    # The production run.sh is unchanged. Stub only train/predict/evaluate;
    # retain fit-model's real Prepared.load compatibility check.
    worker = tmp_path / "init-worker"
    if not worker.exists():
        worker.mkdir()
        for name in (".venv", "src"):
            (worker / name).symlink_to(ROOT / name, target_is_directory=True)
        scripts = worker / "scripts"
        scripts.mkdir()
        train = scripts / "train.sh"
        train.write_text(
            "#!/bin/bash\nset -euo pipefail\n"
            '.venv/bin/python - "$@" <<PY_CHECK\n'
            "import sys\nfrom pathlib import Path\nfrom genrecsys.core import Config, Prepared\n"
            "Prepared.load(Path(sys.argv[1]) / 'checkpoint', config=Config.load_data(sys.argv[2]))\n"
            "PY_CHECK\n"
        )
        train.chmod(0o755)
        for name in ("predict.sh", "evaluate.sh"):
            stub = scripts / name
            stub.write_text("#!/bin/bash\nexit 0\n")
            stub.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/amazon2014-grid/run.sh"),
            str(prepared),
            str(run),
            category,
            str(seed),
            str(config),
        ],
        cwd=worker,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-6000:]


def frozen_inputs(tmp_path, category):
    data = tmp_path / "data"
    raw = _source(tmp_path / "raw")
    build_basis(
        source_dir=raw,
        output_dir=data / "basis",
        category=category,
        augmentation="none",
        max_history=20,
        pack_eventhub=False,
    )
    shutil.copytree(data / "basis/source", data / "prepared")
    catalog = data / "prepared/catalog/items.pq"
    sid_root = data / "grid-sids"
    tensors = sid_root / "inference/pickle"
    tensors.mkdir(parents=True)
    codes = torch.tensor([[i, i % 5, i % 3, 0] for i in range(27)])
    tensor = tensors / "merged_predictions_tensor.pt"
    rows = tensors / "merged_predictions.pkl"
    torch.save(codes.T, tensor)
    with rows.open("wb") as f:
        pickle.dump(
            [{"item_id": i, "cluster_ids": codes[i, :3].tolist()} for i in range(27)], f
        )
    validation = {
        "items": 27,
        "codebook_size": [256, 256, 256],
        "sid_length": 4,
        "fit_scope": "full_author_catalog",
        "category": category,
        "catalog_sha256": file_sha256(catalog),
        "source_report_sha256": file_sha256(data / "prepared/report.json"),
        "sha256": {str(p): file_sha256(p) for p in (tensor, rows)},
    }
    (sid_root / "runtime").mkdir()
    (sid_root / "runtime/sid-validation.json").write_text(json.dumps(validation))
    return data, codes


@pytest.mark.parametrize("category", ["beauty", "toys", "sports"])
def test_frozen_import_history_coverage_and_shared_preparation(tmp_path, category):
    data, codes = frozen_inputs(tmp_path, category)
    command(
        str(ROOT / ".venv/bin/python"),
        "-m",
        "model_datasets.amazon2014_grid",
        "import-grid-sids",
        f"--sid-dir={data / 'grid-sids'}",
        f"--catalog={data / 'prepared/catalog/items.pq'}",
        f"--source-report={data / 'prepared/report.json'}",
        f"--output-dir={data / 'sids'}",
    )
    mapping = pl.read_parquet(data / "sids/item_to_sid.pq")
    actual = {int(r["item_id"]): r["sid"] for r in mapping.iter_rows(named=True)}
    assert actual == {
        i: (codes[i] + torch.tensor(TOKEN_OFFSETS)).tolist() for i in range(27)
    }
    config = Config.load(f"configs/amazon2014-grid/{category}.yaml")
    reader = Reader(config, source_to_path={"amazon2014grid": data / "basis/eventhub"})
    for split, target in [("train", 24), ("validation", 25), ("test", 26)]:
        samples = pl.read_parquet(data / f"basis/basis/{split}/samples.pq")
        assert len(samples) == 5
        assert samples["target"].to_list() == [[str(target)]] * 5
        # Read each request through the real shared EventHub reader.
        for sample in samples.iter_rows(named=True):
            frame = distribute_to_partitions(
                pl.DataFrame(
                    {
                        "client_id": [sample["client_id"]],
                        "_sample_timestamp": [sample["timestamp"]],
                    }
                )
            )
            partition = frame["partition_id"][0]
            events = reader.read(partition, frame)
            assert events.sort("timestamp")["item"].to_list() == [
                str(i) for i in range(target - 20, target)
            ]
    recipe = Config.load("tests/fixtures/amazon2014_grid/tiny_model.yaml")
    # Keep category data and history settings; apply only the tiny smoke model/training.
    config.model = recipe.model
    config.training = recipe.training
    config.inference = recipe.inference
    path = tmp_path / "smoke.yaml"
    config.save(path)
    workdir = tmp_path / "prepared"
    command(
        "bash",
        "scripts/amazon2014-grid/prepare.sh",
        str(data),
        str(workdir),
        str(path),
    )
    assert Config.load(path).training.dataloader.seed == config.training.dataloader.seed
    assert (
        Config.load(workdir / "config.yaml").training.dataloader.seed
        == config.training.dataloader.seed
    )
    assert (workdir / "checkpoint/data_contract.json").is_file()
    assert (workdir / "dataset").is_dir()
    from genrecsys.core import Prepared
    from genrecsys.core.models.onerec.config import _model_config

    compiled = _model_config(Prepared.load(workdir / "checkpoint"))
    assert compiled["decoder_relative_attention"] is False
    assert compiled["inputs"][0]["event_encoder_type"] == "identity"
    assert compiled["input_type_embeddings"] is False
    assert compiled["inputs"][1]["token_layout"] == "sid_sequence"
    assert compiled["inputs"][1]["event_encoder_type"] == "identity"
    for split in ("train", "validation", "test"):
        materialized = pl.read_parquet(workdir / f"basis/{split}/samples.pq")
        assert materialized["context"].struct.field("user_bin").to_list() == list(
            range(5)
        )
        assert len(pl.read_parquet(workdir / f"basis/{split}/artifacts/items.pq")) == 27
    # Artifact-based IDs preserve every old category index, including unseen bins.
    from genrecsys.core.encoders.id import Observer, Preprocessor

    current = Prepared.load(workdir / "checkpoint")
    old_user = Preprocessor.fit(Observer(), categories=list(range(2000)))
    values = pl.Series("user_bin", [*range(2000), 2000, None])
    new_user = current.encoder_to_preprocessor["user_bin"]
    assert new_user.static_transform(values).equals(old_user.static_transform(values))
    assert new_user.num_embeddings == old_user.num_embeddings == 2001
    old_style = copy.deepcopy(current)
    old_style.config.encoders["user_bin"].preprocessor = {
        "categories": list(range(2000))
    }
    old_style.encoder_to_preprocessor["user_bin"] = old_user
    assert _model_config(old_style) == _model_config(current)

    # Separate model seeds reuse one dataset; training cannot mutate its checkpoint.
    before = {
        str(p.relative_to(workdir)): file_sha256(p)
        for p in workdir.rglob("*")
        if p.is_file()
    }
    runs = []
    for seed in (42, 43):
        run = tmp_path / f"seed{seed}"
        if category == "beauty" and seed == 43:
            command(
                "bash",
                "scripts/amazon2014-grid/run.sh",
                str(workdir),
                str(run),
                category,
                str(seed),
                str(path),
            )
        else:
            initialize_without_training(tmp_path, workdir, run, category, seed, path)
        assert (run / "dataset").resolve() == workdir / "dataset"
        assert Config.load(run / "config.yaml").training.dataloader.seed == seed
        assert not (run / "checkpoint/config.yaml").samefile(
            workdir / "checkpoint/config.yaml"
        )
        assert (
            Prepared.load(run / "checkpoint").dataset_identity
            == current.dataset_identity
        )
        assert Prepared.load(run / "checkpoint").data_contract == current.data_contract
        runs.append(run)
        if category == "beauty":
            if seed == 42:
                command(
                    "bash", "scripts/train.sh", str(run), str(run / "config.yaml"), "1"
                )
                command("bash", "scripts/predict.sh", str(run), "1")
                command("bash", "scripts/evaluate.sh", str(run))
            assert (run / "checkpoint/model/weights.pt").is_file()
            assert (run / "metrics.json").is_file()
            assert list((run / "predictions").glob("*.pq"))
    after = {
        str(p.relative_to(workdir)): file_sha256(p)
        for p in workdir.rglob("*")
        if p.is_file()
    }
    assert before == after
    with pytest.raises(AssertionError, match="exists"):
        initialize_without_training(tmp_path, workdir, runs[0], category, 44, path)
    # A different model recipe can reuse the preparation, incompatible data cannot.
    changed = copy.deepcopy(config)
    changed.model.params["dropout"] = 0.1
    changed.encoders["item"].embedder["prefix_length"] = 3
    changed.save(tmp_path / "compatible.yaml")
    initialize_without_training(
        tmp_path,
        workdir,
        tmp_path / "other-model-recipe",
        category,
        44,
        tmp_path / "compatible.yaml",
    )
    changed.inputs["history"].max_events_per_sequence = 19
    changed.save(tmp_path / "incompatible.yaml")
    with pytest.raises(AssertionError, match="data contract"):
        initialize_without_training(
            tmp_path,
            workdir,
            tmp_path / "rejected",
            category,
            44,
            tmp_path / "incompatible.yaml",
        )
    assert not (tmp_path / "rejected/checkpoint/model").exists()
    # Reusing a workdir must fail, never silently overwrite a completed run.
    with pytest.raises(AssertionError):
        command(
            "bash",
            "scripts/amazon2014-grid/prepare.sh",
            str(data),
            str(workdir),
            str(path),
        )


def test_materialize_preserves_input_order_and_model_sid_depth(tmp_path):
    from model_datasets.amazon2014_grid.grid_sids import import_grid_sids
    from model_datasets.amazon2014_grid.materialize import materialize_basis

    data, _ = frozen_inputs(tmp_path, "beauty")
    import_grid_sids(
        sid_dir=data / "grid-sids",
        catalog=data / "basis/catalog/inference/items.pq",
        output_dir=data / "sids",
    )
    recipe = Config.load("configs/amazon2014-grid/beauty.yaml")
    recipe.inputs = dict(reversed(list(recipe.inputs.items())))
    recipe.encoders["item"].embedder["prefix_length"] = 3
    config = tmp_path / "reordered.yaml"
    recipe.save(config)
    mapping = data / "sids/item_to_sid.pq"
    output = tmp_path / "materialized"
    materialize_basis(
        config=config,
        basis=data / "basis",
        sid_mapping=mapping,
        workdir=output,
        user_bins=2000,
    )
    result = Config.load(output / "config.yaml")
    assert list(result.inputs) == ["history", "user"]
    assert result.encoders["item"].embedder["prefix_length"] == 3
    assert result.model.params["decoder"]["generation_prefix_length"] == 4
    assert file_sha256(output / "resources/item_to_sid.pq") == file_sha256(mapping)
    for split in ("train", "validation", "test"):
        actual = pl.read_parquet(output / f"basis/{split}/samples.pq").drop("context")
        expected = pl.read_parquet(data / f"basis/basis/{split}/samples.pq")
        assert actual.equals(expected)
    truncated = tmp_path / "truncated.pq"
    pl.read_parquet(mapping).with_columns(pl.col("sid").list.head(3)).write_parquet(
        truncated
    )
    with pytest.raises(ValueError, match="four integer SID"):
        materialize_basis(
            config=config,
            basis=data / "basis",
            sid_mapping=truncated,
            workdir=tmp_path / "bad-mapping",
            user_bins=2000,
        )
