"""Paper recipes and their native shell execution plan."""

import hashlib
import importlib
import json
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest

from genrecsys.core import Config, DataContract
from scripts import experiments

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "dataset,count,prepares",
    [
        ("tecd-small", 10, 4),
        ("yambda-50m", 7, 2),
        ("tecd-full", 12, 12),
        ("yambda-500m", 7, 2),
    ],
)
def test_paper_recipes_match_shared_preparation_and_training_protocol(
    dataset, count, prepares, tmp_path
):
    stages = experiments.plan(dataset)
    matrix = experiments.models(dataset)
    assert len(matrix) == count
    counts = Counter(stage["action"] for stage in stages)
    assert counts["prepare"] + counts["assemble_full"] == prepares
    for model in matrix:
        config = Config.load(ROOT / "configs" / dataset / (model["name"] + ".yaml"))
        saved = experiments._config_path(dataset, model["name"], tmp_path)
        assert saved.parent == tmp_path / "configs"
        assert asdict(Config.load(saved)) == asdict(config)
        source = Config.load(
            ROOT / "configs" / dataset / ("onerec-" + model["prepared"] + ".yaml")
        )
        DataContract.from_config(source).require_compatible(config)
        assert config.training.num_epochs == {"tecd-full": 4, "yambda-500m": 20}.get(
            dataset, 30
        )
        assert config.training.validation_every_n_epochs == (
            3 if dataset == "tecd-small" else 1
        )
        assert (
            config.training.dataloader.batch_size
            * config.training.gradient_accumulation_steps
            * 8
            == 4096
        )
        assert (
            config.training.mixed_precision
            == config.inference.mixed_precision
            == "bf16"
        )
        assert not config.training.gradient_checkpointing
        if config.model.type == "grid":
            cap = 128 if dataset == "tecd-small" else 102
            width = 3 if dataset == "tecd-small" else 4
            assert config.model.params["max_history_items"] == cap
            assert cap * (width + 1) <= 512
            assert not config.event_encoders
        if config.model.type == "hstu":
            assert config.model.params["weight_initialization"] == "perseus"
            assert config.training.optimizer.params["lr"] == 2e-3
            assert config.task.objective.loss["num_negatives"] == 4096
        if dataset == "tecd-full":
            assert config.inference.predict_kwargs == {"k": 100, "beam_size": 100}
            test = Config.load(
                ROOT / "configs" / dataset / (model["test_config"] + ".yaml")
            )
            assert test.inference.predict_kwargs == {"k": 200, "beam_size": 200}
            assert set(test.task.metrics) == {
                "Recall@100",
                "NDCG@100",
                "Recall@200",
                "NDCG@200",
            }
            DataContract.from_config(config).require_compatible(test)
    if dataset == "tecd-full":
        assert counts["read_full"] == counts["encode_full"] == 120
        assert counts["fit_sid"] == counts["apply_sid"] == 3
        assert counts["partition_history"] == 7
    assert [p.name for p in (ROOT / "scripts" / dataset).glob("*.sh")] == ["run_all.sh"]


@pytest.mark.parametrize("dataset", experiments.DATASETS)
def test_shell_executes_all_native_stages_in_plan_order(dataset, tmp_path, monkeypatch):
    """Capture the native CLI calls, including checkpoint cloning and final report."""
    calls = []
    repository = tmp_path / "repository"
    repository.mkdir()
    for name in ("src", "scripts", "configs"):
        (repository / name).symlink_to(ROOT / name, target_is_directory=True)
    monkeypatch.setattr(experiments, "ROOT", repository)
    data, runs = tmp_path / "data dir", tmp_path / "run dir"

    def run(module, arguments, env, *, ranks=1):
        args = [str(v) for v in arguments]
        if module == "genrecsys":
            parser_module = "genrecsys.cli." + args[0].replace("-", "_")
            tokens = args[1:]
        else:
            parser_module = (
                module if module.endswith((".full", ".payments")) else module + ".cli"
            )
            tokens = args
        importlib.import_module(parser_module).app.parse_args(
            tokens, exit_on_error=False, print_error=False
        )
        options = dict(
            v[2:].split("=", 1) for v in args if v.startswith("--") and "=" in v
        )
        if "config" in options:
            assert Path(options["config"]).parent == runs / "configs"
        calls.append((module, args, ranks))
        if args[0].startswith("build-small-"):
            (Path(options["output-dir"]) / "eventhub" / args[0]).mkdir(parents=True)
        if args[0] == "extract-eventhub":
            (Path(options["basis"]) / "eventhub/aux").mkdir(parents=True)
        if "materialize-basis" in args:
            checkpoint = Path(options["workdir"]) / "checkpoint"
            checkpoint.mkdir(parents=True)
            config = Config.load(options["config"])
            DataContract.from_config(config).save(checkpoint / "data_contract.json")
            mapping = checkpoint / "encoders/preprocessors/item/mapping.pq"
            mapping.parent.mkdir(parents=True)
            mapping.write_bytes(b"same frozen SID mapping")
            config.save(checkpoint / "config.yaml")
            for name in ("dataset", "samples"):
                (checkpoint.parent / name).mkdir()
        if args[:2] == ["prepare-shards", "encode"]:
            output = Path(options["output-dir"])
            output.mkdir(parents=True)
            (output / f"data-{output.name}.arrow").write_bytes(b"encoded events")
        if args[:2] == ["prepare-shards", "assemble"]:
            output = Path(options["output-dir"])
            checkpoint = output / "checkpoint"
            checkpoint.mkdir(parents=True)
            source = output.parent.name
            config = Config.load(ROOT / "configs/tecd-full" / f"onerec-{source}.yaml")
            DataContract.from_config(config).save(checkpoint / "data_contract.json")
            mapping = checkpoint / "encoders/preprocessors/item/mapping.pq"
            mapping.parent.mkdir(parents=True)
            mapping.write_bytes(b"same frozen SID mapping")
            config.save(checkpoint / "config.yaml")
            (output / "samples").mkdir()
        if "fit-model" in args:
            Config.load(options["config"]).save(
                Path(options["workdir"]) / "checkpoint/config.yaml"
            )
        if "evaluate" in args:
            (Path(options["workdir"]) / "metrics.json").write_text(
                json.dumps(
                    {
                        "population_verified": True,
                        "samples": 3,
                        "metrics": {"Recall@100": 1.0, "NDCG@100": 1.0},
                    }
                )
            )

    monkeypatch.setattr(experiments, "_run", run)
    experiments.run_all(dataset, data_dir=data, run_dir=runs)
    stages = experiments.plan(dataset)
    assert len([v for v in calls if "fit-model" in v[1]]) == len(
        experiments.models(dataset)
    )
    actual = [
        (v[1][1], v[2])
        for v in calls
        if v[0] == "genrecsys"
        and v[1][1] in {"fit-model", "make-predictions", "evaluate"}
    ]
    expected = [
        (
            {
                "train": "fit-model",
                "inference": "make-predictions",
                "evaluate": "evaluate",
            }[s["action"]],
            s["gpus"] or 1,
        )
        for s in stages
        if s["action"] in {"train", "inference", "evaluate"}
    ]
    assert actual == expected
    assert len([v for v in calls if "distribute-samples" in v[1]]) == len(
        [s for s in stages if s["action"] in {"prepare", "assemble_full"}]
    )
    fitted = [v for v in calls if "fit-sid-mapper" in v[1]]
    assert len(fitted) == {"tecd-small": 2, "tecd-full": 3}.get(dataset, 1)
    assert all("/representations/train/" in " ".join(v[1]) for v in fitted)
    if dataset == "tecd-small":
        assert all(
            "--train-ratio=0.75" in v[1]
            and "--history-gap-hours=0" in v[1]
            and "--no-include-views" in v[1]
            for v in calls
            if v[1][0].startswith("build-small-")
        )
    report = json.loads((runs / "summary/summary.json").read_text())
    manifest = json.loads((runs / "summary/manifest.json").read_text())
    expected_configs = {
        model[key] + ".yaml"
        for model in experiments.models(dataset)
        for key in ("name", "test_config")
        if key in model
    }
    assert set(manifest["effective_configs"]) == expected_configs
    for name, digest in manifest["effective_configs"].items():
        saved = runs / "configs" / name
        assert hashlib.sha256(saved.read_bytes()).hexdigest() == digest
        source = ROOT / "configs" / dataset / name
        assert asdict(Config.load(saved)) == asdict(Config.load(source))
    assert [r["model"] for r in report] == [
        m["name"] for m in experiments.models(dataset)
    ]
    assert len({r["sid_mapping"] for r in report}) == 1
    assert not (repository / "presets").exists()
    provenance = json.loads((runs / "summary/code-provenance.json").read_text())
    assert all(
        Path(path).parts[0] in {"src", "scripts", "configs"}
        for path in provenance["files"]
    )


@pytest.mark.parametrize("corruption", ["population", "mapping"])
def test_report_rejects_invalid_comparisons(corruption, tmp_path):
    for index, model in enumerate(experiments.models("yambda-50m")):
        directory = tmp_path / "models" / model["name"]
        (directory / "checkpoint").mkdir(parents=True)
        (directory / "metrics.json").write_text(
            json.dumps(
                {
                    "population_verified": corruption != "population",
                    "samples": 3,
                    "metrics": {"Recall@100": 1.0},
                }
            )
        )
        (directory / "checkpoint/experiment-input.json").write_text(
            json.dumps(
                {
                    "prepared": model["prepared"],
                    "data_contract": "contract",
                    "sid_mapping": str(index) if corruption == "mapping" else "same",
                }
            )
        )
    with pytest.raises(ValueError, match="Incomplete test|frozen SID"):
        experiments.run_stage(
            "yambda-50m", "report", data_dir=tmp_path / "data", run_dir=tmp_path
        )


@pytest.mark.parametrize("dataset", experiments.DATASETS)
@pytest.mark.parametrize("custom_paths", [False, True])
def test_main_shell_forwards_dataset_and_paths(dataset, custom_paths, tmp_path):
    executable = tmp_path / ".venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.write_text(
        f"#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n"
    )
    executable.chmod(0o755)
    paths = ["data dir", "run dir"] if custom_paths else []
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / dataset / "run_all.sh"), *paths],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == [
        "-m",
        "scripts.experiments",
        "run-all",
        dataset,
        "--data-dir=" + ("data dir" if custom_paths else f"work/{dataset}/data"),
        "--run-dir=" + ("run dir" if custom_paths else f"work/{dataset}/run"),
    ]
