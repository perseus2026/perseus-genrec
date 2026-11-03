"""Source-only ES1/ES2 entrypoints share inputs across OneRec and native GRID."""

import json
import os
import shutil
import subprocess
import sys
from collections import Counter

import pytest
from test_amazon2014_grid_lifecycle import ROOT, command, frozen_inputs

from genrecsys.core import Config, DataContract, Prepared
from genrecsys.utils.files import file_sha256


@pytest.mark.parametrize("category", ["beauty", "toys", "sports"])
def test_public_models_share_data_contract_without_unifying_recipes(category):
    one = Config.load_data(f"configs/amazon2014-grid/{category}.yaml")
    grid = Config.load_data(f"configs/amazon2014-grid/grid-{category}.yaml")
    one.validate()
    grid.validate()
    assert DataContract.from_config(one) == DataContract.from_config(grid)
    assert one.training.early_stopping.metric == "Recall@10"
    assert grid.training.early_stopping.metric == "NDCG@10"
    assert grid.model.params["training_recipe"]["validation_every_n_steps"] == 100
    assert grid.inference.predict_kwargs["renormalize_logits"] is True


def tiny_config(model, path):
    if model == "onerec":
        config = Config.load("tests/fixtures/amazon2014_grid/tiny_model.yaml")
    else:
        config = Config.load("configs/amazon2014-grid/grid-beauty.yaml")
        config.model.params.update(
            dim=16, num_heads=2, head_dim=8, dim_feedforward=32, dropout=0.0
        )
        config.model.params["encoder"]["num_layers"] = 1
        config.model.params["decoder"]["num_layers"] = 1
        config.model.params["training_recipe"]["validation_every_n_steps"] = 1
        config.training.num_epochs = 1
        config.training.dataloader.batch_size = 8
        config.training.validation_dataloader.batch_size = 8
        config.inference.dataloader.batch_size = 8
    config.save(path)


def test_es2_both_models_reuse_transported_prepared_inputs(tmp_path):
    # Synthetic author SID output; the existing quantizer tests cover fitting.
    data, _ = frozen_inputs(tmp_path, "beauty")
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
    prepared = tmp_path / "prepared"
    fits = tmp_path / "fit"
    shutil.copytree(data / "sids", fits / "imported")
    command(
        "bash",
        "scripts/amazon2014-grid/prepare.sh",
        str(data),
        str(prepared),
        "configs/amazon2014-grid/beauty.yaml",
        str(fits / "imported/item_to_sid.pq"),
        str(fits / "imported/sid_import_report.json"),
    )

    # Mimic cross-job transport: archive the whole prepared, then remove its
    # original location. Both models must work with the relocated EventStore.
    archive = shutil.make_archive(
        str(tmp_path / "artifact"), "gztar", prepared.parent, prepared.name
    )
    destination = tmp_path / "new-worker"
    shutil.unpack_archive(archive, destination)
    shutil.rmtree(prepared)
    prepared = destination / "prepared"

    # Arrow caches use source mtime_ns and may be rebuilt after tar transport;
    # only immutable protocol inputs/checkpoint/provenance belong in this comparison.
    before = {
        str(p.relative_to(prepared)): file_sha256(p)
        for p in prepared.rglob("*")
        if p.is_file() and ".samples-cache" not in p.parts
    }
    shared = Prepared.load(prepared / "checkpoint")
    for model in ("onerec", "grid"):
        config = tmp_path / f"{model}.yaml"
        tiny_config(model, config)
        run = tmp_path / model
        command(
            "bash",
            "scripts/amazon2014-grid/run.sh",
            str(prepared),
            str(run),
            "beauty",
            "42",
            str(config),
        )
        report = json.loads((run / "metrics.json").read_text())
        assert report["population_verified"] is True and report["samples"] == 5
        assert (run / "dataset").resolve() == prepared / "dataset"
        assert Config.load(run / "config.yaml").training.dataloader.seed == 42
        assert (
            Prepared.load(run / "checkpoint").dataset_identity
            == shared.dataset_identity
        )
        assert (run / "checkpoint/model/weights.pt").is_file()
    after = {
        str(p.relative_to(prepared)): file_sha256(p)
        for p in prepared.rglob("*")
        if p.is_file() and ".samples-cache" not in p.parts
    }
    assert before == after


def full_run_sandbox(tmp_path):
    # Real shell orchestration; replace only expensive stage entrypoints.
    scripts = tmp_path / "scripts/amazon2014-grid"
    shutil.copytree(ROOT / "scripts/amazon2014-grid", scripts)
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    recorder = tmp_path / "record.py"
    recorder.write_text(
        "import json, os, sys\n"
        "with open(os.environ['CALLS'], 'a') as f: "
        "f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1] == 'download-embedder.sh': print(sys.argv[2] + '/resolved')\n"
        "sys.exit(7 if os.environ.get('FAIL_STAGE') == sys.argv[1] else 0)\n"
    )
    for name in (
        "download-data.sh",
        "download-embedder.sh",
        "prepare-grid.sh",
        "prepare-data.sh",
        "grid-sids.sh",
        "es2/sids.sh",
        "prepare.sh",
        "run.sh",
        "es1/run-tiger.sh",
    ):
        (scripts / name).write_text(
            f"#!/usr/bin/env bash\nset -euo pipefail\n"
            f'"{sys.executable}" "{recorder}" "{name}" "$@"\n'
        )
    return scripts


def invoke_full_run(tmp_path, scripts, es, category, fail=""):
    log = tmp_path / "calls.jsonl"
    work = tmp_path / es / category
    result = subprocess.run(
        [
            "bash",
            str(scripts / es / "run_all.sh"),
            str(tmp_path / "raw" / category),
            str(work),
            category,
            str(tmp_path / "grid"),
            str(tmp_path / "grid-python"),
            str(tmp_path / "flan"),
        ],
        cwd=tmp_path,
        env={**os.environ, "CALLS": str(log), "FAIL_STAGE": fail},
        capture_output=True,
        text=True,
        check=False,
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    return result, calls, work


@pytest.mark.parametrize("es", ["es1", "es2"])
@pytest.mark.parametrize("category", ["beauty", "toys", "sports"])
def test_full_run_reuses_inputs_and_preserves_seed_protocol(tmp_path, es, category):
    scripts = full_run_sandbox(tmp_path)
    result, all_calls, work = invoke_full_run(tmp_path, scripts, es, category)
    assert result.returncode == 0, result.stderr
    assert all_calls[:3] == [
        ["download-data.sh", str(tmp_path / "raw")],
        ["download-embedder.sh", str(tmp_path / "flan")],
        ["prepare-grid.sh", str(tmp_path / "grid"), str(tmp_path / "grid-python")],
    ]
    calls = all_calls[3:]
    assert calls[0] == [
        "prepare-data.sh",
        str(tmp_path / "raw" / category),
        str(work / "data"),
        category,
        str(tmp_path / "grid"),
        str(tmp_path / "grid-python"),
        str(tmp_path / "flan/resolved"),
    ]
    counts = Counter(call[0] for call in calls)
    assert counts["prepare-data.sh"] == 1
    runs = [c[1:] for c in calls if c[0] == "run.sh"]
    assert len({r[1] for r in runs}) == len(runs)
    assert {r[2] for r in runs} == {category}
    expected_configs = {
        f"configs/amazon2014-grid/{category}.yaml",
        f"configs/amazon2014-grid/grid-{category}.yaml",
    }
    assert {r[4] for r in runs} == expected_configs
    if es == "es1":
        assert calls[1][1] == "es1" and calls[1][-2:] == ["rkmeans", "42"]
        assert counts == {
            "prepare-data.sh": 1,
            "grid-sids.sh": 1,
            "prepare.sh": 1,
            "run.sh": 10,
            "es1/run-tiger.sh": 5,
        }
        assert calls[1][2:5] == [
            str(tmp_path / "raw" / category),
            str(work / "data"),
            str(work / "data/grid-sids"),
        ]
        assert calls[2] == [
            "prepare.sh",
            str(work / "data"),
            str(work / "prepared"),
            f"configs/amazon2014-grid/{category}.yaml",
        ]
        assert {r[0] for r in runs} == {str(work / "prepared")}
        for config in expected_configs:
            assert [r[3] for r in runs if r[4] == config] == list(
                map(str, range(42, 47))
            )
        author_runs = [c for c in calls if c[0] == "es1/run-tiger.sh"]
        assert author_runs == [
            [
                "es1/run-tiger.sh",
                str(tmp_path / "raw" / category),
                str(work / "data/grid-sids/inference/pickle/merged_predictions_tensor.pt"),
                str(work / f"models/tiger-author/seed{seed}"),
                category,
                str(seed),
                str(tmp_path / "grid"),
                str(tmp_path / "grid-python"),
            ]
            for seed in range(42, 47)
        ]
        assert calls[-5:] == author_runs
    else:
        assert counts == {
            "prepare-data.sh": 1,
            "grid-sids.sh": 15,
            "es2/sids.sh": 15,
            "prepare.sh": 30,
            "run.sh": 60,
        }
        fits = [c for c in calls if c[0] in ("grid-sids.sh", "es2/sids.sh")]
        assert len({c[4] if c[0] == "grid-sids.sh" else c[2] for c in fits}) == 30
        slots = set()
        for c in fits:
            if c[0] == "grid-sids.sh":
                assert c[1] == "es2"
                assert c[2:4] == [str(tmp_path / "raw" / category), str(work / "data")]
                builder = "grid-" + c[-2]
            else:
                assert c[1] == str(work / "data/basis/representations")
                builder = "ours-" + c[-2]
            slots.add((builder, c[-1]))
        builders = {
            "grid-rkmeans",
            "grid-rvq",
            "grid-rqvae",
            "ours-rq_kmeans_full",
            "ours-rq_vae",
            "ours-rvq",
        }
        assert slots == {(b, str(seed)) for b in builders for seed in range(42, 47)}
        assert {r[3] for r in runs} == {"42"}
        for i, c in enumerate(calls):
            if c[0] != "prepare.sh":
                continue
            assert c[1] == str(work / "data")
            previous = calls[i - 1]
            fit = previous[4] if previous[0] == "grid-sids.sh" else previous[2]
            assert c[4] == fit + (
                "/imported/item_to_sid.pq"
                if previous[0] == "grid-sids.sh"
                else "/inference/item_to_sid.parquet"
            )
            assert c[5] == (
                fit + "/imported/sid_import_report.json"
                if previous[0] == "grid-sids.sh"
                else ""
            )
            assert [r[0] for r in runs if r[0] == c[2]] == [c[2], c[2]]
            assert {calls[i + j][-1] for j in (1, 2)} == expected_configs
            assert all(calls[i + j][1] == c[2] for j in (1, 2))
    # A second full invocation must not overwrite/recompute existing outputs.
    retry, after, _ = invoke_full_run(tmp_path, scripts, es, category)
    assert retry.returncode != 0
    assert after == all_calls


@pytest.mark.parametrize("es", ["es1", "es2"])
@pytest.mark.parametrize(
    "failure",
    [
        "download-data.sh",
        "download-embedder.sh",
        "prepare-grid.sh",
        "prepare-data.sh",
        "run.sh",
    ],
)
def test_full_run_stops_on_stage_failure(tmp_path, es, failure):
    scripts = full_run_sandbox(tmp_path)
    result, calls, _ = invoke_full_run(tmp_path, scripts, es, "beauty", failure)
    assert result.returncode == 7
    assert calls[-1][0] == failure
    assert sum(c[0] == failure for c in calls) == 1


@pytest.mark.parametrize("es", ["es1", "es2"])
def test_category_quickstart_matches_explicit_paths(tmp_path, es):
    scripts = full_run_sandbox(tmp_path)
    log = tmp_path / "calls.jsonl"
    prefix = ["bash", str(scripts / es / "run_all.sh")]
    env = {**os.environ, "CALLS": str(log), "FAIL_STAGE": ""}
    subprocess.run(prefix + ["beauty"], cwd=tmp_path, env=env, check=True)
    quickstart = log.read_bytes()
    work = tmp_path / f"work/{es}-complete/beauty"
    # Stage stubs leave only an empty output directory.
    work.rmdir()
    log.unlink()
    subprocess.run(
        prefix
        + [
            str(tmp_path / "work/amazon_data/beauty"),
            str(work),
            "beauty",
            str(tmp_path / "work/grid/GRID-main"),
            str(tmp_path / "work/grid/venv/bin/python"),
            str(tmp_path / f"work/{es}/flan-t5-xl"),
        ],
        cwd=tmp_path,
        env=env,
        check=True,
    )
    assert log.read_bytes() == quickstart
    # Repeating the short command preserves existing outputs without executing stages.
    retry = subprocess.run(prefix + ["beauty"], cwd=tmp_path, env=env, check=False)
    assert retry.returncode != 0
    assert log.read_bytes() == quickstart


def test_full_es1_stops_on_author_tiger_failure(tmp_path):
    scripts = full_run_sandbox(tmp_path)
    result, calls, _ = invoke_full_run(
        tmp_path, scripts, "es1", "beauty", "es1/run-tiger.sh"
    )
    assert result.returncode == 7
    assert calls[-1][0] == "es1/run-tiger.sh"
    assert sum(c[0] == "es1/run-tiger.sh" for c in calls) == 1
