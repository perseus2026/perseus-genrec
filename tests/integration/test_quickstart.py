"""Exercise the README CLI stages using small, test-only synthetic inputs."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import polars as pl

from data_sources import event_hub
from genrecsys.core import Config
from tests.integration.quickstart_data import make_data


def test_quickstart(tmp_path):
    root = Path(__file__).resolve().parents[2]
    workdir = tmp_path / "quickstart"
    make_data(workdir)
    shutil.rmtree(
        workdir / "eventhub"
    )  # Require the public import command to create it.
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONPATH": "src",
        "OMP_NUM_THREADS": "1",
        "POLARS_MAX_THREADS": "4",
        "ACCELERATE_TORCH_DEVICE": "cpu",
        "CUDA_VISIBLE_DEVICES": "",
        "INTERNAL_STORAGE_EVENT_HUB": str(workdir / "eventhub"),
    }
    commands = [
        [
            "genrecsys",
            "event-hub",
            "add-events",
            str(workdir / "events.pq"),
            "--name=view",
        ],
        [
            "item_representations.semantic_ids",
            "fit-sid-mapper",
            "--device=cpu",
            "--codebook-size=4",
            "--codebook-size=4",
            "--max-iter=20",
            "--n-init=1",
        ],
        ["item_representations.semantic_ids", "apply-sid-mapper", "--device=cpu"],
        ["genrecsys", "train", "prepare-dataset"],
        ["genrecsys", "train", "fit-model"],
        [
            "genrecsys",
            "inference",
            "distribute-samples",
            "--history-source=event_store",
        ],
        ["genrecsys", "inference", "make-predictions", "--history-source=event_store"],
        ["genrecsys", "inference", "evaluate"],
    ]
    for command in commands:
        options = [] if command[1] == "event-hub" else [f"--workdir={workdir}"]
        result = subprocess.run(
            [str(root / ".venv/bin/python"), "-m", *command, *options],
            cwd=root,
            env=env,
            capture_output=True,
            check=False,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    mapping = pl.read_parquet(workdir / "resources/sid/item_to_sid.parquet")
    assert mapping["item_id"].to_list() == list(range(16))
    assert mapping["sid"].list.len().to_list() == [3] * 16
    assert len({tuple(sid) for sid in mapping["sid"].to_list()}) == 16
    assert (workdir / "sid-fit/checkpoint.pkl").is_file()
    fit = json.loads((workdir / "sid-fit/sid_fit_report.json").read_text())
    assert fit["method"] == "rq_kmeans_full"
    assert all(level["iterations"] > 0 for level in fit["fit"])

    reader = event_hub.Reader(
        Config.load(workdir / "config.yaml"),
        source_to_path={"event_hub": workdir / "eventhub"},
    )
    for split in ("train", "validation", "test"):
        samples = pl.read_parquet(workdir / "basis" / split / "samples.pq")
        assert samples.height == 32
        # The raw store also contains targets/future events. Check the shared
        # reader selects exactly five earlier events for every request.
        for sample in samples.iter_rows(named=True):
            query = pl.DataFrame(
                {
                    "client_id": [sample["client_id"]],
                    "_sample_timestamp": [sample["timestamp"]],
                }
            )
            events = reader.read(
                event_hub.client_partition_id(sample["client_id"]), query
            )
            assert events is not None and events.height == 5
            assert all(
                t.replace(tzinfo=None) < sample["timestamp"].replace(tzinfo=None)
                for t in events["timestamp"]
            )

    report = json.loads((workdir / "metrics.json").read_text())
    assert report["samples"] == 32 and report["population_verified"]
    assert set(report["metrics"]) == {"Recall@5", "NDCG@5"}
    assert all(0 <= value <= 1 for value in report["metrics"].values())
    predictions = pl.read_parquet(workdir / "predictions/*.pq")
    assert predictions.height == predictions["sample_id"].n_unique() == 32
    assert predictions["prediction"].list.len().to_list() == [5] * 32
    assert set(predictions["prediction"].explode().struct.field("item")) <= set(
        range(16)
    )
    training = json.loads((workdir / "checkpoint/model/training.json").read_text())
    assert len(training["history"]) == 3
    assert (workdir / "checkpoint/model/weights.pt").is_file()
