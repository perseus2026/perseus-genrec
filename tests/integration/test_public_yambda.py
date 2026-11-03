"""Run the public CLI and shell lifecycle without network or GPU."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from requests.exceptions import ChunkedEncodingError

from model_datasets.yambda import basis

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("dataset_size", ["50m", "500m"])
def test_public_download_uses_pinned_full_source(tmp_path, monkeypatch, dataset_size):
    import huggingface_hub

    from model_datasets.yambda.basis import SOURCE_REVISION
    from model_datasets.yambda.cli import app

    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        path = kwargs["local_dir"] / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    with pytest.raises(SystemExit) as exit_info:
        app(
            [
                "download-source",
                f"--output-dir={tmp_path}",
                f"--dataset-size={dataset_size}",
            ]
        )
    assert exit_info.value.code == 0
    assert [call["filename"] for call in calls] == [
        f"flat/{dataset_size}/likes.parquet",
        "embeddings.parquet",
        f"flat/{dataset_size}/dislikes.parquet",
        f"flat/{dataset_size}/listens.parquet",
        "album_item_mapping.parquet",
        "artist_item_mapping.parquet",
    ]
    assert all(call["revision"] == SOURCE_REVISION for call in calls)
    assert all(call["repo_id"] == "yandex/yambda" for call in calls)


@pytest.mark.parametrize("dataset_size", ["50m", "500m"])
@pytest.mark.parametrize(
    ("failures", "error_type"),
    [
        (0, ChunkedEncodingError),
        (2, ChunkedEncodingError),
        (basis.DOWNLOAD_ATTEMPTS, ChunkedEncodingError),
        (1, ValueError),
    ],
)
def test_download_resumes_only_interrupted_files(
    tmp_path, monkeypatch, dataset_size, failures, error_type
):
    import huggingface_hub

    calls, sleeps = [], []
    partial = tmp_path / ".cache/huggingface/download/audio.incomplete"
    error = error_type("Connection broken: IncompleteRead")

    def download(**arguments):
        calls.append(arguments)
        path = arguments["local_dir"] / arguments["filename"]
        if arguments["filename"] == "embeddings.parquet":
            attempt = len(calls) - 1
            if attempt > 1:
                assert partial.read_bytes() == b"partial audio"
            if attempt <= failures:
                partial.parent.mkdir(parents=True, exist_ok=True)
                partial.write_bytes(b"partial audio")
                raise error
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"complete")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(basis.time, "sleep", sleeps.append)
    arguments = {
        "output_dir": tmp_path,
        "revision": "pinned",
        "dataset_size": dataset_size,
        "like_only": True,
    }
    should_fail = failures == basis.DOWNLOAD_ATTEMPTS or error_type is ValueError
    if should_fail:
        with pytest.raises(error_type) as caught:
            basis.download_source(**arguments)
        assert caught.value is error
        assert partial.read_bytes() == b"partial audio"
        assert not (tmp_path / "embeddings.parquet").exists()
    else:
        basis.download_source(**arguments)
        assert (tmp_path / "embeddings.parquet").read_bytes() == b"complete"
    attempts = failures if should_fail else failures + 1
    assert [call["filename"] for call in calls] == [
        f"flat/{dataset_size}/likes.parquet",
        *["embeddings.parquet"] * attempts,
    ]
    assert all(
        call
        == {
            "repo_id": basis.SOURCE_REPOSITORY,
            "repo_type": "dataset",
            "revision": "pinned",
            "filename": call["filename"],
            "local_dir": tmp_path,
        }
        for call in calls
    )
    assert sleeps == [basis.DOWNLOAD_RETRY_DELAY_SECONDS] * (attempts - 1)


def _write_raw(raw):
    events = raw / "flat/50m"
    events.mkdir(parents=True)
    pl.DataFrame(
        {
            "uid": [uid for tick in range(100) for uid in range(4)],
            "timestamp": [1000 + tick for tick in range(100) for uid in range(4)],
            "item_id": [tick % 2 for tick in range(100) for uid in range(4)],
            "is_organic": [1] * 400,
        }
    ).write_parquet(events / "likes.parquet")
    pl.DataFrame(
        {
            "uid": [uid for tick in range(100) for uid in range(4)],
            "timestamp": [1000 + tick for tick in range(100) for uid in range(4)],
            "item_id": [tick % 3 for tick in range(100) for uid in range(4)],
            "is_organic": [1] * 400,
            "played_ratio_pct": [60] * 400,
            "track_length_seconds": [
                100 + 60 * (tick % 3) for tick in range(100) for uid in range(4)
            ],
        }
    ).write_parquet(events / "listens.parquet")
    pl.DataFrame(
        {"uid": [0], "timestamp": [1002], "item_id": [3], "is_organic": [1]}
    ).write_parquet(events / "dislikes.parquet")
    for name in ("album", "artist"):
        pl.DataFrame(
            {"item_id": list(range(4)), f"{name}_id": list(range(10, 14))}
        ).write_parquet(raw / f"{name}_item_mapping.parquet")
    pl.DataFrame(
        {
            "item_id": list(range(4)),
            "normalized_embed": [
                np.full(128, item + 1, dtype=np.float32).tolist() for item in range(4)
            ],
        },
        schema_overrides={"item_id": pl.UInt32},
    ).write_parquet(raw / "embeddings.parquet")


def test_public_yambda_lifecycle(tmp_path):
    public = tmp_path / "public checkout"
    public.mkdir()
    for name in ("src", "scripts", "configs/yambda-50m", "tests/fixtures/yambda-50m"):
        shutil.copytree(
            ROOT / name,
            public / name,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    # Reuse the approved local environment; this test installs nothing.
    (public / ".venv").symlink_to(ROOT / ".venv", target_is_directory=True)
    env = {
        **os.environ,
        "PYTHONPATH": "src",
        "ACCELERATE_USE_CPU": "true",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "POLARS_MAX_THREADS": "2",
    }
    env.pop("GENRECSYS_TIMINGS_PATH", None)
    env.pop("INTERNAL_STORAGE_YAMBDA", None)

    def run(*args):
        result = subprocess.run(
            args,
            check=False,
            cwd=public,
            env=env,
            text=True,
            capture_output=True,
            timeout=90,
        )
        assert result.returncode == 0, (
            f"{args}:\n{result.stdout}\n{result.stderr[-8000:]}"
        )
        return result.stdout

    python = ".venv/bin/python"
    for path in sorted((public / "scripts").rglob("*.sh")):
        run("bash", "-n", str(path))

    data = Path("work/like data")
    raw = public / data / "raw"
    _write_raw(raw)
    raw_bytes = {path: path.read_bytes() for path in raw.rglob("*.parquet")}

    run(
        python,
        "-m",
        "model_datasets.yambda",
        "build-basis",
        f"--source-dir={data}/raw",
        f"--output-dir={data}/basis",
        "--validation-size=12",
        "--source-revision=local-fixture",
    )
    # The fixture has two train items. Only SID fit size differs from data.sh.
    run(
        python,
        "-m",
        "item_representations.semantic_ids",
        "fit-sid-mapper",
        f"--items={data}/basis/representations/train/items.pq",
        f"--embeddings={data}/basis/representations/train/embeddings.npy",
        f"--output-dir={data}/sid-mapper",
        "--method=rq_kmeans",
        "--codebook-size=2",
        "--codebook-size=2",
        "--codebook-size=2",
        "--device=cpu",
        "--fit-batch-size=8",
        "--eval-batch-size=8",
        "--max-iter=2",
        "--n-init=1",
        "--mapper-only",
    )
    run(
        python,
        "-m",
        "item_representations.semantic_ids",
        "apply-sid-mapper",
        f"--checkpoint={data}/sid-mapper/checkpoint.pkl",
        f"--items={data}/basis/representations/all/items.pq",
        f"--embeddings={data}/basis/representations/all/embeddings.npy",
        f"--output-dir={data}/sids",
        "--device=cpu",
        "--eval-batch-size=8",
    )
    mapping = pl.read_parquet(public / data / "sids/item_to_sid.parquet")
    assert mapping["sid"].list.len().unique().to_list() == [4]
    assert set(mapping["item_id"]) == {"0", "1", "2", "3"}
    env["INTERNAL_STORAGE_YAMBDA"] = str(public / data / "basis/eventhub")

    def prepare(workdir, config, *options):
        run(
            python,
            "-m",
            "model_datasets.yambda",
            "materialize-basis",
            f"--basis={data}/basis",
            f"--sid-mapping={data}/sids/item_to_sid.parquet",
            f"--workdir={workdir}",
            f"--config={config}",
            *options,
        )
        run(
            python,
            "-m",
            "genrecsys",
            "train",
            "prepare-dataset",
            f"--workdir={workdir}",
        )

    prepare("work/full", "configs/yambda-50m/onerec-like-features.yaml")
    full_manifest = json.loads((public / "work/full/manifest.json").read_text())
    assert "sample_limit_per_split" not in full_manifest["metadata"]["protocol"]
    for split in ("train", "validation", "test"):
        source = pl.read_parquet(public / data / f"basis/basis/{split}/samples.pq")
        prepared = pl.read_parquet(public / f"work/full/dataset/{split}/samples.pq")
        assert prepared["sample_id"].to_list() == source["sample_id"].to_list()
    config = "tests/fixtures/yambda-50m/onerec-smoke.yaml"
    workdir = "work/smoke run"
    prepare(workdir, config, "--sample-limit=8")
    output = public / workdir
    for split in ("train", "validation", "test"):
        assert len(pl.read_parquet(output / f"dataset/{split}/samples.pq")) == 8
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["metadata"]["protocol"]["sample_limit_per_split"] == 8
    assert manifest["metadata"]["source"]["revision"] == "local-fixture"
    identity = (output / "checkpoint/dataset_identity.json").read_bytes()
    assert (output / "dataset/events/index.pq").is_file()
    assert list((output / "dataset/events").glob("*.arrow"))
    assert all(path.read_bytes() == content for path, content in raw_bytes.items())
    history = public / data / "basis/eventhub"
    assert list(history.rglob("*.pq"))

    # Nothing downstream can read raw histories or rely on a storage mount.
    history.rename(history.with_name("unused-eventhub"))
    run("bash", "scripts/train.sh", workdir, config, "1")
    assert (output / "checkpoint/model/weights.pt").is_file()
    assert (output / "checkpoint/dataset_identity.json").read_bytes() == identity
    run("bash", "scripts/predict.sh", workdir, "1")
    run("bash", "scripts/evaluate.sh", workdir)
    metrics = json.loads((output / "metrics.json").read_text())
    assert set(metrics["metrics"]) == {"Recall@10", "NDCG@10"}
    assert all(0.0 <= value <= 1.0 for value in metrics["metrics"].values())
    predictions = pl.read_parquet(list((output / "predictions").glob("*.pq")))
    expected = pl.read_parquet(output / "dataset/test/samples.pq")
    assert sorted(predictions["sample_id"]) == sorted(expected["sample_id"])
    assert all(path.read_bytes() == content for path, content in raw_bytes.items())
