"""Exercise Small Final on actual tiny bases, plus isolated delivery and reporting."""

# ruff: noqa: DTZ001

import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from genrecsys import core
from item_representations.semantic_ids.mapping import fit_sid_mapper
from model_datasets.tecd import cross_history
from model_datasets.tecd.marketplace import build_small_marketplace
from model_datasets.tecd.retail import (
    archive_eventhub,
    build_small_retail,
    extract_eventhub,
)
from scripts.experiments import run_stage

ROOT = Path(__file__).resolve().parents[2]
DOMAINS = ("retail", "marketplace")
SETUPS = ("rq1", "cross-global")


def _raw_small(root, domain):
    events = root / f"dataset/small/{domain}/events"
    events.mkdir(parents=True)
    offset = 0 if domain == "retail" else 100
    start = datetime(2025, 1, 1, 12)
    actions = (
        ("view", "click", "added_to_cart", "order")
        if domain == "retail"
        else ("view", "like", "clickout", "click")
    )
    rows = []
    for day in range(12):
        for user in range(10):
            target = start + timedelta(days=day, minutes=user, hours=3)
            for index, action in enumerate(actions):
                timestamp = target + timedelta(hours=index - 3)
                rows.append(
                    {
                        "timestamp": timedelta(
                            microseconds=int(timestamp.timestamp() * 1_000_000)
                        ),
                        "user_id": user,
                        "item_id": offset + (day + user) % 5,
                        "action_type": action,
                        "subdomain": domain,
                        "os": "ios",
                    }
                )
    pl.DataFrame(rows).write_parquet(events / "events.pq")
    pl.DataFrame(
        {
            "item_id": [offset + item for item in range(5)],
            "brand_id": [0, 1, 2, 0, 1],
            "category": [0, 0, 1, 1, 0],
            "subcategory": list(range(5)),
            "price": [10.0, 20.0, 30.0, 40.0, 50.0],
            "embedding": [
                np.full(300, item + 1, dtype=np.float32).tolist() for item in range(5)
            ],
        },
        schema_overrides={"embedding": pl.Array(pl.Float32, 300)},
    ).write_parquet(root / f"dataset/small/{domain}/items.pq")
    pl.DataFrame(
        {
            "user_id": list(range(9)),
            "socdem_cluster": [user % 3 for user in range(9)],
            "region": [user % 2 for user in range(9)],
        }
    ).write_parquet(root / "dataset/small/users.pq")


@pytest.fixture
def tiny_bases(tmp_path):
    result = {}
    for domain, build in (
        ("retail", build_small_retail),
        ("marketplace", build_small_marketplace),
    ):
        raw = tmp_path / f"raw-{domain}"
        _raw_small(raw, domain)
        output = tmp_path / f"basis-{domain}"
        build(
            source_dir=raw,
            output_dir=output,
            source_revision="pinned-test-revision",
            history_gap_hours=0,
            include_views=False,
            train_ratio=0.75,
        )
        archive_eventhub(basis=output)
        result[domain] = output
    return result


@pytest.mark.parametrize("domain", DOMAINS)
def test_small_split_keeps_the_original_test_population(domain, tiny_bases, tmp_path):
    output = tiny_bases[domain]
    report = json.loads((output / "report.json").read_text())
    assert report["history_gap_hours"] == 0
    assert report["includes_views"] is False
    assert report["split"]["train_ratio"] == 0.75
    assert report["split"]["validation_ratio"] == 0.05
    assert report["split"]["test_ratio"] == 0.2
    assert (
        report["test_target_digest"] == report["perseus_reference_test_target_digest"]
    )
    original = tmp_path / "original-70-10-20"
    build = build_small_retail if domain == "retail" else build_small_marketplace
    build(
        source_dir=tmp_path / f"raw-{domain}",
        output_dir=original,
        source_revision="pinned-test-revision",
        history_gap_hours=0,
        include_views=False,
    )
    for relative in ("basis/test/samples.pq", "basis/test/artifacts/items.pq"):
        assert pl.read_parquet(original / relative).equals(
            pl.read_parquet(output / relative)
        )
    assert len(pl.read_parquet(output / "basis/train/samples.pq")) > len(
        pl.read_parquet(original / "basis/train/samples.pq")
    )


@pytest.mark.parametrize("domain", DOMAINS)
def test_shared_prepare_keeps_targets_sids_and_frozen_test_partitions(
    domain, tiny_bases, tmp_path
):
    data, runs = tmp_path / "data", tmp_path / "runs"
    for name, source in tiny_bases.items():
        destination = data / "basis" / name
        shutil.copytree(source, destination)
        extract_eventhub(basis=destination)
    basis = data / "basis" / domain
    fit_sid_mapper(
        basis / "representations/train/items.pq",
        basis / "representations/train/embeddings.npy",
        data / "sid" / domain,
        method="rq_kmeans_full",
        codebook_size=(2, 2),
        device="cpu",
        fit_batch_size=8,
        eval_batch_size=8,
        max_iter=2,
        n_init=1,
        mapper_only=True,
    )
    clients = [str(user) for user in range(10)]
    history = pl.DataFrame(
        {
            "client_id": clients,
            "timestamp": [datetime(2025, 1, 1, 10)] * len(clients),
            "brand_id": ["1"] * len(clients),
            "price": [5.0] * len(clients),
        }
    )
    cross = data / "cross-history/eventhub"
    cross_history._write_payment_events(cross, history)
    cross_history._write_offer_events(
        cross, history.with_columns(action=pl.lit("click"))
    )
    identities = []
    for setup in SETUPS:
        run_stage(
            "tecd-small",
            f"prepare_{domain}_{setup.replace('-', '_')}",
            data_dir=data,
            run_dir=runs,
        )
        workdir = runs / "prepared" / f"{domain}-{setup}"
        prepared = core.checkpoint.Prepared.load(workdir / "checkpoint")
        config = core.Config.load(
            ROOT / f"configs/tecd-small/onerec-{domain}-{setup}.yaml"
        )
        dataset = core.Dataset.load(workdir / "dataset", config, include_test=True)
        for fold in ("validation", "test"):
            source = pl.read_parquet(basis / f"basis/{fold}/samples.pq")
            keys = ["client_id", "timestamp", "target"]
            assert (
                getattr(dataset, f"{fold}_samples")
                .select(keys)
                .equals(source.select(keys).sort("timestamp", "client_id"))
            )
        assert 0 < len(list((workdir / "samples").glob("*.pq"))) <= 100
        actual = pl.read_parquet(list((workdir / "samples").glob("*.pq")))
        assert sorted(actual["sample_id"]) == sorted(dataset.test_samples["sample_id"])
        identities.append(prepared.encoder_to_preprocessor["item"].mapping)
    assert identities[0].equals(identities[1])
    assert (data / "sid" / domain / "checkpoint.pkl").is_file()
    assert (basis / "representations/all/embeddings.npy").is_file()


def test_payment_worker_merge_consumes_only_successfully_moved_files(tmp_path):
    source = tmp_path / "worker/eventhub"
    path = source / "payments/day/partition.pq"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"worker data")
    destination = tmp_path / "merged/eventhub"
    cross_history._merge_eventhub(source, destination)
    assert not path.exists()
    target = destination / "payments/day/partition.pq"
    assert target.read_bytes() == b"worker data"
    path.write_bytes(b"conflicting data")
    with pytest.raises(FileExistsError):
        cross_history._merge_eventhub(source, destination)
    assert path.read_bytes() == b"conflicting data"
    assert target.read_bytes() == b"worker data"


@pytest.mark.parametrize("failed", [False, True])
def test_cross_source_cleanup_only_follows_success(tmp_path, monkeypatch, failed):
    import huggingface_hub

    source = tmp_path / "download"

    def temporary(**_):
        source.mkdir()
        return str(source)

    def download(**arguments):
        assert arguments["local_dir"] == str(source)
        assert arguments["revision"] == "pinned-revision"
        (source / "download.pq").write_bytes(b"disposable raw copy")

    offers = pl.DataFrame(
        {
            "client_id": ["1"],
            "timestamp": [datetime(2025, 1, 1)],
            "brand_id": ["1"],
            "action": ["click"],
        }
    )

    def workers(**arguments):
        assert arguments["source_dir"] == source
        if failed:
            raise RuntimeError("payment worker failed")
        output = source / "worker"
        cross_history._write_payment_events(output / "eventhub", offers)
        return [output], 1

    monkeypatch.setattr(cross_history, "tempfile", SimpleNamespace(mkdtemp=temporary))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(cross_history, "_run_payment_workers", workers)
    monkeypatch.setattr(cross_history, "_load_offers", lambda _: offers)
    arguments = {
        "output_dir": tmp_path / "output",
        "revision": "pinned-revision",
        "payment_start_day": 1,
        "payment_end_day": 1,
    }
    if failed:
        with pytest.raises(RuntimeError, match="payment worker failed"):
            cross_history.download_and_build_cross_history(**arguments)
        assert (source / "download.pq").read_bytes() == b"disposable raw copy"
    else:
        cross_history.download_and_build_cross_history(**arguments)
        assert not source.exists()
        assert (arguments["output_dir"] / "eventhub.tar").is_file()
