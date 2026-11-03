"""Full SD/MD preparation through native stages preserves the reference dataset."""

import tarfile
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from genrecsys import core
from genrecsys.cli.train import prepare_dataset
from model_datasets.tecd import full, payments
from scripts import experiments
from tests.integration.test_tecd_full_table6 import (
    _patch_short_window,
    _write_offers_source,
)

ROOT = Path(__file__).resolve().parents[2]


def _data(tmp_path, monkeypatch):
    _patch_short_window(monkeypatch)
    data = tmp_path / "data"
    for domain in experiments.FULL_DOMAINS:
        raw = tmp_path / f"raw-{domain}"
        _write_offers_source(raw)
        source = raw / "dataset/full/offers"
        if domain != "offers":
            source.rename(raw / f"dataset/full/{domain}")
            source = raw / f"dataset/full/{domain}"
        for path in [source / "items.pq", *sorted((source / "events").glob("*.pq"))]:
            frame = pl.read_parquet(path).with_columns(
                (pl.lit(domain + "_") + pl.col("item_id")).alias("item_id")
            )
            if domain == "retail" and "brand_id" in frame.columns:
                frame = frame.with_columns(pl.col("brand_id").fill_null(13))
            if domain == "retail" and "action_type" in frame.columns:
                frame = frame.with_columns(
                    pl.col("action_type").replace(
                        {"like": "added_to_cart", "clickout": "order"}
                    )
                )
            frame.write_parquet(path)
        basis = data / "basis" / domain
        full.build_full_table6_sd(
            source_dir=raw,
            output_dir=basis,
            domain=domain,
            history_modes=experiments.FULL_MODES,
            source_revision="fixture",
            shared_holdout=True,
        )
        catalog = pl.read_parquet(basis / "positives/basis/train/artifacts/items.pq")
        mapping = data / "sid" / domain / "catalog/item_to_sid.parquet"
        mapping.parent.mkdir(parents=True)
        pl.DataFrame(
            {
                "item_id": catalog["item_id"],
                "sid": [[2 + i, 258 + i, 514] for i in range(len(catalog))],
            }
        ).write_parquet(mapping)
    for event in (payments.PAYMENT_EVENT, payments.RECEIPT_EVENT):
        payments._write_events(
            data / "payments/eventhub",
            event,
            pl.DataFrame(
                {
                    "client_id": ["0", "1"],
                    "partition_id": [
                        payments.client_partition_id(v) for v in ("0", "1")
                    ],
                    "timestamp": [datetime(1970, 1, 2)] * 2,  # noqa: DTZ001
                    "brand_id": ["10", "11"],
                }
            ),
        )
    with tarfile.open(data / "payments/eventhub.tar", "w") as archive:
        archive.add(data / "payments/eventhub", arcname="eventhub")
    return data


@pytest.mark.parametrize(
    "domain,mode,setup",
    [("retail", "positives", "sd"), ("offers", "all-events", "md")],
)
def test_full_plan_preserves_native_preparation(
    tmp_path, monkeypatch, domain, mode, setup
):
    dataset = "tecd-full"
    data = _data(tmp_path, monkeypatch)
    source = f"{domain}-{mode}-{setup}"
    model_name = "onerec-" + source
    config_path = ROOT / "configs/tecd-full" / f"{model_name}.yaml"
    # A separate native preparation is the reference for all split/history bytes.
    combined = tmp_path / "reference-histories"
    combined.mkdir()
    for other in experiments.FULL_DOMAINS if setup == "md" else (domain,):
        basis = data / "basis" / other / mode
        with tarfile.open(basis / "eventhub.tar") as archive:
            archive.extractall(combined, filter="data")
    if setup == "md":
        with tarfile.open(data / "payments/eventhub.tar") as archive:
            archive.extractall(combined, filter="data")
    monkeypatch.setenv("INTERNAL_STORAGE_TECD", str(combined / "eventhub"))
    reference = tmp_path / "reference"
    full.materialize_full_table6_basis(
        config=config_path,
        basis=data / "basis" / domain / mode,
        sid_mapping=data / "sid" / domain / "catalog/item_to_sid.parquet",
        workdir=reference,
    )
    prepare_dataset(workdir=reference)
    frozen = {
        p.relative_to(data): p.read_bytes() for p in data.rglob("*") if p.is_file()
    }
    stages = experiments.plan(dataset)
    needed = {f"prepare_{source.replace('-', '_')}"}
    for step in reversed(stages):
        if step["task"] in needed:
            needed.update(step["parents"])
    skip = {"build_full", "build_payments", "fit_sid", "apply_sid"}
    for step in stages:
        if step["task"] not in needed or step["action"] in skip:
            continue
        experiments.run_stage(
            dataset, step["task"], data_dir=data, run_dir=tmp_path / "run"
        )
    prepared = tmp_path / "run/prepared" / source
    recipe = core.Config.load(config_path)
    actual = core.Dataset.load(prepared / "dataset", recipe, include_test=True)
    expected = core.Dataset.load(reference / "dataset", recipe, include_test=True)
    for split in ("train", "validation", "test"):
        assert getattr(actual, f"{split}_samples").equals(
            getattr(expected, f"{split}_samples")
        )
        assert getattr(actual, f"{split}_artifacts").items.equals(
            getattr(expected, f"{split}_artifacts").items
        )
    for client in pl.read_parquet(reference / "dataset/events/index.pq")["client_id"]:
        assert (
            actual.event_store.open_reader(None)
            .read(client)
            .equals(expected.event_store.open_reader(None).read(client))
        )
    assert (
        core.checkpoint.Prepared.load(prepared / "checkpoint").data_contract
        == core.checkpoint.Prepared.load(reference / "checkpoint").data_contract
    )
    assert all(
        (data / path).read_bytes() == content for path, content in frozen.items()
    )
