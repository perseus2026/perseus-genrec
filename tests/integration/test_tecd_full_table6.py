"""T-ECD Full Table-6 adapters keep views in history, never in targets."""

from __future__ import annotations

import json
import tarfile
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from model_datasets.tecd import full as preprocessing

ROOT = Path(__file__).parents[2]
CONFIG_ROOT = ROOT / "configs/tecd-full"


def _patch_short_window(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {
        "TRAIN_START_DAY": 1,
        "TRAIN_END_DAY": 2,
        "VALIDATION_GAP_DAY": 3,
        "VALIDATION_START_DAY": 4,
        "VALIDATION_END_DAY": 4,
        "TEST_GAP_DAY": 5,
        "TEST_START_DAY": 6,
        "TEST_END_DAY": 6,
        "REPRESENTATION_BATCH_SIZE": 2,
    }.items():
        monkeypatch.setattr(preprocessing, name, value)


def _write_offers_source(root: Path) -> None:
    event_root = root / "dataset/full/offers/events"
    event_root.mkdir(parents=True)
    rows = {
        1: [
            (0, "item_0", "view"),
            (0, "item_1", "click"),
            (1, "item_0", "view"),
        ],
        2: [
            (0, "item_2", "like"),
            (1, "item_1", "click"),
        ],
        4: [
            (0, "item_0", "view"),
            (0, "item_2", "click"),
            (1, "item_3", "click"),
        ],
        6: [
            (0, "item_1", "view"),
            (0, "item_0", "clickout"),
        ],
    }
    empty = pl.DataFrame(
        schema={
            "timestamp": pl.Duration("us"),
            "user_id": pl.Int64,
            "item_id": pl.String,
            "action_type": pl.String,
        }
    )
    for day in range(1, 7):
        day_rows = [
            {
                "timestamp": timedelta(days=day, minutes=index),
                "user_id": user,
                "item_id": item,
                "action_type": action,
            }
            for index, (user, item, action) in enumerate(rows.get(day, []))
        ]
        frame = pl.DataFrame(day_rows) if day_rows else empty
        frame.write_parquet(event_root / f"{day:05d}.pq")

    pl.DataFrame(
        {
            "item_id": ["item_0", "item_1", "item_2", "item_3"],
            "brand_id": [10, 11, 12, None],
            "embedding": [
                np.full(312, index, dtype=np.float32).tolist() for index in range(4)
            ],
        },
        schema_overrides={
            "brand_id": pl.Int64,
            "embedding": pl.Array(pl.Float32, 312),
        },
    ).write_parquet(root / "dataset/full/offers/items.pq")


@pytest.mark.parametrize("sid_width", [2, 3, 4])
def test_offers_builds_shared_targets_and_two_history_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sid_width: int,
) -> None:
    _patch_short_window(monkeypatch)
    source = tmp_path / "source"
    _write_offers_source(source)
    output = tmp_path / "basis"

    preprocessing.build_full_table6_sd(
        source_dir=source,
        output_dir=output,
        domain="offers",
        history_modes=(preprocessing.POSITIVE_MODE, preprocessing.ALL_EVENTS_MODE),
        source_revision="test-revision",
    )

    positive_report = json.loads((output / "positives/report.json").read_text())
    all_report = json.loads((output / "all-events/report.json").read_text())
    assert positive_report["views_included"] is False
    assert all_report["views_included"] is True
    assert positive_report["views_are_targets"] is False
    assert all_report["views_are_targets"] is False
    assert positive_report["positive_event_rows"] == (
        positive_report["target_eligible_event_rows"] + 1
    )
    assert (
        positive_report["target_digest"]["validation"]
        == all_report["target_digest"]["validation"]
    )
    assert (
        positive_report["target_digest"]["test"] == all_report["target_digest"]["test"]
    )
    assert all_report["rows"]["train"] == positive_report["rows"]["train"] + 1

    for mode in ("positives", "all-events"):
        validation = pl.read_parquet(output / mode / "basis/validation/samples.pq")
        test = pl.read_parquet(output / mode / "basis/test/samples.pq")
        assert validation["target"].to_list() == [["item_2"]]
        assert test["target"].to_list() == [["item_0"]]
        assert validation.select(
            (pl.col("timestamp") - pl.col("history_timestamp")).dt.total_days()
        ).to_series().to_list() == [1]

    with tarfile.open(output / "positives/eventhub.tar") as archive:
        positive_names = archive.getnames()
    with tarfile.open(output / "all-events/eventhub.tar") as archive:
        all_names = archive.getnames()
    assert not any("offers-view" in name for name in positive_names)
    assert any("offers-view" in name for name in all_names)
    assert np.load(output / "representations/all/embeddings.npy").shape == (4, 312)
    candidates = pl.read_parquet(output / "positives/basis/train/artifacts/items.pq")
    assert candidates.filter(pl.col("item_id") == "item_3")["brand_id"].to_list() == [
        None
    ]

    mapping = tmp_path / "item_to_sid.parquet"
    pl.DataFrame(
        {
            "item_id": ["item_0", "item_1", "item_2", "item_3"],
            "sid": [
                [2, 258, 514],
                [3, 259, 514],
                [4, 260, 514],
                [5, 261, 514],
            ],
        }
    ).with_columns(
        sid=pl.col("sid").list.slice(0, sid_width)
        if sid_width <= 3
        else pl.col("sid").list.concat(pl.lit([770]))
    ).write_parquet(mapping)
    for mode in ("positives", "all-events"):
        materialized = tmp_path / f"materialized-{mode}"
        preprocessing.materialize_full_table6_basis(
            config=CONFIG_ROOT / f"onerec-offers-{mode}-sd.yaml",
            basis=output / mode,
            sid_mapping=mapping,
            workdir=materialized,
        )
        provenance = json.loads((materialized / "provenance.json").read_text())
        assert provenance["protocol"]["history_mode"] == mode
        assert provenance["sid_items"] == 4
        assert provenance["protocol"]["sid_width"] == sid_width
        assert "OneRec" not in json.dumps(provenance["protocol"])
        assert "FP32" not in json.dumps(provenance["protocol"])
        assert (materialized / "dataset_identity.json").exists()


def test_download_wrapper_requests_one_fixed_offers_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO", logger=preprocessing.__name__)
    calls: list[dict[str, object]] = []
    builds: list[tuple[str, ...]] = []
    monkeypatch.setattr(preprocessing, "DOWNLOAD_PROGRESS_INTERVAL_SECONDS", 0.01)

    def snapshot_download(**kwargs: object) -> None:
        calls.append(kwargs)
        patterns = kwargs["allow_patterns"]
        assert isinstance(patterns, list)
        first = Path(str(kwargs["local_dir"])) / patterns[0]
        first.parent.mkdir(parents=True)
        first.write_bytes(b"downloaded")
        time.sleep(0.03)

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        snapshot_download,
    )
    monkeypatch.setattr(
        preprocessing,
        "build_full_table6_sd",
        lambda **kwargs: builds.append(kwargs["history_modes"]),
    )

    preprocessing.download_and_build_full_table6(
        domain="offers",
        output_dir=tmp_path / "output",
        revision="revision",
        max_workers=3,
    )

    assert len(calls) == 1
    patterns = calls[0]["allow_patterns"]
    assert isinstance(patterns, list)
    assert patterns[0].endswith("/events/01092.pq")
    assert patterns[-2].endswith("/events/01307.pq")
    assert patterns[-1].endswith("/items.pq")
    assert len(patterns) == 217
    assert calls[0]["revision"] == "revision"
    assert calls[0]["max_workers"] == 3
    assert builds == [("positives", "all-events")]
    messages = [record.getMessage() for record in caplog.records]
    assert any("Starting T-ECD Full Offers download" in message for message in messages)
    assert any(
        "Downloading T-ECD Full Offers: files=1/217" in message for message in messages
    )
    assert any(
        "Completed T-ECD Full Offers download" in message for message in messages
    )


def test_common_holdout_excludes_view_only_history_and_discards_consumed_raw(
    tmp_path, monkeypatch
):
    _patch_short_window(monkeypatch)
    source = tmp_path / "source"
    _write_offers_source(source)
    event_root = source / "dataset/full/offers/events"
    for day, user, action in (
        (1, 2, "view"),
        (1, 3, "view"),
        (4, 3, "click"),
        (6, 2, "click"),
    ):
        path = event_root / f"{day:05d}.pq"
        frame = pl.read_parquet(path)
        pl.concat(
            [
                frame,
                pl.DataFrame(
                    {
                        "timestamp": [timedelta(days=day, minutes=20 + user)],
                        "user_id": [user],
                        "item_id": ["item_0"],
                        "action_type": [action],
                    },
                    schema=frame.schema,
                ),
            ]
        ).write_parquet(path)
    modes = (preprocessing.POSITIVE_MODE, preprocessing.ALL_EVENTS_MODE)
    previous = tmp_path / "previous"
    preprocessing.build_full_table6_sd(
        source_dir=source,
        output_dir=previous,
        domain="offers",
        history_modes=modes,
    )
    assert len(list(source.rglob("*.pq"))) == 7
    for fold, view_only_user in (("validation", "3"), ("test", "2")):
        all_events = pl.read_parquet(previous / f"all-events/basis/{fold}/samples.pq")
        positives = pl.read_parquet(previous / f"positives/basis/{fold}/samples.pq")
        assert view_only_user in all_events["client_id"]
        assert view_only_user not in positives["client_id"]
    output = tmp_path / "common"
    preprocessing.build_full_table6_sd(
        source_dir=source,
        output_dir=output,
        domain="offers",
        history_modes=modes,
        shared_holdout=True,
        delete_source_files=True,
    )
    assert not list(source.rglob("*.pq"))
    for fold in ("validation", "test"):
        positives = pl.read_parquet(output / f"positives/basis/{fold}/samples.pq")
        all_events = pl.read_parquet(output / f"all-events/basis/{fold}/samples.pq")
        assert positives.sort("client_id").equals(all_events.sort("client_id"))
        for artifact in ("items.pq",):
            assert pl.read_parquet(
                output / f"positives/basis/{fold}/artifacts/{artifact}"
            ).equals(
                pl.read_parquet(
                    output / f"all-events/basis/{fold}/artifacts/{artifact}"
                )
            )
    reports = [
        json.loads((output / mode / "report.json").read_text()) for mode in modes
    ]
    assert all(report["shared_holdout"] for report in reports)
    assert reports[0]["source"]["event_compressed_bytes"] > 0
    assert reports[0]["target_digest"]["test"] == reports[1]["target_digest"]["test"]


def test_failed_partition_keeps_its_raw_input(tmp_path, monkeypatch):
    _patch_short_window(monkeypatch)
    source = tmp_path / "source"
    _write_offers_source(source)
    path = source / "dataset/full/offers/events/00001.pq"

    def fail(*args, **kwargs):
        raise RuntimeError("partition write failed")

    monkeypatch.setattr(preprocessing, "_append_eventhub_partition", fail)
    with pytest.raises(RuntimeError, match="partition write failed"):
        preprocessing.build_full_table6_sd(
            source_dir=source,
            output_dir=tmp_path / "output",
            domain="offers",
            history_modes=("positives", "all-events"),
            delete_source_files=True,
        )
    assert path.is_file()
