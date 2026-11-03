"""Exercise the ported protocol, frozen train-only SID lifecycle and real prepare."""

import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from genrecsys import core
from genrecsys.cli.train import prepare_dataset
from item_representations.semantic_ids.mapping import apply_sid_mapper, fit_sid_mapper
from model_datasets.yambda.basis import _events, _split, build_basis
from model_datasets.yambda.materialize import materialize_basis

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs/yambda-50m"


def _raw(root):
    events = root / "flat/50m"
    events.mkdir(parents=True)
    likes = []
    listens = []
    for tick in range(100):
        for uid in range(4):
            # Cold tracks 2/5 start after train; 2 also survives in listen history.
            item = tick % 2
            if tick >= 75 and tick % 3 == 0:
                item = 2
            if tick >= 80 and tick % 4 == 0:
                item = 5
            likes.append(
                {
                    "uid": uid,
                    "timestamp": 1000 + tick,
                    "item_id": item,
                    "is_organic": tick % 2,
                }
            )
            listens.append(
                {
                    "uid": uid,
                    "timestamp": 1000 + tick,
                    "item_id": 2 if tick >= 75 and tick % 3 == 0 else 3,
                    "is_organic": 1,
                    "played_ratio_pct": 51 + tick % 3,
                    "track_length_seconds": 120 + tick % 2,
                }
            )
            for track in (0, 1):
                listens.append(
                    {
                        "uid": uid,
                        "timestamp": 1000 + tick,
                        "item_id": track,
                        "is_organic": 0,
                        "played_ratio_pct": 60,
                        "track_length_seconds": 100 + 100 * track,
                    }
                )
    # Equal-clock likes must survive as distinct, deterministic single targets.
    likes[1]["item_id"] = 1
    likes.append(dict(likes[1]))  # exact duplicates disappear before activity filtering
    likes.append({"uid": 0, "timestamp": 1000, "item_id": 1, "is_organic": 1})
    # Boundary ratio=50 is excluded, including from duration estimation/history universe.
    listens.append(
        {
            "uid": 0,
            "timestamp": 1000,
            "item_id": 9,
            "is_organic": 1,
            "played_ratio_pct": 50,
            "track_length_seconds": 999,
        }
    )
    pl.DataFrame(likes).write_parquet(events / "likes.parquet")
    pl.DataFrame(listens).write_parquet(events / "listens.parquet")
    pl.DataFrame(
        {"uid": [0], "timestamp": [1001], "item_id": [4], "is_organic": [0]}
    ).write_parquet(events / "dislikes.parquet")
    for name in ("artist", "album"):
        pl.DataFrame(
            {"item_id": [0, 0, 1, 2, 3, 4], f"{name}_id": [10, 99, 11, 12, 13, 14]}
        ).write_parquet(root / f"{name}_item_mapping.parquet")
    pl.DataFrame(
        {
            "item_id": [0, 1, 2, 3, 100],
            "normalized_embed": [
                np.full(128, item + 1, dtype=np.float32).tolist()
                for item in (0, 1, 2, 3, 100)
            ],
        },
        schema_overrides={"item_id": pl.UInt32},
    ).write_parquet(root / "embeddings.parquet")


@pytest.fixture
def basis(tmp_path):
    raw = tmp_path / "raw"
    _raw(raw)
    output = tmp_path / "basis"
    build_basis(
        source_dir=raw,
        output_dir=output,
        validation_size=7,
        source_revision="fixture-revision",
    )
    return output


def test_protocol_catalog_and_row_aligned_audio(basis, tmp_path):
    report = json.loads((basis / "report.json").read_text())
    frames = {
        name: pl.read_parquet(basis / f"basis/{name}/samples.pq")
        for name in ("train", "validation", "validation-full", "test")
    }
    assert report["rows"]["validation"] == 7
    assert report["rows"]["validation_full"] > 7
    assert report["split_ratios"] == [0.75, 0.05, 0.2]
    assert report["split"]["validation_cutoff_ns"] == 1074_000_000_000
    assert report["split"]["test_cutoff_ns"] == 1079_000_000_000
    assert report["split"]["validation_cold_target_samples"] > 0
    assert report["split"]["test_cold_target_samples"] > 0
    assert (
        frames["train"]["timestamp"].max()
        < frames["validation-full"]["timestamp"].min()
    )
    assert (
        frames["validation-full"]["timestamp"].max() < frames["test"]["timestamp"].min()
    )
    assert frames["train"]["sample_id"].n_unique() == len(frames["train"])
    train_targets = set(frames["train"]["target"].explode())
    assert train_targets == {"0", "1"}
    for name in frames:
        catalog = pl.read_parquet(basis / f"basis/{name}/artifacts/items.pq")
        assert set(catalog["item_id"]) == train_targets
        assert set(frames[name]["target"].explode()) <= train_targets
    assert report["candidate_policy"] == "unique filtered train like target items"
    assert report["validation_only_candidate_items"] == 0
    assert report["seen_filtering"] is False
    assert "history_timestamp" not in frames["train"].columns
    for scope, expected in (("train", {"0", "1"}), ("all", {"0", "1", "2", "3", "4"})):
        items = pl.read_parquet(basis / f"representations/{scope}/items.pq")
        matrix = np.load(basis / f"representations/{scope}/embeddings.npy")
        assert set(items["item_id"]) == expected
        for index, item in enumerate(items["item_id"]):
            np.testing.assert_array_equal(
                matrix[index], np.full(128, int(item) + 1 if item != "4" else 0)
            )
    assert report["embeddings"]["missing_items"] == 1
    features = pl.read_parquet(basis / "item_features.pq")
    assert features.filter(pl.col("item_id") == "0")["album_id"].item() == "10"
    assert (
        features.filter(pl.col("item_id") == "3")["item_track_length_seconds"].item()
        == 120
    )
    likes = pl.read_parquet(list((basis / "eventhub/yambda-50m-like").rglob("*.pq")))
    ties = likes.filter(
        (pl.col("client_id") == "0")
        & (pl.col("timestamp").cast(pl.Int64) < 1001_000_000_000)
    )
    assert ties["timestamp"].cast(pl.Int64).to_list() == [
        1000_000_000_000,
        1000_000_001_000,
    ]
    listens = pl.read_parquet(
        list((basis / "eventhub/yambda-50m-listen").rglob("*.pq"))
    )
    assert set(listens["item_id"]) == {"0", "1", "2", "3"}
    assert listens["played_ratio_pct"].min() == 51
    # Rebuild after changing raw row order: samples, subset and target digest stay fixed.
    raw = tmp_path / "reordered"
    _raw(raw)
    path = raw / "flat/50m/likes.parquet"
    pl.read_parquet(path).reverse().write_parquet(path)
    second = tmp_path / "second"
    build_basis(
        source_dir=raw,
        output_dir=second,
        validation_size=7,
        source_revision="fixture-revision",
    )
    other = json.loads((second / "report.json").read_text())
    for name in ("test_target_digest", "candidate_digest", "validation_subset_digest"):
        assert report[name] == other[name]


def test_activity_filter_and_iterative_warm_catalog():
    rows = [
        {"uid": uid, "timestamp": 1000 + tick, "item_id": tick % 2, "is_organic": 1}
        for uid in range(100)
        for tick in range(10)
    ]
    rows += [
        {"uid": 100, "timestamp": 1000 + tick, "item_id": 99, "is_organic": 1}
        for tick in range(100)
    ]
    splits, history, _report = _split(_events(pl.DataFrame(rows), "like"))
    assert "100" not in history["client_id"]
    assert all("99" not in frame["item_id"] for frame in splits.values())


@pytest.fixture
def mapping(basis, tmp_path, request):
    levels = getattr(request, "param", 3)
    fitted = tmp_path / "fit"
    fit_sid_mapper(
        basis / "representations/train/items.pq",
        basis / "representations/train/embeddings.npy",
        fitted,
        method="rq_kmeans_full",
        codebook_size=(2,) * levels,
        device="cpu",
        fit_batch_size=8,
        eval_batch_size=8,
        max_iter=2,
        n_init=1,
        mapper_only=True,
    )
    output = tmp_path / "sid"
    apply_sid_mapper(
        fitted / "checkpoint.pkl",
        basis / "representations/all/items.pq",
        basis / "representations/all/embeddings.npy",
        output,
        device="cpu",
        eval_batch_size=8192,
    )
    contract = json.loads((output / "sid_mapper.json").read_text())
    assert contract["sid_width"] == levels + 1
    assert contract["mapping_rows"] == 5
    return output / "item_to_sid.parquet"


@pytest.mark.parametrize(
    ("branch", "mapping"),
    [
        ("like", 2),
        ("like-features", 2),
        ("all-features", 2),
        ("like", 3),
        ("like", 4),
    ],
    indirect=["mapping"],
)
def test_real_prepare_preserves_samples_catalog_sids_and_causal_histories(
    branch, basis, mapping, tmp_path, monkeypatch
):
    # Inputs are disposable job-local downloads; retain originals for assertions.
    local = tmp_path / "download"
    shutil.copytree(basis, local)
    local_mapping = tmp_path / "mapping.parquet"
    shutil.copyfile(mapping, local_mapping)
    workdir = tmp_path / "prepared"
    width = len(pl.read_parquet(mapping)["sid"].to_list()[0])
    config_path = CONFIGS / f"onerec-{branch}.yaml"
    if width != 4:
        config_path = tmp_path / "config.yaml"
        recipe = core.Config.load(CONFIGS / f"onerec-{branch}.yaml")
        recipe.encoders["item"].embedder["prefix_length"] = width
        recipe.save(config_path)
    monkeypatch.setenv("INTERNAL_STORAGE_YAMBDA", str(local / "eventhub"))
    materialize_basis(
        config=config_path,
        basis=local,
        sid_mapping=local_mapping,
        workdir=workdir,
    )
    prepare_dataset(workdir=workdir)
    prepared = core.checkpoint.Prepared.load(workdir / "checkpoint")
    source = pl.read_parquet(basis / "basis/test/samples.pq")
    test = pl.read_parquet(workdir / "dataset/test/samples.pq")
    assert test["sample_id"].to_list() == source["sample_id"].to_list()
    assert len(pl.read_parquet(workdir / "dataset/validation/samples.pq")) == 7
    assert set(
        pl.read_parquet(workdir / "dataset/test/artifacts/items.pq")["item_id"]
    ) == {"0", "1"}
    sid = prepared.encoder_to_preprocessor["item"]
    assert set(sid.mapping["item_id"]) == {"0", "1", "2", "3", "4"}
    if branch in {"like-features", "all-features"}:
        for feature in ("album_id", "artist_id"):
            encoder = prepared.encoder_to_preprocessor[feature]
            assert "12" not in encoder.values["value"]
            assert encoder.static_transform(pl.Series(["12"])).item() == 0
    assert list((local / "eventhub").rglob("*.pq"))
    provenance = json.loads((workdir / "provenance.json").read_text())
    assert provenance["validation_only_candidate_items"] == 0
    assert provenance["sid_items"] == 5
    # Prepared events retain future rows for rolling test, while observers fit on train.
    store = core.EventStore(workdir / "dataset/events", backend="disk")
    reader = store.open_reader(None)
    events = reader.read("0")
    assert events["timestamp"].max() >= source["timestamp"].min()
    history = reader.read("0", before=source["timestamp"].min())
    assert history["timestamp"].max() < source["timestamp"].min()
    target_item = source["target"].to_list()[0][0]
    assert (
        sid.static_transform(pl.Series([target_item])).to_list()[0]
        in history["item"].to_list()
    )
    # The test target is already in recent history and remains a valid candidate.
    assert (
        target_item
        in pl.read_parquet(workdir / "dataset/test/artifacts/items.pq")["item_id"]
    )
    if branch == "all-features":
        # Cold listen tracks are usable as input without becoming output candidates.
        assert (
            sid.static_transform(pl.Series(["2"])).to_list()[0]
            in events["item"].to_list()
        )
        assert "track_length_seconds" in events.columns
        assert "item_track_length_seconds" in events.columns
        assert events["item_track_length_seconds"].drop_nulls().len() > 0
    if branch == "all-features" or width != 3:
        # Exercise the actual 512-position encoder, SID loss and every decoder width.
        from genrecsys.core.models.onerec import architecture
        from genrecsys.core.models.onerec.catalog import (
            _allowed_codes,
            _catalog,
            _items_by_sid,
            _training_targets,
        )
        from genrecsys.core.models.onerec.config import _model_config
        from genrecsys.core.models.onerec.inference import _predict_batch
        from genrecsys.core.models.onerec.samples import Samples, _loader

        config = core.Config.load(config_path)
        config.training.dataloader.batch_size = 2
        config.training.dataloader.num_workers = 0
        config.training.dataloader.pin_memory = False
        prepared.config = config
        dataset = core.Dataset.load(workdir / "dataset", config, include_test=True)
        model = architecture.OneRec(_model_config(prepared))
        if branch == "organic/like":
            encoder = model.event_encoders["history"]
            assert set(encoder.embeddings) == {"event", "is_organic"}
            assert encoder.embeddings["is_organic"].embedding_dim == 256
            assert encoder.projection[0].in_features == width * 256
            assert encoder.additive == {"event", "is_organic"}
        train = Samples(
            dataset.train_samples.tail(2),
            config,
            code_length=width,
            event_store=store,
            index_to_sid=_training_targets(dataset, dataset.train_artifacts, sid),
            train=True,
        )
        batch = next(iter(_loader(train, config.training.dataloader)))
        loss, _ = model(batch, labels=batch["target"])
        assert torch.isfinite(loss)
        loss.backward()
        assert any(parameter.grad is not None for parameter in model.parameters())
        if branch == "organic/like":
            organic_gradient = (
                model.event_encoders["history"].embeddings["is_organic"].weight.grad
            )
            assert organic_gradient is not None and organic_gradient.abs().sum() > 0
        model.eval()
        test = Samples(
            dataset.test_samples.head(1), config, code_length=width, event_store=store
        )
        batch = next(iter(_loader(test, config.training.dataloader)))
        catalog = _catalog(dataset.test_artifacts, sid, width)
        rankings = _predict_batch(
            model,
            batch,
            _allowed_codes(catalog),
            _items_by_sid(catalog),
            k=2,
            beam_size=2,
        )
        assert {entry["item"] for entry in rankings[0]} == {"0", "1"}


def test_materialization_rejects_incomplete_history_mapping(basis, mapping, tmp_path):
    broken = tmp_path / "broken.parquet"
    pl.read_parquet(mapping).filter(pl.col("item_id") != "3").write_parquet(broken)
    with pytest.raises(ValueError, match="misses history items"):
        materialize_basis(
            config=CONFIGS / "onerec-all-features.yaml",
            basis=basis,
            sid_mapping=broken,
            workdir=tmp_path / "workdir",
        )


def test_materialization_allows_a_shorter_model_input_prefix(basis, mapping, tmp_path):
    recipe = core.Config.load_data(CONFIGS / "onerec-like.yaml")
    recipe.encoders["item"].embedder["prefix_length"] = 1
    config = tmp_path / "short-prefix.yaml"
    recipe.save(config)
    output = tmp_path / "short-prefix-bundle"
    materialize_basis(config=config, basis=basis, sid_mapping=mapping, workdir=output)
    manifest = core.DatasetManifest.load(output / "manifest.json")
    assert manifest.metadata["dataset"] == "yambda-50m"
    assert manifest.metadata["source"]["revision"]
    assert (
        manifest.metadata["protocol"]["name"]
        == "perseus-train-warm-75-5-20-next-like-v1"
    )
    assert (output / "dataset_identity.json").exists()
