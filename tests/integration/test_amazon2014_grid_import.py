"""Prepared GRID snapshot preservation and failure checks, without TensorFlow."""

import gzip
import json
import struct
from pathlib import Path

import polars as pl
import pytest

from model_datasets.amazon2014_grid.prepare import import_beauty
from model_datasets.amazon2014_grid.tfrecord import records


def _varint(value: int) -> bytes:
    output = bytearray()
    while value > 127:
        output.append((value & 127) | 128)
        value >>= 7
    output.append(value)
    return bytes(output)


def _wire(number: int, payload: bytes) -> bytes:
    return _varint(number * 8 + 2) + _varint(len(payload)) + payload


def _crc32c(value: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in value:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    crc ^= 0xFFFFFFFF
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def _write(path: Path, rows: list[dict]) -> None:
    """Independent protobuf wire writer with actual TFRecord masked CRC32C."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as output:
        for row in rows:
            feature_entries = []
            for name, (kind, values) in row.items():
                if kind == "int64_list":
                    feature = _wire(3, _wire(1, b"".join(_varint(v) for v in values)))
                elif kind == "float_list":
                    feature = _wire(
                        2, _wire(1, struct.pack(f"<{len(values)}f", *values))
                    )
                else:
                    feature = _wire(1, b"".join(_wire(1, v.encode()) for v in values))
                entry = _wire(1, name.encode()) + _wire(2, feature)
                feature_entries.append(_wire(1, entry))
            payload = _wire(1, b"".join(feature_entries))
            length = struct.pack("<Q", len(payload))
            output.write(length + struct.pack("<I", _crc32c(length)))
            output.write(payload + struct.pack("<I", _crc32c(payload)))


def _source(
    root: Path,
    *,
    bad_holdout: bool = False,
    full_sequences: list[list[int]] | None = None,
) -> Path:
    texts = {
        i: f"Title: Product {i}; Brand: B; Categories: ['Beauty']; Price: 1; "
        for i in range(27)
    }
    _write(
        root / "items/data_0.tfrecord.gz",
        [
            {
                "id": ("int64_list", [i]),
                "text": ("bytes_list", [text]),
                "embedding": ("float_list", [0.0] * 768),
            }
            for i, text in texts.items()
        ],
    )
    if full_sequences is None:
        full_sequences = [list(range(27)) for _ in range(5)]
    rows = {name: [] for name in ("training", "evaluation", "testing")}
    for user, full in enumerate(full_sequences):
        train = full[:-2]
        validation = train[-20:] + [full[-2]]
        test = full[:-1][-20:] + [full[-1]]
        if bad_holdout:
            validation[0] = 0  # Known item, but outside the allowed source history.
        for name, sequence in (
            ("training", train),
            ("evaluation", validation),
            ("testing", test),
        ):
            rows[name].append(
                {
                    "user_id": ("int64_list", [user]),
                    "sequence_data": ("int64_list", sequence),
                    "text": ("bytes_list", [texts[i] for i in sequence]),
                    "embedding": ("float_list", [0.0] * (768 * len(sequence))),
                }
            )
    for name, split_rows in rows.items():
        _write(
            root / name / "partition_0.tfrecord.gz",
            split_rows,
        )
    return root


def test_preserves_long_train_and_pretrimmed_holdouts(tmp_path: Path):
    source = _source(tmp_path / "source")
    output = tmp_path / "imported"
    import_beauty(source_dir=source, output_dir=output)
    report = json.loads((output / "report.json").read_text())
    assert report["users"] == 5
    assert report["items"] == 27
    assert report["full_interactions"] == 135
    assert report["minimum_full_user_interactions"] == 27
    assert report["minimum_full_item_interactions"] == 5
    assert report["holdout_only_item_ids"] == ["25", "26"]
    assert report["train_contiguous_subsequences_before_sampling"] == 5 * 300
    assert report["framework_dataset_ready"] is False
    assert report["archive_embeddings"]["recomputed"] is False
    train = pl.read_parquet(output / "sequences/train.pq")
    validation = pl.read_parquet(output / "sequences/validation.pq")
    assert train["item_ids"][0].to_list() == [str(i) for i in range(25)]
    assert validation["item_ids"][0].to_list() == [str(i) for i in range(5, 26)]
    catalog = pl.read_parquet(output / "catalog/items.pq")
    assert catalog.filter(pl.col("grid_item_id") == 0)["text"].item().endswith("; ")
    assert not (output / "basis").exists()
    assert not (output / "eventhub").exists()
    assert not list(output.rglob("*.npy"))
    assert len(report["source_files"]) == 4
    assert all(len(value["sha256"]) == 64 for value in report["source_files"])
    with pytest.raises(FileExistsError):
        import_beauty(source_dir=source, output_dir=output)


def test_rejects_inconsistent_holdout_without_publishing(tmp_path: Path):
    source = _source(tmp_path / "source", bad_holdout=True)
    output = tmp_path / "imported"
    with pytest.raises(ValueError, match="holdout/history mismatch"):
        import_beauty(source_dir=source, output_dir=output)
    assert not output.exists()


def test_rejects_truncated_record(tmp_path: Path):
    path = tmp_path / "broken.tfrecord.gz"
    with gzip.open(path, "wb") as output:
        output.write(struct.pack("<Q", 100) + b"\0" * 4 + b"short")
    with pytest.raises(ValueError, match="truncated TFRecord payload"):
        list(records(path))


@pytest.mark.parametrize("item_interactions", [0, 4])
def test_rejects_items_below_five_interactions(tmp_path: Path, item_interactions: int):
    # Only item 0 violates 5-core; each user and every other item remain valid.
    full = [
        list(range(27)) if user < item_interactions else list(range(1, 27))
        for user in range(5)
    ]
    source = _source(tmp_path / "source", full_sequences=full)
    output = tmp_path / "imported"
    with pytest.raises(ValueError, match="5-core"):
        import_beauty(source_dir=source, output_dir=output)
    assert not output.exists()


def test_rejects_user_below_five_interactions(tmp_path: Path):
    # Item counts remain >= 5, but user 0 has only four full interactions.
    full = [[0, 0, 0, 0]] + [list(range(27)) for _ in range(5)]
    source = _source(tmp_path / "source", full_sequences=full)
    output = tmp_path / "imported"
    with pytest.raises(ValueError, match="short training sequence for user 0"):
        import_beauty(source_dir=source, output_dir=output)
    assert not output.exists()


def test_five_core_counts_interactions_including_repeats(tmp_path: Path):
    full = [[0, 0, 0, 0, 0]] + [list(range(27)) for _ in range(5)]
    source = _source(tmp_path / "source", full_sequences=full)
    output = tmp_path / "imported"
    import_beauty(source_dir=source, output_dir=output)
    report = json.loads((output / "report.json").read_text())
    assert report["minimum_full_user_interactions"] == 5
    assert report["minimum_full_item_interactions"] == 5
    sequences = pl.read_parquet(output / "sequences/full.pq")
    assert (
        sequences.filter(pl.col("grid_user_id") == 0)["item_ids"].item().to_list()
        == ["0"] * 5
    )
