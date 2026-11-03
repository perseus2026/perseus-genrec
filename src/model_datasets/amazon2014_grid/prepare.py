"""CPU-only import of prepared GRID Amazon snapshots, preserving author splits."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import polars as pl

from genrecsys.utils.files import file_sha256, publish_directory
from model_datasets.amazon2014_grid.tfrecord import records

logger = logging.getLogger(__name__)
SOURCE_URL = (
    "https://drive.google.com/file/d/1B5_q_MT3GYxmHLrMK0-lAqgpbAuikKEz/view?usp=sharing"
)
PROTOCOL = "amazon2014-grid-p5-prepared-beauty-v1"
SPLITS = {"training": "train", "evaluation": "validation", "testing": "test"}
Category = Literal["beauty", "toys", "sports"]


def source_protocol(category: str) -> str:
    if category not in ("beauty", "toys", "sports"):
        raise ValueError(f"Unsupported GRID Amazon category: {category}")
    return f"amazon2014-grid-p5-prepared-{category}-v1"


def event_name(category: str) -> str:
    source_protocol(category)
    return f"amazon2014grid-{category}-review"


def check_source_inputs(
    *,
    prepared_source_dir: Path,
    source_dir: Path,
    output_path: Path,
    scope: Literal["items", "sequences", "all"] = "all",
) -> None:
    """Verify actual native GRID inputs against the CPU import, before GPU work."""
    if scope not in ("items", "sequences", "all"):
        raise ValueError("scope must be items, sequences or all")
    report_path = prepared_source_dir / "report.json"
    report = json.loads(report_path.read_text())
    if report["protocol"] != source_protocol(report["category"]):
        raise ValueError("Expected a validated GRID Amazon import")
    for relative, digest in report["artifact_files"].items():
        if file_sha256(prepared_source_dir / relative) != digest:
            raise ValueError(f"Prepared source SHA-256 mismatch: {relative}")
    folders = {"items"} if scope == "items" else set(SPLITS)
    if scope == "all":
        folders.add("items")
    expected = {
        r["path"]: r["sha256"]
        for r in report["source_files"]
        if Path(r["path"]).parts[0] in folders
    }
    actual = {
        str(p.relative_to(source_dir)): file_sha256(p)
        for folder in folders
        for p in (source_dir / folder).glob("*.tfrecord.gz")
    }
    if not expected or expected != actual:
        raise ValueError("Native GRID input files differ from the validated source")
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "category": report["category"],
                "source_protocol": report["protocol"],
                "source_report_sha256": file_sha256(report_path),
                "catalog_sha256": file_sha256(prepared_source_dir / "catalog/items.pq"),
                "sid_fit_diagnostics": report["sid_fit_diagnostics"],
                "checked_scope": scope,
                "source_files_sha256": actual,
            },
            indent=2,
        )
        + "\n"
    )


def _features(row: dict[str, tuple[str, Any]], expected: dict[str, str]) -> None:
    actual = {name: value[0] for name, value in row.items()}
    if actual != expected:
        raise ValueError(f"Unexpected GRID record schema: {actual!r}")


def _identifier(values: Any) -> int:
    if len(values) != 1 or values[0] < 0:
        raise ValueError("Expected one nonnegative GRID identifier")
    return int(values[0])


def import_beauty(*, source_dir: Path, output_dir: Path) -> None:
    """Backward-compatible Beauty entrypoint."""
    import_amazon(source_dir=source_dir, output_dir=output_dir, category="beauty")


def import_amazon(
    *, source_dir: Path, output_dir: Path, category: Category = "beauty"
) -> None:
    """Import author TFRecords into Parquet catalogs and prepared sequences.

    source_dir points to amazon_data/<category>. This is an upstream source artifact,
    NOT a materialized framework dataset: no EventHub, samples, config or SID
    mapping are invented. Existing nonempty output directories are not replaced.
    """
    protocol = source_protocol(category)
    paths = {}
    for name in ("items", *SPLITS):
        paths[name] = sorted((source_dir / name).glob("*.tfrecord.gz"))
        if not paths[name]:
            raise FileNotFoundError(f"No TFRecord files in {source_dir / name}")

    with publish_directory(output_dir, overwrite=False) as output:
        source_files = []

        def read_folder(name: str):
            for path in paths[name]:
                source_files.append(
                    {
                        "path": str(path.relative_to(source_dir)),
                        "bytes": path.stat().st_size,
                        "sha256": file_sha256(path),
                    }
                )
                yield from records(path)

        items = {}
        for row in read_folder("items"):
            _features(
                row,
                {"id": "int64_list", "text": "bytes_list", "embedding": "float_list"},
            )
            item_id = _identifier(row["id"][1])
            if item_id in items:
                raise ValueError(f"Duplicate item ID: {item_id}")
            if len(row["text"][1]) != 1 or len(row["embedding"][1]) != 768:
                raise ValueError(
                    f"Unexpected text or embedding shape for item {item_id}"
                )
            items[item_id] = bytes(row["text"][1][0]).decode("utf-8")
        if not items or set(items) != set(range(len(items))):
            raise ValueError(
                "Expected a nonempty contiguous zero-based GRID item catalog"
            )

        (output / "catalog").mkdir()
        pl.DataFrame(
            {
                "item_id": [str(i) for i in sorted(items)],
                "grid_item_id": sorted(items),
                "text": [items[i] for i in sorted(items)],
            }
        ).write_parquet(output / "catalog/items.pq")

        sequences = {}
        statistics = {}
        (output / "sequences").mkdir()
        for folder, split in SPLITS.items():
            users = {}
            item_counts = Counter()
            for row in read_folder(folder):
                _features(
                    row,
                    {
                        "user_id": "int64_list",
                        "sequence_data": "int64_list",
                        "text": "bytes_list",
                        "embedding": "float_list",
                    },
                )
                user_id = _identifier(row["user_id"][1])
                sequence = list(row["sequence_data"][1])
                if user_id in users:
                    raise ValueError(f"Duplicate {folder} user ID: {user_id}")
                if len(sequence) < {"train": 3, "validation": 4, "test": 5}[split]:
                    raise ValueError(
                        f"Unexpected short {folder} sequence for user {user_id}"
                    )
                if any(item not in items for item in sequence):
                    raise ValueError(f"Unknown item in {folder} user {user_id}")
                texts = row["text"][1]
                if len(texts) != len(sequence) or any(
                    bytes(text).decode("utf-8") != items[item]
                    for text, item in zip(texts, sequence, strict=False)
                ):
                    raise ValueError(f"Item text mismatch in {folder} user {user_id}")
                if len(row["embedding"][1]) != 768 * len(sequence):
                    raise ValueError(
                        f"Unexpected embedding length in {folder} user {user_id}"
                    )
                users[user_id] = sequence
                item_counts.update(sequence)
            if not users:
                raise ValueError(f"No users in {folder}")
            sequences[split] = users
            ordered_users = sorted(users)
            pl.DataFrame(
                {
                    "client_id": [str(u) for u in ordered_users],
                    "grid_user_id": ordered_users,
                    "item_ids": [[str(i) for i in users[u]] for u in ordered_users],
                },
                schema={
                    "client_id": pl.String,
                    "grid_user_id": pl.Int64,
                    "item_ids": pl.List(pl.String),
                },
            ).write_parquet(output / f"sequences/{split}.pq")
            lengths = [len(s) for s in users.values()]
            statistics[split] = {
                "files": len(paths[folder]),
                "users": len(users),
                "interactions": sum(lengths),
                "items": len(item_counts),
                "minimum_length": min(lengths),
                "maximum_length": max(lengths),
                "sequences_longer_than_20": sum(n > 20 for n in lengths),
            }
            logger.info("Imported %s: %s", split, statistics[split])

        train, validation, test = (
            sequences[name] for name in ("train", "validation", "test")
        )
        if train.keys() != validation.keys() or train.keys() != test.keys():
            raise ValueError("GRID split user sets differ")
        full = {}
        for user, sequence in train.items():
            val, tst = validation[user], test[user]
            if val[:-1] != sequence[-20:] or tst[:-1] != (sequence + [val[-1]])[-20:]:
                raise ValueError(f"GRID holdout/history mismatch for user {user}")
            full[user] = sequence + [val[-1], tst[-1]]
        counts = Counter(item for sequence in full.values() for item in sequence)
        if set(counts) != set(items) or min(counts.values()) < 5:
            raise ValueError(
                "Reconstructed full sequences do not cover the published 5-core catalog"
            )
        observed = {item for sequence in train.values() for item in sequence}
        pl.DataFrame(
            {
                "client_id": [str(u) for u in sorted(full)],
                "grid_user_id": sorted(full),
                "item_ids": [[str(i) for i in full[u]] for u in sorted(full)],
            }
        ).write_parquet(output / "sequences/full.pq")
        pl.DataFrame({"item_id": [str(i) for i in sorted(observed)]}).write_parquet(
            output / "catalog/observed_train_items.pq"
        )
        report = {
            "protocol": protocol,
            "category": category,
            "stage": "prepared_source_import",
            "framework_dataset_ready": False,
            "source_url": SOURCE_URL,
            "source_files": source_files,
            "source_integrity": {
                "gzip_crc": "verified",
                "record_framing": "verified",
                "tfrecord_crc32c": "not_verified",
            },
            "users": len(train),
            "items": len(items),
            "full_interactions": sum(map(len, full.values())),
            "minimum_full_user_interactions": min(map(len, full.values())),
            "minimum_full_item_interactions": min(counts.values()),
            "split_statistics": statistics,
            "holdout_history_limit_items": 20,
            "train_source_truncated": False,
            "train_contiguous_subsequences_before_sampling": sum(
                len(s) * (len(s) - 1) // 2 for s in train.values()
            ),
            "holdout_only_item_ids": [str(i) for i in sorted(set(items) - observed)],
            "sid_fit_diagnostics": {
                "fit_scope": "full_author_catalog",
                "catalog_items": len(items),
                "raw_train_items": len(observed),
                "outside_raw_train_items": len(set(items) - observed),
                "outside_raw_train_fraction": len(set(items) - observed) / len(items),
                "holdout_targets_outside_raw_train": {
                    split: {
                        "count": sum(s[-1] not in observed for s in users.values()),
                        "fraction": sum(s[-1] not in observed for s in users.values())
                        / len(users),
                    }
                    for split, users in (("validation", validation), ("test", test))
                },
                "sid_fitted": False,
            },
            "timestamps": "absent; no real or synthetic times generated by this import",
            "item_ids": "original GRID integer ID serialized as string; grid_item_id retained",
            "text": "author UTF-8 text preserved verbatim; no metadata reserialization",
            "archive_embeddings": {
                "dimension": 768,
                "model_provenance": "unknown",
                "exported": False,
                "recomputed": False,
            },
            "grid_quantizer_fit_scope": "full items catalog in author config; observed_train_items is diagnostic only",
            "artifact_files": {
                str(path.relative_to(output)): file_sha256(path)
                for path in sorted(output.rglob("*.pq"))
            },
            "pending": [
                "EventHub identity/time protocol",
                "train window materialization or exact online sampler",
                "frozen SID mapping",
                "framework config and presets",
                "materialized manifests",
                "representation coverage and framework end-to-end test",
            ],
        }
        (output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
