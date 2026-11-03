"""Bind shared Yambda samples and frozen SIDs to each feature recipe."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from genrecsys.core import Config
from genrecsys.core.dataset_manifest import finalize_bundle
from genrecsys.core.encoders.semantic_id import Preprocessor
from genrecsys.core.event_feature_mapping import Spec
from genrecsys.core.resources import copy_resource
from genrecsys.utils.files import file_sha256, publish_directory
from genrecsys.utils.progress import logged_stage
from model_datasets.yambda.basis import ITEM_FEATURES


@logged_stage
def materialize_basis(
    *,
    config: Path,
    basis: Path,
    sid_mapping: Path,
    workdir: Path,
    sample_limit: int | None = None,
) -> None:
    """Bind frozen SIDs; optionally limit every split for a local smoke run."""
    if sample_limit is not None and sample_limit < 1:
        raise ValueError("sample_limit must be positive")
    recipe = Config.load_data(config)
    mapping = pl.read_parquet(sid_mapping)
    sid = Preprocessor(mapping, item_id_column="item_id")
    features = pl.read_parquet(basis / "item_features.pq")
    if len(features.join(mapping.select("item_id"), on="item_id", how="anti")):
        raise ValueError("Yambda SID mapping misses history items")
    source_report = json.loads((basis / "report.json").read_text())
    provenance = {
        **source_report,
        "sid_mapping_sha256": file_sha256(sid_mapping),
        "sid_items": len(mapping),
        "sid_code_length": sid.code_length,
        "sid_vocabulary_size": sid.vocab_size,
    }
    with publish_directory(workdir, overwrite=False) as output:
        resources = output / "resources"
        resources.mkdir()
        copy_resource(sid_mapping, resources / "item_to_sid.parquet")
        features.rename({"item_id": "item"}).write_parquet(
            resources / "item_features.pq"
        )
        (resources / "manifest.json").write_text(
            json.dumps(
                {
                    "item_to_sid": "item_to_sid.parquet",
                    "item_features": "item_features.pq",
                },
                indent=2,
            )
            + "\n"
        )
        for split in ("train", "validation", "test"):
            source = basis / "basis" / split
            destination = output / "basis" / split
            (destination / "artifacts").mkdir(parents=True)
            if sample_limit is None:
                copy_resource(source / "samples.pq", destination / "samples.pq")
            else:
                pl.scan_parquet(source / "samples.pq").head(
                    sample_limit
                ).collect().write_parquet(destination / "samples.pq")
            copy_resource(
                source / "artifacts/items.pq", destination / "artifacts/items.pq"
            )
        mappings = tuple(
            Spec(
                feature=name,
                resource="item_features",
                attributes=("item",),
                value_column=name,
                events=tuple(recipe.events_for_feature(name)),
            )
            for name in ITEM_FEATURES
            if name in recipe.features
        )
        protocol = {"name": source_report["protocol"]}
        protocol.update(
            {
                key: source_report[key]
                for key in (
                    "split",
                    "split_ratios",
                    "candidate_policy",
                    "candidate_digest",
                    "validation_seed",
                    "validation_requested_size",
                    "validation_subset_digest",
                    "history_gap_hours",
                    "history_cutoff",
                    "seen_filtering",
                    "sid_fit_scope",
                )
            }
        )
        if sample_limit is not None:
            protocol["sample_limit_per_split"] = sample_limit
            provenance["sample_limit_per_split"] = sample_limit
        finalize_bundle(
            output,
            recipe,
            dataset=source_report["dataset"],
            source={
                "repository": source_report["source_repository"],
                "revision": source_report["source_revision"],
            },
            protocol=protocol,
            event_feature_mappings=mappings,
        )
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
