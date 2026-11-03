"""Assemble T-ECD bases for the shared GenRecSys lifecycle."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from genrecsys.core import Config
from genrecsys.core.dataset_manifest import finalize_bundle
from genrecsys.core.event_feature_mapping import Spec
from genrecsys.core.resources import copy_resource
from genrecsys.utils.files import publish_directory
from genrecsys.utils.progress import logged_stage

ITEM_FEATURES = ("brand_id", "category", "subcategory", "price")


@logged_stage
def materialize_basis(
    *,
    config: Path,
    basis: Path,
    sid_checkpoint: Path,
    embedding_items: Path,
    embeddings: Path,
    workdir: Path,
    auxiliary_item_features: Path | None = None,
) -> None:
    """Copy the frozen comparison basis and bind published vectors to one SID mapper."""

    recipe = Config.load_data(config)
    item_features = pl.read_parquet(basis / "item_features.pq").select(
        "item_id", *ITEM_FEATURES
    )
    if auxiliary_item_features is not None:
        item_features = pl.concat(
            [
                item_features,
                pl.read_parquet(auxiliary_item_features).select(
                    "item_id", *ITEM_FEATURES
                ),
            ],
            how="vertical_relaxed",
        )
    if item_features["item_id"].null_count() or item_features[
        "item_id"
    ].n_unique() != len(item_features):
        raise ValueError("T-ECD item features require unique non-null item_id")
    embedding_catalog = pl.read_parquet(embedding_items, columns=["item_id"])
    if embedding_catalog["item_id"].n_unique() != len(embedding_catalog):
        raise ValueError("T-ECD embedding items must be unique")
    missing_features = embedding_catalog.join(
        item_features.select("item_id"), on="item_id", how="anti"
    )
    if len(missing_features):
        raise ValueError("T-ECD embeddings contain items without static features")
    source_report = json.loads((basis / "report.json").read_text())
    limitations = [
        "candidate catalog follows the public Perseus warm-item policy",
        "validation is split from the public 80% development pool",
        "missing T-ECD user attributes are retained and encoded as unknown ID 0",
        (
            "items without source embeddings use a zero-vector SID input "
            "during frozen apply"
        ),
    ]
    if source_report["source_revision"] == "main":
        limitations.append("source Hugging Face revision defaults to mutable main")
    protocol = {
        "name": source_report["protocol"],
        "source_revision": source_report["source_revision"],
        "test_target_digest": source_report["test_target_digest"],
        "candidate_policy": source_report["candidate_policy"],
        "holdout_targets_filtered": source_report["holdout_targets_filtered"],
        "history_gap_hours": source_report["history_gap_hours"],
        "sid_fit_scope": source_report["sid_fit_scope"],
        "item_catalog": source_report["item_catalog"],
        "embeddings": source_report.get("embeddings"),
        "limitations": limitations,
    }

    with publish_directory(workdir, overwrite=False) as output:
        resources = output / "resources"
        resources.mkdir()
        copy_resource(sid_checkpoint, resources / "sid_mapper.pkl")
        copy_resource(embedding_items, resources / "embedding_items.pq")
        copy_resource(embeddings, resources / "item_embeddings.npy")
        item_features.rename({"item_id": "item"}).write_parquet(
            resources / "item_features.pq"
        )
        resource_manifest = {
            "embedding_items": "embedding_items.pq",
            "item_embeddings": {
                "path": "item_embeddings.npy",
                "format": "npy",
            },
            "item_to_sid": {
                "path": "sid_mapper.pkl",
                "format": "sid_mapper",
                "items": "embedding_items",
                "embeddings": "item_embeddings",
                "batch_size": 8192,
            },
            "item_features": "item_features.pq",
        }
        (resources / "manifest.json").write_text(
            json.dumps(resource_manifest, indent=2) + "\n"
        )
        for split in ("train", "validation", "test"):
            source = basis / "basis" / split
            destination = output / "basis" / split
            (destination / "artifacts").mkdir(parents=True)
            copy_resource(source / "samples.pq", destination / "samples.pq")
            copy_resource(
                source / "artifacts" / "items.pq",
                destination / "artifacts" / "items.pq",
            )
        mappings = []
        for feature in ITEM_FEATURES:
            if (
                feature not in recipe.features
                or not recipe.features[feature].located_in.event
            ):
                continue
            events = tuple(
                event_name
                for event_name in recipe.events_for_feature(feature)
                if "item" in recipe.events[event_name].attributes
                and feature not in recipe.events[event_name].attributes
            )
            if events:
                mappings.append(
                    Spec(
                        feature=feature,
                        resource="item_features",
                        attributes=("item",),
                        value_column=feature,
                        events=events,
                    )
                )
        finalize_bundle(
            output,
            recipe,
            dataset=source_report["dataset"],
            source={
                "repository": source_report["source_repository"],
                "revision": source_report["source_revision"],
            },
            protocol=protocol,
            event_feature_mappings=tuple(mappings),
        )
        provenance = {
            "dataset": source_report["dataset"],
            "protocol": protocol["name"],
            **{key: value for key, value in protocol.items() if key != "name"},
        }
        (output / "provenance.json").write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
