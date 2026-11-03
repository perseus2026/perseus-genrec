"""Bind a prepared GRID Amazon basis to a frozen item-to-SID mapping."""

import json
import logging
from pathlib import Path

import polars as pl

from genrecsys.core import Config, DatasetManifest
from genrecsys.core.encoders.semantic_id import Preprocessor
from genrecsys.core.resources import copy_resource
from genrecsys.utils.files import file_sha256, publish_directory
from model_datasets.amazon2014_grid.basis import SOURCE
from model_datasets.amazon2014_grid.prepare import event_name, source_protocol

logger = logging.getLogger(__name__)


def materialize_basis(
    *,
    config: Path,
    basis: Path,
    sid_mapping: Path,
    workdir: Path,
    sid_import_report: Path | None = None,
    user_bins: int | None = None,
) -> None:
    """Publish unchanged Stage 02 splits and the frozen full-catalog Stage 03 SID.

    EventHub remains an external Stage 02 artifact for shared preparation at
    Stage 08. This operation does not fit representations or change sampling.
    """
    recipe = Config.load_data(config)
    if user_bins is not None:
        if (
            user_bins < 1
            or "user" not in recipe.inputs
            or "user_bin" not in recipe.features
        ):
            raise ValueError(
                "user_bins requires a positive count and configured user context"
            )
        if recipe.inputs["user"].type != "context":
            raise ValueError("user input must be context")
        user_encoder = recipe.encoders[recipe.features["user_bin"].encoder]
        if user_encoder.type != "id" or user_encoder.preprocessor not in (
            {"artifact": "user_bin_categories"},
            {"categories": list(range(user_bins))},
        ):
            raise ValueError("user_bin requires the frozen vocabulary in bin order")
    report_path = basis / "report.json"
    report = json.loads(report_path.read_text())
    category = report.get("category")
    protocol, category_event = source_protocol(category), event_name(category)
    mode = report["augmentation"]["mode"]
    maximum = report["history"]["max_train_items"]
    if (
        report.get("source_protocol") != protocol
        or report.get("stage") != "02_basis_catalog_eventhub"
        or mode not in {"none", "prefix", "full"}
        or maximum < 1
        or report.get("protocol") != f"amazon2014grid-{category}-{mode}-h{maximum}-v1"
        or report["history"]["max_holdout_items"] != maximum
        or report["eventhub"]["event"] != category_event
        or report["eventhub"]["source"] != SOURCE
    ):
        raise ValueError("Expected the prepared GRID Amazon Stage 02 protocol")
    event = recipe.events.get(category_event)
    if (
        event is None
        or event.source != SOURCE
        or event.attributes.get("item") is None
        or event.attributes["item"].field != "item_id"
    ):
        raise ValueError("Config must bind the GRID EventHub item_id to item")
    history = recipe.inputs.get("history")
    if (
        history is None
        or history.events != (category_event,)
        or history.max_events_per_sequence != maximum
        or history.max_duration_per_sequence is not None
        or recipe.max_duration_per_sequence is not None
    ):
        raise ValueError(
            "Config history bounds must match Stage 02 without a time window"
        )
    encoder = recipe.encoders.get("item")
    if (
        encoder is None
        or encoder.type != "semantic_id"
        or encoder.preprocessor.get("artifact") != "item_to_sid"
        or encoder.preprocessor.get("item_id_column") != "item_id"
    ):
        raise ValueError("Config must use the frozen item_to_sid mapping")
    preprocessor = Preprocessor(pl.read_parquet(sid_mapping), item_id_column="item_id")
    tokens = preprocessor.mapping["sid"].explode()
    if (
        preprocessor.code_length != 4
        or not tokens.dtype.is_integer()
        or tokens.null_count()
        or tokens.min() < 2
    ):
        raise ValueError("Expected four integer SID tokens with 0/1 reserved")
    sid_ids = preprocessor.mapping["item_id"]
    candidate_path = basis / "catalog/inference/items.pq"
    candidate_ids = pl.read_parquet(candidate_path, columns=["item_id"])["item_id"]
    if (
        sid_ids.dtype != pl.String
        or candidate_ids.dtype != pl.String
        or candidate_ids.null_count()
        or candidate_ids.n_unique() != len(candidate_ids)
        or len(candidate_ids) != report["items"]
        or set(candidate_ids) != set(sid_ids)
    ):
        raise ValueError("SID mapping must cover exactly the full string-ID catalog")
    inputs = [config, report_path, sid_mapping, candidate_path]
    sid_application = "frozen_framework_stage03_full_catalog_mapping"
    if sid_import_report is not None:
        imported = json.loads(sid_import_report.read_text())
        if (
            imported.get("stage") != "import_grid_sids"
            or imported.get("sid_application")
            != "frozen_grid_stage03_full_catalog_mapping"
            or imported.get("mapping_sha256") != file_sha256(sid_mapping)
            or imported.get("items") != len(candidate_ids)
            or (
                category != "beauty"
                and (
                    imported.get("category") != category
                    or imported.get("source_report_sha256")
                    != report.get("source_report_sha256")
                )
            )
        ):
            raise ValueError("GRID SID import report does not match the frozen mapping")
        sid_application = imported["sid_application"]
        inputs.append(sid_import_report)
    with publish_directory(workdir, overwrite=False) as output:
        resources = output / "resources"
        copy_resource(sid_mapping, resources / "item_to_sid.pq")
        if sid_import_report is not None:
            copy_resource(sid_import_report, resources / "sid_import_report.json")
        resource_manifest = {"item_to_sid": "item_to_sid.pq"}
        if user_bins is not None:
            pl.DataFrame(
                {"value": range(user_bins), "index": range(1, user_bins + 1)},
                schema={"value": pl.Int64, "index": pl.Int64},
            ).write_parquet(resources / "user_bin_categories.pq")
            resource_manifest["user_bin_categories"] = "user_bin_categories.pq"
        (resources / "manifest.json").write_text(
            json.dumps(resource_manifest, indent=2) + "\n"
        )
        for split in ("train", "validation", "test"):
            source = basis / "basis" / split
            samples_path, items_path = (
                source / "samples.pq",
                source / "artifacts/items.pq",
            )
            inputs.extend((samples_path, items_path))
            items = pl.read_parquet(items_path, columns=["item_id"])["item_id"]
            if (
                items.dtype != pl.String
                or items.null_count()
                or items.n_unique() != len(items)
                or set(items) != set(candidate_ids)
            ):
                raise ValueError(f"{split} must preserve the full candidate catalog")
            samples = pl.read_parquet(samples_path)
            targets = samples["target"].explode()
            if (
                len(samples) != report["rows"][split]
                or targets.dtype != pl.String
                or targets.null_count()
                or not targets.is_in(items.implode()).all()
                or samples["target"].list.len().min() != 1
                or samples["target"].list.len().max() != 1
                or samples["history_length"].min() < 1
                or samples["history_length"].max() > maximum
            ):
                raise ValueError(
                    f"{split} targets, history or row counts disagree with Stage 02"
                )
            destination = output / "basis" / split
            if user_bins is None:
                copy_resource(samples_path, destination / "samples.pq")
            else:
                if (
                    "grid_user_id" not in samples.columns
                    or samples["grid_user_id"].null_count()
                ):
                    raise ValueError(
                        "User context requires original non-null grid_user_id"
                    )
                if "context" in samples.columns:
                    raise ValueError("Refusing to replace existing context")
                samples = samples.with_columns(
                    pl.struct(
                        (pl.col("grid_user_id") % user_bins).alias("user_bin")
                    ).alias("context")
                )
                destination.mkdir(parents=True, exist_ok=True)
                samples.write_parquet(destination / "samples.pq")
            copy_resource(items_path, destination / "artifacts/items.pq")
        copy_resource(report_path, output / "source_report.json")
        recipe.save(output / "config.yaml")
        manifest = DatasetManifest.build(output)
        manifest.validate(output, recipe)
        manifest.save(output / "manifest.json")
        provenance = {
            "stage": "07_materialize_basis",
            "protocol": report["protocol"],
            "source_protocol": protocol,
            "category": category,
            "sid_fit_diagnostics": report.get("sid_fit_diagnostics"),
            "augmentation": mode,
            "max_history": maximum,
            "candidate_policy": report["candidate_policy"],
            "quantizer_fit_scope": report["quantizer_fit_scope"],
            "sid_application": sid_application,
            "sid_width": preprocessor.code_length,
            "user_bins": user_bins,
            "candidate_items": len(candidate_ids),
            "history_source": "external_stage02_amazon2014grid_eventhub_at_stage08",
            "inputs": {
                str(path): {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
                for path in inputs
            },
        }
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    logger.info(
        "GRID %s bundle materialized: mode=%s history=%s items=%s; Stage 08 remains separate",
        category,
        mode,
        maximum,
        len(candidate_ids),
    )
