"""Optional data diagnostics; never required to train or infer."""

from typing import Any

import polars as pl

from genrecsys import core


def preparation_statistics(
    config: core.Config,
    encoder_to_preprocessor: dict[str, core.encoders.Preprocessor],
    train_samples: pl.DataFrame,
    validation_samples: pl.DataFrame,
    test_samples: pl.DataFrame | None,
    train_artifacts: core.tasks.retrieval.Artifacts,
    validation_artifacts: core.tasks.retrieval.Artifacts,
    test_artifacts: core.tasks.retrieval.Artifacts | None,
    event_store_rows: int,
    event_store_clients: int,
    event_feature_count_parts: dict[str, list[pl.DataFrame]],
    train_event_feature_count_parts: dict[str, list[pl.DataFrame]],
) -> dict[str, Any]:
    """Build a compact, stable report for cross-run data-contract comparisons."""

    def sample_statistics(samples: pl.DataFrame) -> dict[str, int]:
        """Summarize split rows, users, targets, and empty target lists."""

        targets = samples["target"]
        lengths = targets.list.len().fill_null(0)
        exploded = targets.explode(empty_as_null=True).drop_nulls()
        return {
            "rows": len(samples),
            "users": samples["client_id"].n_unique(),
            "target_occurrences": int(lengths.sum()),
            "distinct_targets": exploded.n_unique(),
            "empty_target_rows": int((lengths == 0).sum()),
        }

    item_feature = config.task.objective.target
    item_encoder = config.features[item_feature].encoder
    item_preprocessor = encoder_to_preprocessor[item_encoder]
    mapping = getattr(item_preprocessor, "mapping", None)
    item_id_column = getattr(item_preprocessor, "item_id_column", None)
    sid_column = getattr(item_preprocessor, "sid_column", None)

    def artifact_statistics(
        artifacts: core.tasks.retrieval.Artifacts,
    ) -> dict[str, Any]:
        """Summarize catalog coverage, metadata, and SID-prefix bucket sizes."""

        items = getattr(artifacts, "items", None)
        if not isinstance(items, pl.DataFrame):
            return {}
        result: dict[str, Any] = {
            "rows": len(items),
            "distinct_items": items["item_id"].n_unique(),
        }
        if "created_at" in items.columns:
            result["missing_created_at"] = items["created_at"].null_count()
        if "generation_allowed" in items.columns:
            result["generation_allowed_rows"] = int(
                items["generation_allowed"].fill_null(False).sum()
            )
        if (
            isinstance(mapping, pl.DataFrame)
            and isinstance(item_id_column, str)
            and isinstance(sid_column, str)
        ):
            bucket_sizes = (
                items.select(item_id_column)
                .join(
                    mapping.select(item_id_column, sid_column),
                    on=item_id_column,
                    how="left",
                    validate="m:1",
                )
                .select(prefix=pl.col(sid_column).list.first())
                .drop_nulls()
                .group_by("prefix")
                .len()
                .get_column("len")
            )
            if len(bucket_sizes):
                mean = float(bucket_sizes.mean())
                std = float(bucket_sizes.std(ddof=0) or 0.0)
                result["sid_prefix_buckets"] = {
                    "used": len(bucket_sizes),
                    "mean": mean,
                    "std": std,
                    "cv": std / mean if mean else 0.0,
                    "min": int(bucket_sizes.min()),
                    "p50": float(bucket_sizes.quantile(0.50, interpolation="linear")),
                    "p90": float(bucket_sizes.quantile(0.90, interpolation="linear")),
                    "p95": float(bucket_sizes.quantile(0.95, interpolation="linear")),
                    "p99": float(bucket_sizes.quantile(0.99, interpolation="linear")),
                    "max": int(bucket_sizes.max()),
                }
        return result

    event_statistics: dict[str, Any] = {}
    feature_to_encoder = config.feature_to_encoder(event=True)
    for feature, encoder in feature_to_encoder.items():
        if feature == item_feature:
            continue
        if config.encoders[encoder].type != "id":
            continue
        preprocessor = encoder_to_preprocessor[encoder]
        values = getattr(preprocessor, "values", pl.DataFrame())
        if not isinstance(values, pl.DataFrame):
            continue
        vocabulary = (
            values.filter(pl.col("feature") == feature).drop("feature")
            if "feature" in values.columns
            else values
        )
        parts = train_event_feature_count_parts.get(feature, [])
        if parts:
            counts = (
                pl.concat(parts)
                .group_by("value")
                .agg(pl.col("count").sum())
                .with_columns(
                    pl.col("value").cast(vocabulary.schema["value"], strict=False)
                )
                .drop_nulls("value")
            )
            vocabulary = vocabulary.join(counts, on="value", how="left")
        else:
            vocabulary = vocabulary.with_columns(count=pl.lit(0, dtype=pl.UInt64))
        event_statistics[feature] = {
            "encoder": encoder,
            "vocab_size": len(vocabulary),
            "selected_train_occurrences": int(vocabulary["count"].fill_null(0).sum()),
            "tokens": vocabulary.sort("index")
            .with_columns(pl.col("count").fill_null(0))
            .to_dicts(),
        }

    event_store_feature_counts = {
        feature: (
            pl.concat(parts)
            .group_by(feature)
            .agg(pl.col("rows").sum(), pl.col("clients").sum())
            .sort("rows", descending=True)
            .to_dicts()
        )
        for feature, parts in event_feature_count_parts.items()
        if parts
    }

    splits: list[
        tuple[str, pl.DataFrame | None, core.tasks.retrieval.Artifacts | None]
    ] = [
        ("train", train_samples, train_artifacts),
        ("validation", validation_samples, validation_artifacts),
        ("test", test_samples, test_artifacts),
    ]
    return {
        "version": 1,
        "samples": {
            name: sample_statistics(samples)
            for name, samples, _ in splits
            if samples is not None
        },
        "catalogs": {
            name: artifact_statistics(artifacts)
            for name, _, artifacts in splits
            if artifacts is not None
        },
        "events": event_statistics,
        "event_store": {
            "rows": event_store_rows,
            "clients": event_store_clients,
            "feature_counts": event_store_feature_counts,
        },
        "history": {
            "max_duration_per_sequence": config.max_duration_per_sequence,
            "selection": "independent_per_pathway_before_sample_timestamp",
            "pathways": {
                name: {
                    "events": list(value.events),
                    "max_events_per_sequence": value.max_events_per_sequence,
                    "max_duration_per_sequence": value.max_duration_per_sequence,
                }
                for name, value in config.inputs.items()
                if value.type == "sequence"
            },
        },
    }
