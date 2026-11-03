"""Common semantic checks for retrieval queries and candidate catalogs."""

from pathlib import Path

import polars as pl

from genrecsys.core.config import Config


def validate_samples(
    samples: pl.DataFrame | pl.LazyFrame,
    items: pl.DataFrame | pl.LazyFrame,
    *,
    split: str,
) -> None:
    """Reject ambiguous queries, invalid clocks, labels, and candidate identities.

    Holdout targets outside the candidate catalog are retained: their availability
    is a dataset protocol decision. Train targets must be in the train catalog.
    """
    samples = samples.lazy()
    items = items.lazy()
    schema, item_schema = samples.collect_schema(), items.collect_schema()
    if missing := {"client_id", "timestamp", "target"} - set(schema):
        raise ValueError(f"{split} samples miss columns: {sorted(missing)}")
    if "item_id" not in item_schema:
        raise ValueError(f"{split} candidate catalog misses item_id")
    target_type = schema["target"]
    if not isinstance(target_type, pl.List):
        raise TypeError(f"{split} target must be a list of item identities")
    target = pl.col("target")
    item_type = target_type.inner
    if isinstance(item_type, pl.Struct):
        fields = {field.name: field.dtype for field in item_type.fields}
        if not {"item", "relevance"} <= fields.keys():
            raise ValueError(f"{split} target structs require item and relevance")
        target = target.list.eval(pl.element().struct.field("item"))
        item_type = fields["item"]
    if item_type != pl.Null and item_type != item_schema["item_id"]:
        raise ValueError(f"{split} target and candidate item_id types differ")
    if not isinstance(schema["timestamp"], pl.Datetime):
        raise TypeError(f"{split} timestamp must be Datetime")
    identity = ["sample_id"] if "sample_id" in schema else ["client_id", "timestamp"]
    checks = {
        "null query identity": pl.any_horizontal(
            pl.col(name).is_null() for name in {*identity, "client_id", "timestamp"}
        ).any(),
        "duplicate query identity": pl.struct(identity).n_unique() != pl.len(),
        "null target identity": (
            target.is_null() | target.list.eval(pl.element().is_null()).list.any()
        ).any(),
    }
    if "history_timestamp" in schema:
        if not isinstance(schema["history_timestamp"], pl.Datetime):
            raise ValueError(f"{split} history_timestamp must be Datetime")
        checks["invalid history cutoff"] = (
            pl.col("history_timestamp").is_null()
            | (
                pl.col("history_timestamp").dt.replace_time_zone(None)
                > pl.col("timestamp").dt.replace_time_zone(None)
            )
        ).any()
    if "target_labels" in schema:
        if not isinstance(schema["target_labels"], pl.List):
            raise ValueError(f"{split} target_labels must be a list")
        checks["target and target_labels lengths differ"] = (
            pl.col("target_labels").is_not_null()
            & (pl.col("target_labels").list.len() != target.list.len())
        ).any()
    result = samples.select(**checks).collect().row(0, named=True)
    for error, invalid in result.items():
        if invalid:
            raise ValueError(f"{split}: {error}")
    invalid_items = (
        items.select(
            invalid=pl.col("item_id").is_null().any()
            | (pl.col("item_id").n_unique() != pl.len())
        )
        .collect()
        .item()
    )
    if invalid_items:
        raise ValueError(f"{split} candidate item_id must be unique and non-null")
    if split == "train":
        missing = (
            samples.select(target.explode().alias("item_id"))
            .drop_nulls()
            .unique()
            .join(items.select("item_id"), on="item_id", how="anti")
            .select(pl.len())
            .collect()
            .item()
        )
        if missing:
            raise ValueError(
                f"training targets are absent from the candidate catalog: {missing}"
            )


def validate_basis(workdir: Path, config: Config) -> None:
    """Validate the three split files before history reading or model training."""
    config.validate_data()
    for split in ("train", "validation", "test"):
        root = workdir / "basis" / split
        validate_samples(
            pl.scan_parquet(root / "samples.pq"),
            pl.scan_parquet(root / "artifacts/items.pq"),
            split=split,
        )
