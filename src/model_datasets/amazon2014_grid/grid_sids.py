"""Import frozen author GRID codes without fitting or resolving collisions again."""

import json
import pickle
from pathlib import Path

import polars as pl
import torch

from genrecsys.core.encoders.semantic_id import Preprocessor
from genrecsys.utils.files import file_sha256, publish_directory
from model_datasets.amazon2014_grid.prepare import source_protocol

TOKEN_OFFSETS = (2, 258, 514, 770)


def import_grid_sids(
    *, sid_dir: Path, catalog: Path, output_dir: Path, source_report: Path | None = None
) -> None:
    """Convert trusted GRID Stage 03 [4, N] codes to item_id -> List[Int64].

    Column i denotes original numeric GRID item ID i. Preserve the fourth
    collision code exactly; offsets only reserve 0/1 and separate token levels.
    """
    tensor_path = sid_dir / "inference/pickle/merged_predictions_tensor.pt"
    rows_path = sid_dir / "inference/pickle/merged_predictions.pkl"
    validation_path = sid_dir / "runtime/sid-validation.json"
    validation = json.loads(validation_path.read_text())
    provenance = {}
    if source_report is not None:
        source = json.loads(source_report.read_text())
        provenance = {
            "category": source["category"],
            "source_report_sha256": file_sha256(source_report),
        }
        if (
            source["protocol"] != source_protocol(source["category"])
            or any(validation.get(k) != v for k, v in provenance.items())
            or validation.get("catalog_sha256") != file_sha256(catalog)
        ):
            raise ValueError("GRID SID category/source provenance mismatch")
    ids = pl.read_parquet(catalog, columns=["item_id"])["item_id"]
    count = len(ids)
    if (
        count == 0
        or ids.dtype != pl.String
        or ids.null_count()
        or ids.n_unique() != count
        or set(ids) != {str(i) for i in range(count)}
    ):
        raise ValueError("Expected the full contiguous original GRID item catalog")
    if (
        validation["items"] != count
        or validation["codebook_size"] != [256, 256, 256]
        or validation["sid_length"] != 4
        or validation["fit_scope"] != "full_author_catalog"
    ):
        raise ValueError("GRID SID validation report disagrees with the full catalog")
    for path in (tensor_path, rows_path):
        suffix = "/" + path.relative_to(sid_dir).as_posix()
        recorded = [
            digest
            for name, digest in validation["sha256"].items()
            if name.endswith(suffix)
        ]
        if recorded != [file_sha256(path)]:
            raise ValueError(f"GRID SID validation SHA-256 mismatch: {path.name}")
    sids = torch.load(tensor_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(sids, torch.Tensor)
        or sids.dtype != torch.int64
        or sids.shape != (4, count)
    ):
        raise ValueError("Expected author GRID SID int64 tensor [4, N]")
    codes = sids.T
    if (
        not ((codes[:, :3] >= 0) & (codes[:, :3] < 256)).all()
        or not (codes[:, 3] >= 0).all()
        or len(torch.unique(codes, dim=0)) != count
    ):
        raise ValueError("Invalid GRID codebook indices or non-unique full SID")
    # Trusted author output, checked against the Stage 03 report before unpickling.
    with rows_path.open("rb") as stream:
        rows = pickle.load(stream)
    item_ids = [row["item_id"] for row in rows]
    if len(item_ids) != count or set(item_ids) != set(range(count)):
        raise ValueError("GRID SID pickle must cover every item ID exactly once")
    if any(
        not torch.equal(torch.as_tensor(row["cluster_ids"]), codes[row["item_id"], :3])
        for row in rows
    ):
        raise ValueError("GRID SID pickle/PT item ID mismatch")
    mapping = pl.DataFrame(
        {
            "item_id": [str(i) for i in range(count)],
            "sid": (codes + torch.tensor(TOKEN_OFFSETS)).tolist(),
        },
        schema={"item_id": pl.String, "sid": pl.List(pl.Int64)},
    ).sort("item_id")
    Preprocessor(mapping, item_id_column="item_id")
    with publish_directory(output_dir, overwrite=False) as output:
        mapping_path = output / "item_to_sid.pq"
        mapping.write_parquet(mapping_path)
        report = {
            **provenance,
            "stage": "import_grid_sids",
            "sid_application": "frozen_grid_stage03_full_catalog_mapping",
            "items": count,
            "sid_width": 4,
            "token_offsets": list(TOKEN_OFFSETS),
            "collision_suffix": "preserved_from_author_tensor",
            "mapping_sha256": file_sha256(mapping_path),
            "grid_validation": validation,
            "inputs": {
                str(path): {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
                for path in (tensor_path, rows_path, validation_path, catalog)
            },
        }
        (output / "sid_import_report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
