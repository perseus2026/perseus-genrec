"""Prepare one local Flan-T5-XL snapshot, reused by all category runs."""

import argparse
import json
import shutil
from pathlib import Path

from filelock import FileLock
from huggingface_hub import snapshot_download

from genrecsys.utils.files import publish_directory

MODEL = "google/flan-t5-xl"
REVISION = "7d6315df2c2fb742f0f5b556879d730926ca9001"
PATTERNS = ["*.json", "*.safetensors", "spiece.model", "tokenizer.*"]


def check_snapshot(path):
    for name in ("config.json", "tokenizer_config.json", "spiece.model"):
        if not (path / name).is_file():
            raise ValueError(f"Incomplete model snapshot: {path / name}")
    index = path / "model.safetensors.index.json"
    shards = (
        set(json.loads(index.read_text())["weight_map"].values())
        if index.exists()
        else {"model.safetensors"}
    )
    # HF snapshots link to sibling blobs; validate index names, not link targets.
    if not shards or any(
        Path(name).is_absolute()
        or ".." in Path(name).parts
        or not (path / name).is_file()
        for name in shards
    ):
        raise ValueError(f"Missing model weights in {path}")


def prepare(output, revision=None):
    revision = revision or REVISION
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock"):
        if output.exists():
            record = output / "snapshot.json"
            if record.exists():
                metadata = json.loads(record.read_text())
                if metadata["model"] != MODEL or (
                    revision and revision != metadata["revision"]
                ):
                    raise ValueError(
                        "Existing snapshot does not match the requested model/revision"
                    )
                path = (output / metadata["revision"]).resolve()
                if not path.is_relative_to(output):
                    raise ValueError("Invalid snapshot path")
            else:
                if revision and output.name != revision:
                    raise ValueError(
                        "Use the matching cached snapshot or a new directory for this revision"
                    )
                path = output
            check_snapshot(path)
            return path
        source = Path(
            snapshot_download(repo_id=MODEL, revision=revision, allow_patterns=PATTERNS)
        )
        check_snapshot(source)
        with publish_directory(output, overwrite=False) as staging:
            # Dereference HF cache links; outputs can be transferred independently.
            shutil.copytree(source, staging / source.name, symlinks=False)
            (staging / "snapshot.json").write_text(
                json.dumps({"model": MODEL, "revision": source.name}, indent=2) + "\n"
            )
        return output / source.name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--revision",
        default=REVISION,
        help=f"Hugging Face commit (default: {REVISION})",
    )
    args = parser.parse_args()
    print(prepare(args.output, args.revision))
