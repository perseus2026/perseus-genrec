"""Download the author Amazon ZIP once and publish only its TFRecords."""

import argparse
import json
import os
import shutil
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from filelock import FileLock
from tqdm import tqdm

from genrecsys.utils.files import file_sha256, publish_directory

FILE_ID = "1B5_q_MT3GYxmHLrMK0-lAqgpbAuikKEz"
SOURCE_URL = f"https://drive.google.com/file/d/{FILE_ID}/view"
CATEGORIES = ("beauty", "toys", "sports")
SPLITS = ("items", "training", "evaluation", "testing")


def counts(root):
    result = {}
    for category in CATEGORIES:
        result[category] = {}
        for split in SPLITS:
            files = list((root / category / split).rglob("*.tfrecord.gz"))
            if not files or any(
                not p.is_file() or p.stat().st_size == 0 for p in files
            ):
                raise ValueError(
                    f"Missing or empty TFRecords: {root / category / split}"
                )
            result[category][split] = len(files)
    return result


def prepare(root, archive=None):
    root = Path(root).resolve()
    archive = (
        Path(archive).resolve()
        if archive
        else root.parent / "downloads/amazon_data.zip"
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    # Separate category/experiment invocations may share the same archive.
    with FileLock(str(archive) + ".lock"):
        if root.exists():
            counts(root)
            return root
        if not archive.exists():
            try:
                import gdown
            except ImportError as error:
                raise RuntimeError(
                    "Install the optional group: poetry install --only amazon-download; "
                    "or supply an existing ZIP with --archive."
                ) from error
            downloaded = gdown.download(
                id=FILE_ID,
                output=str(archive),
                resume=True,
                quiet=False,
                use_cookies=False,
                verify=os.environ.get("REQUESTS_CA_BUNDLE") or True,
            )
            if downloaded is None:
                raise RuntimeError("Google Drive download did not complete")
        with zipfile.ZipFile(archive) as source:
            selected = []
            found = Counter()
            for entry in source.infolist():
                path = PurePosixPath(entry.filename)
                parts = path.parts
                if path.is_absolute() or ".." in parts or "\\" in entry.filename:
                    raise ValueError(f"Unsafe archive path: {entry.filename}")
                if entry.is_dir() or any(
                    p in ("__MACOSX", ".DS_Store") or p.startswith("._") for p in parts
                ):
                    continue
                if (
                    len(parts) >= 4
                    and parts[0] == "amazon_data"
                    and parts[1] in CATEGORIES
                    and parts[2] in SPLITS
                    and parts[-1].endswith(".tfrecord.gz")
                ):
                    selected.append(entry)
                    found[parts[1], parts[2]] += 1
            missing = [
                f"{c}/{s}" for c in CATEGORIES for s in SPLITS if not found[c, s]
            ]
            if missing:
                raise ValueError(f"Missing TFRecord sections: {missing}")
            with publish_directory(root, overwrite=False) as staging:
                for entry in tqdm(selected, desc="Extracting Amazon", unit="file"):
                    target = staging.joinpath(*PurePosixPath(entry.filename).parts[1:])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    # Reading each entry to EOF checks its ZIP CRC.
                    with source.open(entry) as reader, target.open("xb") as writer:
                        shutil.copyfileobj(reader, writer, length=1024 * 1024)
                    if target.stat().st_size != entry.file_size:
                        raise ValueError(f"Wrong extracted size: {entry.filename}")
                report = {
                    "source_url": SOURCE_URL,
                    "archive_sha256": file_sha256(archive),
                    "archive_bytes": archive.stat().st_size,
                    "files": counts(staging),
                }
                (staging / "download.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    print(prepare(args.raw_root, args.archive))
