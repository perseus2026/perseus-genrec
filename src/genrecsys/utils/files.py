"""Small filesystem primitives shared by artifact-producing stages."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


def discard_files(paths: list[Path], *, logger: logging.Logger) -> None:
    """Delete explicitly disposable local copies after successful consumption."""
    if not paths:
        return
    size = sum(path.stat().st_size for path in paths)
    for path in paths:
        path.unlink()
    logger.info(
        "Removed consumed local files=%s freed_bytes=%s first=%s",
        len(paths),
        size,
        paths[0],
    )


def file_sha256(path: str | Path, /, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file without materializing it in memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_provenance(repository: Path, paths: Iterable[Path]) -> dict:
    """Identify files explicitly selected by the caller, relative to a repository."""
    files = {
        str(path.relative_to(repository)): file_sha256(path) for path in sorted(paths)
    }
    return {
        "source": "repository",
        "sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
        "files": files,
    }


@contextmanager
def publish_directory(
    destination: str | Path,
    /,
    *,
    overwrite: bool,
) -> Generator[Path, None, None]:
    """Build a directory in staging and replace the destination on success."""

    destination = Path(destination)
    occupied = destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    )
    if occupied and not overwrite:
        raise FileExistsError(f"output is not empty: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{destination.name}-",
        dir=destination.parent,
    ) as temporary:
        staging = Path(temporary) / destination.name
        staging.mkdir()
        yield staging
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        staging.replace(destination)


__all__ = ["discard_files", "file_sha256", "publish_directory"]
