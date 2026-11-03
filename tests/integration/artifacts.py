"""Copy selected local artifacts to check that preparation can be relocated."""

import shutil


def copy_artifacts(source, destination, mask="**/*"):
    assert source.exists(), f"Missing artifact source: {source}"
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return
    paths = {
        path
        for pattern in mask.split(";")
        for path in source.glob(pattern)
        if path.is_file()
    }
    assert paths, f"Artifact mask {mask} selected no files in {source}"
    for path in paths:
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
