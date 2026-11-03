"""Download the pinned author source and overlays, without installing packages."""

import argparse
import shutil
import tarfile
from pathlib import Path
from tempfile import TemporaryFile
from urllib.request import urlopen

from genrecsys.utils.files import publish_directory

REPO = Path(__file__).resolve().parents[2]
COMMIT = "2fe3475b2d369580234093f35d52b1a2f54d0472"


def patch_tiger_prefix_mask(destination):
    """Fix the pinned TIGER first-step mask; accept an already patched checkout."""
    path = Path(destination) / "src/models/modules/semantic_id/tiger_generation_model.py"
    before = (
        "                ).reshape(-1, self.num_embeddings_per_hierarchy)\n"
        '            candidate_logits[~valid_prefix_mask] = float("-inf")\n'
        "\n"
        "        candidate_logits = torch.nn.functional.softmax(candidate_logits, dim=-1)\n"
    )
    after = before.replace(
        "\n            candidate_logits[", "\n                candidate_logits["
    )
    source = path.read_text()
    if source.count(before) == 0 and source.count(after) == 1:
        return
    if source.count(before) != 1 or source.count(after) != 0:
        raise ValueError(f"Unexpected GRID TIGER prefix-mask code: {path}")
    # The first-step [vocab] mask already indexes columns above. The [batch*beam,
    # vocab] mask must be applied only in the subsequent-step else branch.
    path.write_text(source.replace(before, after, 1))


def install_overlays(destination):
    """Refresh repository-owned experiment configs in a pinned GRID checkout."""
    for folder, group in (
        ("upstream", ""),
        ("es1/upstream", ""),
        ("es2/upstream", "es2"),
    ):
        for source in (REPO / "configs/amazon2014-grid" / folder).glob("*.yaml"):
            target = Path(destination) / "configs/experiment" / group / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def restore(destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    url = f"https://codeload.github.com/snap-research/GRID/tar.gz/{COMMIT}"
    with (
        TemporaryFile() as downloaded,
        publish_directory(destination, overwrite=False) as staging,
    ):
        with urlopen(url, timeout=60) as response:
            shutil.copyfileobj(response, downloaded)
        downloaded.seek(0)
        with tarfile.open(fileobj=downloaded) as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                target = staging / Path(member.name).relative_to(f"GRID-{COMMIT}")
                if not target.resolve().is_relative_to(staging):
                    raise ValueError(f"Unsafe archive path: {member.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    shutil.copyfileobj(archive.extractfile(member), stream)
        patch_tiger_prefix_mask(staging)
        (staging / ".grid-commit").write_text(COMMIT + "\n")
        (staging / ".project-root").touch()
        install_overlays(staging)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    restore(parser.parse_args().destination)
