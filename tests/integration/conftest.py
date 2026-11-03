"""Offline fixtures for scripts that download external GRID sources."""

import io
import os
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def mock_grid_download(monkeypatch):
    """Exercise restoration with a tiny archive or an explicitly cached upstream."""

    def configure(module, *, upstream=False):
        commit = module.COMMIT
        if upstream:
            path = Path(
                os.environ.get(
                    "GRID_TEST_ARCHIVE", ROOT / f"work/downloads/grid-{commit}.tar.gz"
                )
            )
            if not path.is_file():
                pytest.skip("Upstream composition requires a local GRID_TEST_ARCHIVE")
            data = path.read_bytes()
        else:
            files = {"README.md": b"Synthetic GRID fixture\n", "requirements.txt": b""}
            files["src/models/modules/semantic_id/tiger_generation_model.py"] = (
                ROOT / "tests/fixtures/grid/tiger_prefix_mask.py"
            ).read_bytes()
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
                for name, content in files.items():
                    info = tarfile.TarInfo(f"GRID-{commit}/{name}")
                    info.size = len(content)
                    archive.addfile(info, io.BytesIO(content))
            data = buffer.getvalue()

        def download(url, *, timeout):
            assert (
                url == f"https://codeload.github.com/snap-research/GRID/tar.gz/{commit}"
            )
            assert timeout > 0
            return io.BytesIO(data)

        monkeypatch.setattr(module, "urlopen", download)
        return module

    return configure
