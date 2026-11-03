"""Asset bootstrap on tiny local fixtures; never downloads or installs packages."""

import importlib.util
import io
import json
import runpy
import shutil
import subprocess
import sys
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def helper(name):
    spec = importlib.util.spec_from_file_location(
        name.replace("-", "_"), ROOT / f"scripts/amazon2014-grid/{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_zip(path, missing=None, extra=None):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for category in ("beauty", "toys", "sports"):
            for split in ("items", "training", "evaluation", "testing"):
                name = f"amazon_data/{category}/{split}/part.tfrecord.gz"
                if (category, split) != missing:
                    archive.writestr(name, f"synthetic-record-{category}-{split}")
        archive.writestr("amazon_data/beauty/.DS_Store", b"ignored")
        archive.writestr("__MACOSX/amazon_data/._.DS_Store", b"ignored")
        archive.writestr("amazon_data/toys/items/._part.tfrecord.gz", b"ignored")
        archive.writestr("amazon_data/notes.txt", b"ignored")
        if extra:
            archive.writestr(extra, b"untrusted")


def test_zip_publish_reuse_and_no_gdown_dependency(tmp_path, monkeypatch):
    api = helper("download-data")
    archive = tmp_path / "amazon_data.zip"
    make_zip(archive)
    # Installed/not installed gdown is irrelevant for local ZIPs and raw inputs.
    monkeypatch.setitem(sys.modules, "gdown", None)
    output = api.prepare(tmp_path / "raw", archive)
    report = json.loads((output / "download.json").read_text())
    assert report["archive_sha256"] == api.file_sha256(archive)
    assert all(n == 1 for c in report["files"].values() for n in c.values())
    assert len(list(output.rglob("*.tfrecord.gz"))) == 12
    assert {p.name for p in output.iterdir()} == {
        "beauty",
        "toys",
        "sports",
        "download.json",
    }
    assert not any("._" in p.name or p.name == ".DS_Store" for p in output.rglob("*"))
    before = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
    archive.unlink()
    assert api.prepare(output, archive) == output
    assert before == {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}


@pytest.mark.parametrize("failure", ["missing", "traversal", "crc", "duplicate"])
def test_bad_zip_does_not_publish_dataset(tmp_path, failure):
    api = helper("download-data")
    archive = tmp_path / "source.zip"
    make_zip(
        archive,
        missing=("sports", "testing") if failure == "missing" else None,
        extra="amazon_data/beauty/items/../../../../escape.tfrecord.gz"
        if failure == "traversal"
        else None,
    )
    if failure == "crc":
        archive.write_bytes(
            archive.read_bytes().replace(
                b"synthetic-record-beauty-items", b"corrupted-record-beauty-items", 1
            )
        )
    if failure == "duplicate":
        with zipfile.ZipFile(archive, "a") as source, pytest.warns(UserWarning):
            source.writestr("amazon_data/beauty/items/part.tfrecord.gz", b"duplicate")
    output = tmp_path / "raw"
    with pytest.raises((ValueError, zipfile.BadZipFile, FileExistsError)):
        api.prepare(output, archive)
    assert not output.exists()
    assert not (tmp_path / "escape.tfrecord.gz").exists()
    assert archive.exists()


def test_missing_optional_dependency_is_actionable(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "gdown", None)
    with pytest.raises(RuntimeError, match="poetry install --only amazon-download"):
        helper("download-data").prepare(tmp_path / "raw")
    assert not (tmp_path / "raw").exists()


def test_existing_incomplete_data_is_not_overwritten(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "keep").write_text("existing")
    with pytest.raises(ValueError, match="Missing or empty"):
        helper("download-data").prepare(raw)
    assert (raw / "keep").read_text() == "existing"


def tiny_snapshot(path):
    path.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json"):
        (path / name).write_text("{}")
    (path / "spiece.model").write_bytes(b"tokenizer")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "shard.safetensors"}})
    )
    (path / "shard.safetensors").write_bytes(b"synthetic weights")
    return path


def test_embedder_download_once_and_reuse_self_contained_snapshot(
    tmp_path, monkeypatch
):
    api = helper("download-embedder")
    source = tiny_snapshot(tmp_path / "hf-cache" / api.REVISION)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(source)

    monkeypatch.setattr(api, "snapshot_download", download)
    output = tmp_path / "model"
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(api.prepare, [output, output]))
    assert paths == [output / api.REVISION] * 2 and len(calls) == 1
    assert calls[0]["repo_id"] == "google/flan-t5-xl"
    assert calls[0]["revision"] == api.REVISION
    assert "*.safetensors" in calls[0]["allow_patterns"]
    assert (
        json.loads((output / "snapshot.json").read_text())["revision"] == api.REVISION
    )
    shutil.rmtree(source)
    assert api.prepare(output, api.REVISION) == output / api.REVISION
    assert api.prepare(output / api.REVISION) == output / api.REVISION
    with pytest.raises(ValueError, match="revision"):
        api.prepare(output, "another-commit")


@pytest.mark.parametrize("override", [None, "another-commit"])
def test_embedder_cli_downloads_requested_revision(
    tmp_path, monkeypatch, capsys, override
):
    import huggingface_hub

    api = helper("download-embedder")
    revision = override or api.REVISION
    source = tiny_snapshot(tmp_path / "cache" / revision)
    output = tmp_path / "model"

    def download(**kwargs):
        assert kwargs["revision"] == revision
        return str(source)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    args = ["download-embedder.py", str(output)]
    if override:
        args += ["--revision", override]
    monkeypatch.setattr(sys, "argv", args)
    runpy.run_path(
        str(ROOT / "scripts/amazon2014-grid/download-embedder.py"), run_name="__main__"
    )
    assert capsys.readouterr().out.strip() == str(output / revision)
    if override:
        with pytest.raises(ValueError, match="revision"):
            api.prepare(output)


def test_incomplete_embedder_is_not_accepted(tmp_path, monkeypatch):
    api = helper("download-embedder")
    source = tiny_snapshot(tmp_path / "hf" / api.REVISION)
    (source / "shard.safetensors").unlink()
    monkeypatch.setattr(api, "snapshot_download", lambda **_: source)
    with pytest.raises(ValueError, match="weights"):
        api.prepare(tmp_path / "new-model")
    assert not (tmp_path / "new-model").exists()
    with pytest.raises(ValueError, match="weights"):
        api.prepare(source)


def grid_setup_body():
    return (
        (ROOT / "scripts/amazon2014-grid/prepare-grid.sh")
        .read_text()
        .split("<<'PY_GRID'\n", 1)[1]
        .rsplit("\nPY_GRID", 1)[0]
    )


def run_grid_setup():
    # Execute only the repository-owned heredoc, with venv/pip mocked by the caller.
    exec(compile(grid_setup_body(), "prepare-grid.sh", "exec"), {})  # noqa: S102


def grid_sandbox(tmp_path, monkeypatch, mock_grid_download):
    # Execute the actual setup body; restore is real, venv/pip calls are recorded.
    restore = mock_grid_download(helper("restore-grid"))
    script = tmp_path / "scripts/amazon2014-grid/restore-grid.py"
    script.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/amazon2014-grid/restore-grid.py", script)
    shutil.copytree(ROOT / "configs/amazon2014-grid", tmp_path / "configs/amazon2014-grid")
    framework = tmp_path / ".venv"
    framework.mkdir()
    monkeypatch.chdir(tmp_path)
    grid = tmp_path / "GRID-main"
    python = tmp_path / "grid-venv/bin/python"
    calls = []

    def run(args, **kwargs):
        assert kwargs["check"]
        calls.append(args)
        if "scripts/amazon2014-grid/restore-grid.py" in args:
            restore.restore(grid)
        elif args[1:3] == ["-m", "venv"]:
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("placeholder; never executed")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["-", str(grid), str(python)])
    return grid, python, calls


def test_grid_bootstrap_reuses_environment_without_touching_framework(
    tmp_path, monkeypatch, mock_grid_download
):
    grid, python, calls = grid_sandbox(tmp_path, monkeypatch, mock_grid_download)
    run_grid_setup()
    assert len(calls) == 4  # restore, venv, install, check
    assert (grid / "configs/experiment/es2/rqvae_train.yaml").exists()
    assert "--index-url=https://pypi.org/simple" in calls[2]
    assert not (python.parent.parent / ".amazon-grid-setup-pending").exists()
    run_grid_setup()
    assert len(calls) == 4
    assert not list((tmp_path / ".venv").iterdir())


def test_grid_setup_refreshes_owned_overlays(tmp_path, monkeypatch, mock_grid_download):
    grid, _, calls = grid_sandbox(tmp_path, monkeypatch, mock_grid_download)
    run_grid_setup()
    config = grid / "configs/experiment/reproduce_tiger_train.yaml"
    config.unlink()
    run_grid_setup()
    assert config.read_bytes() == (
        ROOT / "configs/amazon2014-grid/es1/upstream/reproduce_tiger_train.yaml"
    ).read_bytes()
    assert len(calls) == 4


def test_grid_setup_retries_failed_install(tmp_path, monkeypatch, mock_grid_download):
    _grid, python, calls = grid_sandbox(tmp_path, monkeypatch, mock_grid_download)
    success = subprocess.run

    def fail_install(args, **kwargs):
        if "install" in args:
            raise subprocess.CalledProcessError(1, args)
        success(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fail_install)
    with pytest.raises(subprocess.CalledProcessError):
        run_grid_setup()
    assert (python.parent.parent / ".amazon-grid-setup-pending").exists()
    monkeypatch.setattr(subprocess, "run", success)
    run_grid_setup()
    assert not (python.parent.parent / ".amazon-grid-setup-pending").exists()
    assert sum("scripts/amazon2014-grid/restore-grid.py" in c for c in calls) == 1


def test_grid_setup_rejects_framework_environment(
    tmp_path, monkeypatch, mock_grid_download
):
    grid, _, calls = grid_sandbox(tmp_path, monkeypatch, mock_grid_download)
    monkeypatch.setattr(
        sys, "argv", ["-", str(grid), str(tmp_path / ".venv/bin/python")]
    )
    with pytest.raises(ValueError, match="separate"):
        run_grid_setup()
    assert not calls


@pytest.mark.parametrize("failure", ["download", "archive"])
def test_grid_restore_failure_does_not_publish_checkout(
    tmp_path, monkeypatch, mock_grid_download, failure
):
    api = mock_grid_download(helper("restore-grid"))
    destination = tmp_path / "grid"
    if failure == "download":

        def fail(*args, **kwargs):
            raise OSError("network unavailable")

        monkeypatch.setattr(api, "urlopen", fail)
        error = OSError
    else:
        monkeypatch.setattr(
            api, "urlopen", lambda *args, **kwargs: io.BytesIO(b"invalid archive")
        )
        error = tarfile.ReadError
    with pytest.raises(error):
        api.restore(destination)
    assert not destination.exists()


def test_grid_setup_rejects_other_revision(tmp_path, monkeypatch, mock_grid_download):
    grid, _, calls = grid_sandbox(tmp_path, monkeypatch, mock_grid_download)
    run_grid_setup()
    (grid / ".grid-commit").write_text("another revision")
    with pytest.raises(ValueError, match="revision differs"):
        run_grid_setup()
    assert len(calls) == 4


def test_grid_patch_masks_tokens_on_first_and_subsequent_steps(
    tmp_path, mock_grid_download
):
    import torch

    api = mock_grid_download(helper("restore-grid"))
    grid = api.restore(tmp_path / "grid")
    path = grid / "src/models/modules/semantic_id/tiger_generation_model.py"
    patched = path.read_bytes()
    api.patch_tiger_prefix_mask(grid)
    assert path.read_bytes() == patched
    model = runpy.run_path(str(path))["PrefixMask"]()
    model.should_check_prefix = True
    model.num_embeddings_per_hierarchy = 4
    model.top_k_for_generation = 2
    codes = torch.tensor([[0, 1], [0, 3], [2, 2]])
    model._check_valid_prefix = lambda prefixes: (
        prefixes[:, None, :] == codes[None, :, : prefixes.shape[1]]
    ).all(dim=-1).any(dim=-1)

    # Batch != vocabulary size exposes the erroneous first-axis indexing.
    actual = model.probabilities(torch.zeros(3, 4), None, 0, 3)
    torch.testing.assert_close(actual, torch.tensor([[0.5, 0, 0.5, 0]] * 3))
    generated = torch.tensor([[[0], [2]]] * 3)
    actual = model.probabilities(torch.zeros(6, 4), generated, 1, 3)
    expected = torch.tensor([[0, 0.5, 0, 0.5], [0, 0, 1, 0]] * 3)
    torch.testing.assert_close(actual, expected)
    model.should_check_prefix = False
    torch.testing.assert_close(
        model.probabilities(torch.zeros(3, 4), None, 0, 3),
        torch.full((3, 4), 0.25),
    )


def test_grid_setup_patches_existing_checkout_without_reinstalling(
    tmp_path, monkeypatch, mock_grid_download
):
    grid, _, calls = grid_sandbox(tmp_path, monkeypatch, mock_grid_download)
    run_grid_setup()
    path = grid / "src/models/modules/semantic_id/tiger_generation_model.py"
    patched = path.read_bytes()
    # Simulate a checkout created before the fix was added to bootstrap.
    shutil.copy2(ROOT / "tests/fixtures/grid/tiger_prefix_mask.py", path)
    assert path.read_bytes() != patched
    run_grid_setup()
    assert path.read_bytes() == patched
    assert len(calls) == 4
    run_grid_setup()
    assert path.read_bytes() == patched
    assert len(calls) == 4


def test_grid_patch_rejects_unexpected_source_without_overwriting(tmp_path):
    api = helper("restore-grid")
    path = tmp_path / "src/models/modules/semantic_id/tiger_generation_model.py"
    path.parent.mkdir(parents=True)
    path.write_text("# unexpected source\n")
    with pytest.raises(ValueError, match="Unexpected GRID TIGER"):
        api.patch_tiger_prefix_mask(tmp_path)
    assert path.read_text() == "# unexpected source\n"


def test_embedder_accepts_hf_cache_symlinks_and_exports_real_files(tmp_path, monkeypatch):
    api = helper("download-embedder")
    cache = tmp_path / "hf-cache"
    source = tiny_snapshot(cache / "snapshots" / api.REVISION)
    blobs = cache / "blobs"
    blobs.mkdir()
    for path in source.iterdir():
        blob = blobs / path.name
        path.rename(blob)
        path.symlink_to(blob)
    monkeypatch.setattr(api, "snapshot_download", lambda **_: str(source))
    output = tmp_path / "model"
    result = api.prepare(output)
    assert (result / "shard.safetensors").read_bytes() == b"synthetic weights"
    assert all(not path.is_symlink() for path in result.iterdir())
    shutil.rmtree(cache)
    assert api.prepare(output) == result


@pytest.mark.parametrize("absolute", [False, True])
def test_embedder_rejects_weight_index_paths_outside_snapshot(tmp_path, monkeypatch, absolute):
    api = helper("download-embedder")
    source = tiny_snapshot(tmp_path / "snapshot")
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"weights")
    name = str(outside) if absolute else "../outside.safetensors"
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": name}})
    )
    monkeypatch.setattr(api, "snapshot_download", lambda **_: str(source))
    with pytest.raises(ValueError, match="weights"):
        api.prepare(tmp_path / "output")
    assert not (tmp_path / "output").exists()
