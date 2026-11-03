"""Workdir defaults are opt-in per path; explicit CLI paths keep their old meaning."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from item_representations.semantic_ids import cli

DEFAULTS = {
    "fit-sid-mapper": {
        "items": "embeddings/train/items.pq",
        "embeddings": "embeddings/train/embeddings.npy",
        "output-dir": "sid-fit",
    },
    "apply-sid-mapper": {
        "checkpoint": "sid-fit/checkpoint.pkl",
        "items": "embeddings/inference/items.pq",
        "embeddings": "embeddings/inference/embeddings.npy",
        "output-dir": "resources/sid",
    },
}


@pytest.mark.parametrize("command", DEFAULTS)
@pytest.mark.parametrize("workdir", [None, "work/example"])
def test_workdir_defaults(command, workdir, monkeypatch):
    callback = Mock()
    monkeypatch.setattr(cli, command.replace("-", "_") + "_api", callback)
    tokens = [command] + ([] if workdir is None else [f"--workdir={workdir}"])
    function, bound, _ = cli.app.parse_args(tokens, exit_on_error=False)
    function(*bound.args, **bound.kwargs)
    assert callback.call_args.args == tuple(
        Path(workdir or "workdir") / path for path in DEFAULTS[command].values()
    )
    if command == "fit-sid-mapper":
        assert callback.call_args.kwargs["method"] == "rq_kmeans_full"
        assert callback.call_args.kwargs["codebook_size"] == (512, 512)
        assert callback.call_args.kwargs["max_iter"] == 100
        assert callback.call_args.kwargs["n_init"] == 3
        assert callback.call_args.kwargs["random_state"] == 42


@pytest.mark.parametrize("command", DEFAULTS)
@pytest.mark.parametrize("workdir", [None, "work/ignored"])
def test_old_explicit_paths_still_work(command, workdir, monkeypatch):
    callback = Mock()
    monkeypatch.setattr(cli, command.replace("-", "_") + "_api", callback)
    paths = {
        name: (Path("/external") if name == "embeddings" else Path("old")) / name
        for name in DEFAULTS[command]
    }
    tokens = [command, *(f"--{name}={path}" for name, path in paths.items())]
    if workdir:
        tokens.append(f"--workdir={workdir}")
    tokens += ["--eval-batch-size=12", "--overwrite"]
    function, bound, _ = cli.app.parse_args(tokens, exit_on_error=False)
    function(*bound.args, **bound.kwargs)
    assert callback.call_args.args == tuple(paths.values())
    assert callback.call_args.kwargs["eval_batch_size"] == 12
    assert callback.call_args.kwargs["overwrite"] is True


@pytest.mark.parametrize("command", DEFAULTS)
def test_partial_override_uses_workdir_only_for_missing_paths(command, monkeypatch):
    callback = Mock()
    monkeypatch.setattr(cli, command.replace("-", "_") + "_api", callback)
    function, bound, _ = cli.app.parse_args(
        [command, "--workdir=work/example", "--items=custom/items.pq"],
        exit_on_error=False,
    )
    function(*bound.args, **bound.kwargs)
    expected = {
        name: Path("custom/items.pq")
        if name == "items"
        else Path("work/example") / path
        for name, path in DEFAULTS[command].items()
    }
    assert callback.call_args.args == tuple(expected.values())
