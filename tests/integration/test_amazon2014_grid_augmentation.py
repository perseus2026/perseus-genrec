"""Require an explicit training population without changing the three protocols."""

import json

import polars as pl
import pytest
from cyclopts.exceptions import MissingArgumentError
from test_amazon2014_grid_import import _source

from model_datasets.amazon2014_grid.basis import build_basis
from model_datasets.amazon2014_grid.cli import app


def test_augmentation_is_required_by_python_and_cli(tmp_path):
    output = tmp_path / "output"
    with pytest.raises(TypeError, match="augmentation"):
        build_basis(output_dir=output, source_dir=tmp_path / "missing")
    with pytest.raises(MissingArgumentError, match="augmentation"):
        app.parse_args(
            [
                "build-basis",
                f"--output-dir={output}",
                f"--source-dir={tmp_path / 'missing'}",
            ],
            exit_on_error=False,
        )
    assert not output.exists()


def test_explicit_modes_preserve_holdouts_and_noaug_population(tmp_path):
    source = _source(tmp_path / "raw")
    holdouts = None
    # Five users, 25 training items each: one target, 24 prefixes, 300 windows.
    for mode, train_rows in (("none", 5), ("prefix", 120), ("full", 1500)):
        output = tmp_path / mode
        command, bound, _ = app.parse_args(
            [
                "build-basis",
                f"--output-dir={output}",
                f"--source-dir={source}",
                f"--augmentation={mode}",
                "--max-history=20",
            ],
            exit_on_error=False,
        )
        command(*bound.args, **bound.kwargs)
        train = pl.read_parquet(output / "basis/train/samples.pq")
        assert len(train) == train_rows
        assert train["history_length"].min() >= 1
        assert train["history_length"].max() <= 20
        if mode == "none":
            assert train["grid_user_id"].n_unique() == len(train)
            assert train["target"].to_list() == [["24"]] * 5
            assert train["history_length"].to_list() == [20] * 5
        current = [
            pl.read_parquet(output / f"basis/{split}/samples.pq")
            for split in ("validation", "test")
        ]
        if holdouts is None:
            holdouts = current
        else:
            assert all(a.equals(b) for a, b in zip(holdouts, current, strict=True))
        for frame, target in zip(current, ("25", "26"), strict=True):
            assert len(frame) == 5
            assert frame["target"].to_list() == [[target]] * 5
            assert frame["history_length"].to_list() == [20] * 5
        assert (
            json.loads((output / "report.json").read_text())["augmentation"]["mode"]
            == mode
        )
