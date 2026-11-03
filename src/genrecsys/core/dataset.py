"""Prepared dataset state for the shared GenRecSys lifecycle."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import polars as pl

from genrecsys.core import tasks
from genrecsys.core.config import Config
from genrecsys.core.event_store import EventStore
from genrecsys.core.sample_table import MappedSamples


@dataclass
class Dataset:
    train_samples: pl.DataFrame | MappedSamples
    train_artifacts: tasks.retrieval.Artifacts
    validation_samples: pl.DataFrame | MappedSamples
    validation_artifacts: tasks.retrieval.Artifacts
    event_store: EventStore | None
    task_preprocessor: tasks.retrieval.Preprocessor
    test_samples: pl.DataFrame | None = None
    test_artifacts: tasks.retrieval.Artifacts | None = None
    statistics: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path, /) -> None:
        """Save prepared splits, artifacts, preprocessing state, and statistics.

        Move the EventStore into events/. Samples must be DataFrames; existing files
        at the same paths may be replaced.
        """

        if not isinstance(self.train_samples, pl.DataFrame) or not isinstance(
            self.validation_samples, pl.DataFrame
        ):
            raise TypeError("save the prepared DataFrames before memory-mapped loading")
        path.mkdir(parents=True, exist_ok=True)

        train_path = path / "train"
        train_path.mkdir(parents=True, exist_ok=True)
        self.train_samples.write_parquet(train_path / "samples.pq")
        self.train_artifacts.save(train_path / "artifacts")

        validation_path = path / "validation"
        validation_path.mkdir(parents=True, exist_ok=True)
        self.validation_samples.write_parquet(validation_path / "samples.pq")
        self.validation_artifacts.save(validation_path / "artifacts")

        if self.test_samples is not None:
            if self.test_artifacts is None:
                raise ValueError("test_samples require test_artifacts")
            test_path = path / "test"
            test_path.mkdir(parents=True, exist_ok=True)
            self.test_samples.write_parquet(test_path / "samples.pq")
            self.test_artifacts.save(test_path / "artifacts")

        if self.event_store is not None:
            self.event_store.move(path / "events")
        self.task_preprocessor.save(path / "preprocessor")
        if self.statistics:
            (path / "statistics.json").write_text(
                json.dumps(self.statistics, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    @classmethod
    def load(
        cls,
        path: Path,
        config: Config,
        /,
        *,
        include_test: bool = False,
        memory_map_samples: bool = False,
    ) -> Self:
        """Load prepared data without sharding samples between distributed ranks.

        Optionally memory-map sample files through an Arrow cache. Catalogs stay in RAM;
        load the test split only when include_test=True.
        """

        config.validate_data()
        artifacts_cls = tasks.retrieval.Artifacts
        train_path = path / "train"
        validation_path = path / "validation"
        test_path = path / "test"
        has_test = include_test and (test_path / "samples.pq").exists()
        # Sample ownership belongs exclusively to the Accelerate DataLoader.
        load_samples = (
            MappedSamples.from_parquet if memory_map_samples else pl.read_parquet
        )
        return cls(
            train_samples=load_samples(train_path / "samples.pq"),
            train_artifacts=artifacts_cls.load(train_path / "artifacts"),
            validation_samples=load_samples(validation_path / "samples.pq"),
            validation_artifacts=artifacts_cls.load(validation_path / "artifacts"),
            event_store=EventStore(path / "events", backend="disk"),
            task_preprocessor=tasks.retrieval.Preprocessor.load(path / "preprocessor"),
            test_samples=(load_samples(test_path / "samples.pq") if has_test else None),
            test_artifacts=(
                artifacts_cls.load(test_path / "artifacts") if has_test else None
            ),
            statistics=(
                json.loads((path / "statistics.json").read_text(encoding="utf-8"))
                if (path / "statistics.json").exists()
                else {}
            ),
        )
