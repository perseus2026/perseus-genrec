"""File commands for the shared sharded preparation API."""

from pathlib import Path

from cyclopts import App

from genrecsys.api import prepare_shards
from genrecsys.api.prepare_shards import (
    assemble,
    encode,
    fit,
    partition_histories,
    read,
)
from genrecsys.cli.preparation_inputs import Metadata, parse_mappings, parse_resources

app = App()
for command in (partition_histories, read, fit, encode, assemble):
    app.command(command)


@app.command
def plan(
    *,
    workdir: Path,
    output_dir: Path,
    config: Path | None = None,
    shards: int = 10,
    resource: tuple[str, ...] = (),
    event_feature_mapping: tuple[str, ...] | None = None,
    metadata: Metadata | None = None,
) -> None:
    """Plan history workers; resource options match train prepare-dataset.

    resource: Repeat NAME=PATH, NAME=JSON or NAME for a default path.
    event_feature_mapping: Repeat JSON objects describing exact event-feature joins.
    metadata: JSON object with dataset, source and protocol information.
    """
    prepare_shards.plan(
        workdir=workdir,
        output_dir=output_dir,
        config=config,
        shards=shards,
        resources=parse_resources(resource),
        event_feature_mappings=parse_mappings(event_feature_mapping),
        metadata=metadata,
    )
