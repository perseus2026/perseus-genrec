"""Amazon GRID data adapter; training/inference use the shared genrecsys CLI."""

from cyclopts import App

from .basis import build_basis
from .grid_embeddings import export_grid_embeddings
from .grid_sids import import_grid_sids
from .materialize import materialize_basis
from .prepare import check_source_inputs, import_amazon, import_beauty

app = App()
for command in (
    import_amazon,
    import_beauty,
    check_source_inputs,
    build_basis,
    export_grid_embeddings,
    import_grid_sids,
    materialize_basis,
):
    app.command(command)
