"""Yambda commands before the shared training and inference lifecycle."""

from cyclopts import App

from model_datasets.yambda.basis import build_basis, download_source
from model_datasets.yambda.materialize import materialize_basis

app = App()
app.command(download_source)
app.command(build_basis)
app.command(materialize_basis)
