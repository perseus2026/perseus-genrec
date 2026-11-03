"""T-ECD preprocessing commands before the shared GenRecSys lifecycle."""

from cyclopts import App

from model_datasets.tecd.cross_history import (
    download_and_build_cross_history,
    download_and_build_payment_range,
)
from model_datasets.tecd.marketplace import (
    build_small_marketplace,
    download_and_build_small_marketplace,
    download_small_marketplace,
)
from model_datasets.tecd.materialize import materialize_basis
from model_datasets.tecd.retail import (
    archive_eventhub,
    build_small_retail,
    download_and_build_small_retail,
    download_small_retail,
    extract_eventhub,
)

app = App()
app.command(download_small_retail)
app.command(build_small_retail)
app.command(download_and_build_small_retail)
app.command(archive_eventhub)
app.command(extract_eventhub)
app.command(materialize_basis)
app.command(download_small_marketplace)
app.command(build_small_marketplace)
app.command(download_and_build_small_marketplace)
app.command(download_and_build_cross_history)
app.command(download_and_build_payment_range)

__all__ = [
    "app",
    "archive_eventhub",
    "build_small_marketplace",
    "build_small_retail",
    "download_and_build_cross_history",
    "download_and_build_payment_range",
    "download_and_build_small_marketplace",
    "download_and_build_small_retail",
    "download_small_marketplace",
    "download_small_retail",
    "extract_eventhub",
    "materialize_basis",
]
