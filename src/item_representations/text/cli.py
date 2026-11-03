"""Shared text-representation commands."""

import logging

from cyclopts import App

from item_representations.text.encode_catalog import encode_catalog

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
app = App()
app.command(encode_catalog)
