"""Command-line entrypoint for Amazon GRID preparation."""

import logging

from .cli import app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
if __name__ == "__main__":
    app()
