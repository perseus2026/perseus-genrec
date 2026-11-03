"""Module entrypoint for T-ECD model-dataset commands."""

import logging

from model_datasets.tecd.cli import app

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app()
