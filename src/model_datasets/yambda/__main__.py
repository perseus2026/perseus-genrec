"""Run the Yambda dataset commands."""

import logging

from model_datasets.yambda.cli import app

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    app()
