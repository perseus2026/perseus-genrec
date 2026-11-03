"""Module entrypoint for semantic-ID representation commands."""

import logging

from item_representations.semantic_ids.cli import app

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app()
