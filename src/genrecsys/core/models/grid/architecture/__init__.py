"""GRID architecture construction, following the OneRec package layout."""

from .grid import GRID


def build_model(config: dict) -> GRID:
    """Build an encoder-decoder from architecture settings and a frozen SID vocabulary."""

    return GRID(config)


__all__ = ["GRID", "build_model"]
