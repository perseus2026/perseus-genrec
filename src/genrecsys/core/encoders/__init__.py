"""Feature encoder registry."""

from genrecsys.core.encoders import id as id_
from genrecsys.core.encoders import numeric, ple, semantic_id
from genrecsys.core.encoders.base import (
    Encoder,
    Observer,
    Preprocessor,
)

registry: dict[str, Encoder] = {
    "id": id_.encoder,
    "numeric": numeric.encoder,
    "ple": ple.encoder,
    "semantic_id": semantic_id.encoder,
}

__all__ = [
    "Encoder",
    "Observer",
    "Preprocessor",
    "registry",
]
