"""Final T-ECD recipes keep a fixed width, including cold-user batches."""

from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
import torch

from genrecsys import core
from genrecsys.core.encoders.semantic_id import Preprocessor
from genrecsys.core.models.onerec.architecture.onerec import OneRec
from genrecsys.core.models.onerec.collate import onerec_collate_fn
from genrecsys.core.models.onerec.config import _model_config, model_parameters

ROOT = Path(__file__).resolve().parents[5]
RECIPES = [
    f"configs/tecd-small/onerec-{domain}-{setup}.yaml"
    for domain in ("retail", "marketplace")
    for setup in ("rq1", "cross-global")
]


@pytest.fixture
def single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _native_config(config):
    sid = Preprocessor(
        pl.DataFrame({"item_id": ["a", "b"], "sid": [[2, 5, 8], [3, 6, 9]]}),
        item_id_column="item_id",
    )
    preprocessors = {
        name: sid
        if encoder.type == "semantic_id"
        else SimpleNamespace(num_embeddings=8, thresholds=[0.0, 1.0])
        for name, encoder in config.encoders.items()
    }
    native = _model_config(
        SimpleNamespace(config=config, encoder_to_preprocessor=preprocessors)
    )
    # Keep production input caps and query counts; only shrink the CPU test model.
    native.update(
        d_model=16,
        d_ff=32,
        d_kv=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dropout_rate=0.0,
    )
    for description in native["inputs"]:
        if description["compression"]:
            description["compression"].update(num_layers=1, dim_feedforward=32)
    return native


def _row(inputs, history_length):
    row = {"user_id": history_length, "inputs": {}}
    for description in inputs:
        length = history_length if description["type"] == "sequence" else 1
        features = {}
        for feature in description["features"]:
            values = (
                [2, 5, 8][: feature["prefix_length"]]
                if feature["type"] == "semantic_id"
                else [0.5]
                if feature["type"] in {"numeric", "ple"}
                else [1]
            )
            features[feature["name"]] = {
                "values": [values] * length,
                "width": len(values),
                "dtype": "float32" if feature["type"] in {"numeric", "ple"} else "long",
            }
        row["inputs"][description["name"]] = {
            "padding_length": description["max_len"],
            "features": features,
        }
    return row


@pytest.mark.parametrize("recipe", RECIPES)
def test_tecd_encoder_always_receives_512_embeddings(recipe, single_torch_thread):
    config = core.Config.load(ROOT / recipe)
    assert model_parameters(config)["source_position_layout"] == "fixed"
    native = _native_config(config)
    model = OneRec(native).eval()
    assert model.max_encoder_positions == 512

    for history_lengths in ((0, 0), (0, 3)):
        batch = onerec_collate_fn(
            [_row(native["inputs"], length) for length in history_lengths]
        )
        with torch.no_grad():
            embeddings, mask = model.build_encoder_inputs(batch)
        assert embeddings.shape == (2, 512, 16)
        assert mask.shape == (2, 512)
        assert torch.isfinite(embeddings).all()
        assert mask.bool().any(dim=1).all()
