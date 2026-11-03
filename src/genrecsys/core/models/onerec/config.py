"""Validation of OneRec SID retrieval recipes."""

from typing import Any

from genrecsys.core.checkpoint import Prepared
from genrecsys.core.config import Config, duration_days
from genrecsys.core.models.config import (
    _positive,
    validate_execution,
)


def validate(config: Config) -> None:
    """Check the model recipe without reading histories or constructing GPU modules."""
    _validate_objective(config)
    _validate_model(config)
    _validate_inputs(config)
    validate_execution(config)
    _validate_inference(config)


def _validate_objective(config: Config) -> None:
    """Validate autoregressive SID cross-entropy and its target reduction."""
    objective = config.task.objective
    if objective.target not in config.features:
        raise ValueError("OneRec objective.target must reference a declared feature")
    if objective.target_mode not in {
        "all_targets_independent",
        "random",
        "label_weighted",
    }:
        raise ValueError("unsupported OneRec target_mode")
    if objective.type != "autoregressive_semantic_id":
        raise ValueError("unsupported OneRec objective type")
    reductions = (
        {"mean_tokens_then_targets_per_sample", "mean_tokens_then_targets_global"}
        if objective.target_mode == "all_targets_independent"
        else {"mean_tokens"}
    )
    if (
        objective.loss.get("type") != "cross_entropy"
        or objective.loss.get("reduction") not in reductions
        or set(objective.loss) != {"type", "reduction"}
    ):
        raise ValueError(
            "OneRec objective.loss must use cross_entropy with reduction in "
            f"{sorted(reductions)}"
        )


def model_parameters(config: Config) -> dict[str, Any]:
    """Flatten the explicit encoder/decoder sections; old flat recipes are not supported."""
    params = dict(config.model.params)
    if forbidden := set(params) & {
        "num_encoder_layers",
        "num_decoder_layers",
        "position_embeddings",
        "source_position_layout",
        "generation_prefix_length",
    }:
        raise ValueError(
            f"model.params contains unknown fields; use encoder/decoder sections: {sorted(forbidden)}"
        )
    encoder, decoder = params.pop("encoder"), params.pop("decoder")
    if unknown := set(encoder) - {
        "num_layers",
        "position_embeddings",
        "source_position_layout",
    }:
        raise ValueError(
            f"model.params.encoder contains unknown fields: {sorted(unknown)}"
        )
    if unknown := set(decoder) - {
        "num_layers",
        "generation_prefix_length",
        "level_embeddings",
        "relative_attention",
    }:
        raise ValueError(
            f"model.params.decoder contains unknown fields: {sorted(unknown)}"
        )
    params.update(
        weight_initialization=params.get("weight_initialization", "default"),
        num_encoder_layers=encoder["num_layers"],
        num_decoder_layers=decoder["num_layers"],
        position_embeddings=encoder["position_embeddings"],
        source_position_layout=encoder.get("source_position_layout", "dynamic"),
        decoder_level_embeddings=decoder.get("level_embeddings", False),
        decoder_relative_attention=decoder.get("relative_attention", False),
    )
    if "generation_prefix_length" in decoder:
        params["generation_prefix_length"] = decoder["generation_prefix_length"]
    return params


def _validate_inputs(config: Config) -> None:
    """Check each input independently, allowing arbitrary names/counts and optional compression."""
    if not config.inputs:
        raise ValueError("OneRec requires at least one input")
    target_encoder = config.features[config.task.objective.target].encoder
    if config.encoders[target_encoder].type != "semantic_id":
        raise ValueError("OneRec target feature requires a semantic_id encoder")
    encoder_features = {}
    dim = config.model.params["dim"]
    used_event_encoders = set()
    for name, value in config.inputs.items():
        kinds = {
            feature: config.encoders[config.features[feature].encoder].type
            for feature in value.features
        }
        if set(kinds.values()) - {"semantic_id", "id", "numeric", "ple"}:
            raise ValueError(f"inputs.{name} has unsupported feature encoders")
        for feature, kind in kinds.items():
            encoder_name = config.features[feature].encoder
            if kind == "semantic_id":
                if encoder_name != target_encoder:
                    raise ValueError(
                        "SID inputs must share the target's frozen vocabulary"
                    )
                encoder = config.encoders[encoder_name]
                if not encoder.preprocessor.get("artifact") or set(
                    encoder.embedder
                ) != {"prefix_length"}:
                    raise ValueError(
                        "semantic_id requires a frozen artifact and prefix_length"
                    )
        if value.type == "context":
            if value.event_encoder is not None or value.compression is not None:
                raise ValueError(
                    f"context input {name!r} cannot use event_encoder/compression"
                )
            if value.aggregation is not None and (
                value.aggregation.type not in {"concat_projection", "identity"}
                or value.aggregation.params
            ):
                raise ValueError(
                    f"inputs.{name}.aggregation requires parameter-free concat_projection or identity"
                )
            if value.aggregation is not None and value.aggregation.type == "identity":
                if len(kinds) != 1 or set(kinds.values()) != {"id"}:
                    raise ValueError(
                        "identity context requires one categorical feature"
                    )
                feature = next(iter(kinds))
                embedder = config.encoders[config.features[feature].encoder].embedder
                if embedder.get("dim", dim) != dim:
                    raise ValueError(
                        "identity context embedding dim must equal model dim"
                    )
            continue
        if value.aggregation is not None:
            raise ValueError(f"sequence input {name!r} cannot use context aggregation")
        if value.event_encoder not in config.event_encoders:
            raise ValueError(
                f"inputs.{name}.event_encoder: unknown {value.event_encoder!r}"
            )
        used_event_encoders.add(value.event_encoder)
        event_encoder = config.event_encoders[value.event_encoder]
        if event_encoder.type not in {"identity", "concat_projection"}:
            raise ValueError(
                "OneRec event encoder must be identity or concat_projection"
            )
        layout = event_encoder.params.get("token_layout", "event")
        if layout not in {"event", "sid_sequence"}:
            raise ValueError("token_layout must be event or sid_sequence")
        if (
            layout == "event"
            and event_encoder.type == "identity"
            and (len(value.features) != 1 or "id" not in kinds.values())
        ):
            raise ValueError("identity event encoder requires one categorical feature")
        additive = event_encoder.params.get("additive_features", [])
        if any(
            kinds.get(feature) not in {"id", "ple", "numeric"} for feature in additive
        ) or not set(value.features) - set(additive):
            raise ValueError(
                "additive_features must be categorical or numeric, with at least one projected feature"
            )
        if layout == "sid_sequence":
            sid_features = [f for f, kind in kinds.items() if kind == "semantic_id"]
            if len(sid_features) != 1:
                raise ValueError("sid_sequence requires exactly one SID feature")
            if event_encoder.type == "identity" and set(value.features) - set(
                additive
            ) != set(sid_features):
                raise ValueError(
                    "identity sid_sequence requires all non-SID features to be additive"
                )
            for feature in additive:
                if (
                    kinds[feature] == "id"
                    and config.encoders[config.features[feature].encoder].embedder.get(
                        "dim", dim
                    )
                    != dim
                ):
                    raise ValueError(
                        "additive categorical embedding dim must equal model dim"
                    )
        previous = encoder_features.setdefault(value.event_encoder, value.features)
        if previous != value.features:
            raise ValueError(
                "inputs sharing an event_encoder must declare the same features"
            )
        if layout == "sid_sequence":
            optional_key = (value.event_encoder, "optional_features")
            optional = set(value.optional_features)
            if encoder_features.setdefault(optional_key, optional) != optional:
                raise ValueError(
                    "SID inputs sharing an event_encoder must declare the same optional features"
                )
        if value.compression is not None:
            compression = value.compression
            if compression.type != "qformer":
                raise ValueError(f"inputs.{name}.compression only supports qformer")
            required = {"num_queries", "num_layers", "num_heads", "dim_feedforward"}
            if set(compression.params) != required:
                raise ValueError(
                    f"inputs.{name}.compression.params must contain exactly {sorted(required)}"
                )
            for key in required:
                _positive(
                    f"inputs.{name}.compression.params.{key}", compression.params[key]
                )
            if dim % compression.params["num_heads"]:
                raise ValueError(f"inputs.{name} QFormer heads must divide model dim")
    if unused := set(config.event_encoders) - used_event_encoders:
        raise ValueError(f"event encoders are declared, but unused: {sorted(unused)}")


def _validate_model(config: Config) -> None:
    """Keep current backbones, two initialization recipes and one positional-option list."""
    params = model_parameters(config)
    required = {
        "dim",
        "num_encoder_layers",
        "num_decoder_layers",
        "num_heads",
        "dim_feedforward",
        "dropout",
        "weight_initialization",
        "position_embeddings",
        "source_position_layout",
        "decoder_level_embeddings",
        "decoder_relative_attention",
    }
    if missing := required - set(params):
        raise ValueError(f"model.params misses fields: {sorted(missing)}")
    if (
        unknown := set(params)
        - required
        - {
            "backbone",
            "generation_prefix_length",
            "target_chunk_size",
            "decoder_activation_checkpointing",
            "d_kv",
            "input_type_embeddings",
        }
    ):
        raise ValueError(f"model.params contains unknown fields: {sorted(unknown)}")
    for name in (
        "dim",
        "num_encoder_layers",
        "num_decoder_layers",
        "num_heads",
        "dim_feedforward",
    ):
        if not isinstance(params[name], int) or isinstance(params[name], bool):
            raise TypeError(f"model.params.{name} must be an integer")
        _positive(f"model.params.{name}", params[name])
    if "d_kv" in params:
        if not isinstance(params["d_kv"], int) or isinstance(params["d_kv"], bool):
            raise TypeError("model.params.d_kv must be an integer")
        _positive("model.params.d_kv", params["d_kv"])
    elif params["dim"] % params["num_heads"]:
        raise ValueError("model dim must be divisible by num_heads")
    if not 0 <= params["dropout"] < 1:
        raise ValueError("model.params.dropout must be in [0, 1)")
    if params["weight_initialization"] not in {"default", "xavier_uniform"}:
        raise ValueError("OneRec initialization must be xavier_uniform or default")
    if params.get("backbone", "t5") not in {"t5", "switch_dense"}:
        raise ValueError("OneRec backbone must be one of: switch_dense, t5")
    positions = params["position_embeddings"]
    if not isinstance(positions, list) or set(positions) - {
        "absolute",
        "relative",
        "daily_recency",
        "qformer_recency",
    }:
        raise ValueError(
            "position_embeddings must list absolute, relative, daily_recency and/or qformer_recency"
        )
    if not isinstance(params["decoder_relative_attention"], bool):
        raise TypeError("decoder.relative_attention must be a boolean")
    if not isinstance(params.get("input_type_embeddings", True), bool):
        raise TypeError("input_type_embeddings must be a boolean")
    if "daily_recency" in positions and config.max_duration_per_sequence is None:
        raise ValueError(
            "daily_recency requires max_duration_per_sequence for a bounded vocabulary"
        )
    if params["source_position_layout"] not in {"dynamic", "fixed"}:
        raise ValueError("source_position_layout must be dynamic or fixed")
    if not isinstance(params["decoder_level_embeddings"], bool):
        raise TypeError("decoder.level_embeddings must be a boolean")
    for name in ("generation_prefix_length", "target_chunk_size"):
        if name in params:
            _positive(f"model.params.{name}", params[name])


def _model_config(checkpoint: Prepared) -> dict[str, Any]:
    """Resolve frozen dimensions and assemble every input in config declaration order."""
    from genrecsys.core.models.onerec.catalog import _item_preprocessor

    config = checkpoint.config
    params = model_parameters(config)
    item = _item_preprocessor(checkpoint)
    code_len = int(params.get("generation_prefix_length", item.code_length))
    if not 1 <= code_len <= item.code_length:
        raise ValueError("generation_prefix_length exceeds the frozen SID depth")
    inputs = []
    for name, value in config.inputs.items():
        event_encoder = (
            config.event_encoders[value.event_encoder]
            if value.type == "sequence"
            else None
        )
        features = []
        for feature_name in value.features:
            encoder_name = config.features[feature_name].encoder
            encoder = config.encoders[encoder_name]
            preprocessor = checkpoint.encoder_to_preprocessor[encoder_name]
            feature = {
                "name": feature_name,
                "type": encoder.type,
                "encoder": encoder_name,
            }
            if encoder.type == "id":
                feature["vocab_size"] = preprocessor.num_embeddings
                if encoder.embedder.get("artifact"):
                    feature["embedding_dim"] = encoder.embedder["dim"]
                    feature["frozen_embedding"] = encoder.embedder["freeze"]
            elif encoder.type == "semantic_id":
                feature["prefix_length"] = encoder.embedder["prefix_length"]
                if feature_name in value.optional_features:
                    feature["optional"] = True
                if not 1 <= feature["prefix_length"] <= item.code_length:
                    raise ValueError(
                        f"inputs.{name}: SID prefix_length exceeds the frozen SID depth"
                    )
            elif encoder.type == "ple":
                feature["thresholds"] = list(preprocessor.thresholds)
            if encoder.type in {"id", "ple"} and "dim" in encoder.embedder:
                feature["embedding_dim"] = int(encoder.embedder["dim"])
            features.append(feature)
        inputs.append(
            {
                "name": name,
                "type": value.type,
                "features": features,
                "event_encoder": value.event_encoder
                if event_encoder
                else f"context:{name}",
                "event_encoder_type": event_encoder.type
                if event_encoder
                else value.aggregation.type
                if value.aggregation
                else "concat_projection",
                "additive_features": event_encoder.params.get("additive_features", [])
                if event_encoder
                else [],
                "max_len": int(value.max_events_per_sequence)
                if value.type == "sequence"
                else 1,
                "compression": dict(value.compression.params)
                if value.compression
                else None,
            }
        )
        if event_encoder and "token_layout" in event_encoder.params:
            inputs[-1]["token_layout"] = event_encoder.params["token_layout"]
    vocabulary_depth = max(
        [code_len]
        + [
            feature["prefix_length"]
            for value in inputs
            for feature in value["features"]
            if feature["type"] == "semantic_id"
        ]
    )
    return {
        "num_encoder_layers": params["num_encoder_layers"],
        "num_decoder_layers": params["num_decoder_layers"],
        "d_model": params["dim"],
        "d_ff": params["dim_feedforward"],
        "num_heads": params["num_heads"],
        "d_kv": params.get("d_kv", params["dim"] // params["num_heads"]),
        "dropout_rate": params["dropout"],
        "vocab_size": int(
            item.mapping[item.sid_column]
            .list.slice(0, vocabulary_depth)
            .list.max()
            .max()
        )
        + 1,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "code_len": code_len,
        "inputs": inputs,
        "position_embeddings": params["position_embeddings"],
        "decoder_level_embeddings": params["decoder_level_embeddings"],
        "decoder_relative_attention": params["decoder_relative_attention"],
        "input_type_embeddings": params.get("input_type_embeddings", True),
        **(
            {
                "daily_recency_vocab_size": duration_days(
                    config.max_duration_per_sequence,
                    field_name="max_duration_per_sequence",
                )
                + 2
            }
            if "daily_recency" in params["position_embeddings"]
            else {}
        ),
        "weight_initialization": params["weight_initialization"],
        "multi_target_decoder_chunk_size": params.get("target_chunk_size", 128),
        "decoder_activation_checkpointing": params.get(
            "decoder_activation_checkpointing", True
        ),
        "multi_target_loss_reduction": (
            config.task.objective.loss["reduction"]
            if config.task.objective.target_mode == "all_targets_independent"
            else "mean_tokens_then_targets_per_sample"
        ),
        "backbone": params.get("backbone", "t5"),
    }


def _validate_inference(config: Config) -> None:
    """Require a sufficiently wide autoregressive beam for all metrics."""
    inference = config.inference
    if set(inference.predict_kwargs) != {"k", "beam_size"}:
        raise ValueError("inference.predict_kwargs must contain k and beam_size")
    for name in ("k", "beam_size"):
        value = inference.predict_kwargs[name]
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"inference.predict_kwargs.{name} must be an integer")
        _positive(f"inference.predict_kwargs.{name}", value)
    required_beam = max(
        [
            inference.predict_kwargs["k"],
            *(int(metric.params["k"]) for metric in config.task.metrics.values()),
        ]
    )
    if inference.predict_kwargs["beam_size"] < required_beam:
        raise ValueError(f"inference beam_size must be >= {required_beam}")
