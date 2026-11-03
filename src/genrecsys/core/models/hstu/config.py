"""HSTU model requirements; preparation remains architecture independent."""

from genrecsys.core.checkpoint import Prepared
from genrecsys.core.config import Config
from genrecsys.core.encoders.semantic_id import Preprocessor
from genrecsys.core.models.config import _positive, validate_execution


def context_features(config: Config) -> tuple[str, ...]:
    """Use the already prepared user/context features to condition the readout."""
    return tuple(
        dict.fromkeys(
            feature
            for value in config.inputs.values()
            if value.type == "context"
            for feature in value.features
        )
    )


def validate(config: Config) -> None:
    """Require lifelong input and independent item targets on the native protocol."""
    params = config.model.params
    required = {
        "dim",
        "embedding_dim",
        "num_heads",
        "num_layers",
        "dropout",
        "catalog_chunk_size",
        "max_len",
        "num_time_buckets",
        "weight_initialization",
        "feature_fusion",
    }
    if set(params) != required:
        raise ValueError(f"HSTU model.params must contain exactly {sorted(required)}")
    if params["weight_initialization"] not in {"original", "perseus"}:
        raise ValueError("HSTU weight_initialization must be original or perseus")
    if params["feature_fusion"] not in {"sum", "concat_projection"}:
        raise ValueError("HSTU feature_fusion must be sum or concat_projection")
    for name in required - {"dropout", "weight_initialization", "feature_fusion"}:
        if not isinstance(params[name], int) or isinstance(params[name], bool):
            raise TypeError(f"HSTU model.params.{name} must be an integer")
        _positive(f"model.params.{name}", params[name])
    if params["dim"] % params["num_heads"]:
        raise ValueError("HSTU model dim must be divisible by num_heads")
    if params["max_len"] < 2:
        raise ValueError("HSTU max_len must include an event slot and the readout")
    if not 0 <= params["dropout"] < 1:
        raise ValueError("HSTU dropout must be in [0, 1)")
    objective = config.task.objective
    if (
        objective.type != "item_cross_entropy"
        or objective.target_mode != "all_targets_independent"
        or objective.loss.get("type") not in {"cross_entropy", "sampled_cross_entropy"}
        or objective.loss["reduction"]
        not in {
            "mean_targets_per_sample",
            "mean_targets_global",
        }
    ):
        raise ValueError("HSTU requires independent item_cross_entropy targets")
    expected_loss = {"type", "reduction"}
    if objective.loss["type"] == "sampled_cross_entropy":
        expected_loss.add("num_negatives")
        negatives = objective.loss.get("num_negatives")
        if not isinstance(negatives, int) or isinstance(negatives, bool):
            raise ValueError("sampled_cross_entropy requires integer num_negatives")
        _positive("objective.loss.num_negatives", negatives)
    if set(objective.loss) != expected_loss:
        raise ValueError(
            f"HSTU objective.loss must contain exactly {sorted(expected_loss)}"
        )
    if "lifelong" not in config.inputs:
        raise ValueError("HSTU requires inputs.lifelong")
    history = config.inputs["lifelong"]
    if history.type != "sequence" or history.compression is not None:
        raise ValueError("HSTU lifelong must be an uncompressed sequence")
    if objective.target not in history.features:
        raise ValueError("HSTU lifelong must include the target item feature")
    if params["feature_fusion"] == "concat_projection":
        event_encoder = config.event_encoders[history.event_encoder]
        if event_encoder.type != "concat_projection":
            raise ValueError("HSTU concat fusion requires concat_projection")
    target_encoder = config.features[objective.target].encoder
    if config.encoders[target_encoder].type != "semantic_id":
        raise ValueError(
            "HSTU consumes full item identities from prepared semantic_id data"
        )
    for name in (*history.features, *context_features(config)):
        encoder = config.encoders[config.features[name].encoder]
        if name != objective.target and encoder.type not in {"id", "numeric", "ple"}:
            raise ValueError(f"unsupported HSTU feature {name!r}")
        if encoder.embedder.get("artifact"):
            raise ValueError("HSTU currently trains its own feature embeddings")
    validate_execution(config)
    predict = config.inference.predict_kwargs
    if (
        set(predict) != {"k"}
        or not isinstance(predict["k"], int)
        or isinstance(predict["k"], bool)
    ):
        raise ValueError("HSTU inference.predict_kwargs must contain an integer k")
    _positive("inference.predict_kwargs.k", predict["k"])
    if predict["k"] < max(
        metric.params["k"] for metric in config.task.metrics.values()
    ):
        raise ValueError("HSTU inference k must cover all configured metrics")


def item_preprocessor(prepared: Prepared) -> Preprocessor:
    """Return the complete, bijective item/SID mapping, including collision levels."""
    encoder = prepared.config.features[prepared.config.task.objective.target].encoder
    item = prepared.encoder_to_preprocessor[encoder]
    if not isinstance(item, Preprocessor):
        raise TypeError("HSTU requires the frozen semantic_id item preprocessor")
    return item


def model_config(prepared: Prepared) -> dict:
    """Resolve lifelong and user feature widths from frozen preprocessing."""
    config = prepared.config
    params = config.model.params
    history = config.inputs["lifelong"]

    def describe(name):
        encoder = config.features[name].encoder
        preprocessor = prepared.encoder_to_preprocessor[encoder]
        kind = config.encoders[encoder].type
        description = {"name": name, "encoder": encoder, "type": kind}
        if name == config.task.objective.target:
            description.update(
                type="id", vocab_size=len(item_preprocessor(prepared).mapping) + 1
            )
        elif kind == "id":
            description["vocab_size"] = preprocessor.num_embeddings
        elif kind == "ple":
            description["thresholds"] = list(preprocessor.thresholds)
        if params["feature_fusion"] == "concat_projection":
            if name == config.task.objective.target:
                description["embedding_dim"] = params["embedding_dim"]
            elif "dim" in config.encoders[encoder].embedder:
                description["embedding_dim"] = config.encoders[encoder].embedder["dim"]
        elif kind in {"id", "semantic_id"}:
            description["embedding_dim"] = params["embedding_dim"]
        return description

    event_description = {}
    if params["feature_fusion"] == "concat_projection":
        event_encoder = config.event_encoders[history.event_encoder]
        event_description = {
            "event_encoder_type": event_encoder.type,
            "additive_features": event_encoder.params["additive_features"],
        }
    return {
        **params,
        **event_description,
        "target_encoder": config.features[config.task.objective.target].encoder,
        "features": [describe(name) for name in history.features],
        "context_features": [describe(name) for name in context_features(config)],
        "loss_type": config.task.objective.loss["type"],
        "num_negatives": config.task.objective.loss.get("num_negatives"),
        "reduction": config.task.objective.loss["reduction"],
        "gradient_checkpointing": config.training.gradient_checkpointing,
    }
