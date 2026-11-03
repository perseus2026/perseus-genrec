"""Explicit encoder-decoder architecture settings for GRID and TIGER configurations.

Reference: https://arxiv.org/abs/2507.22224, Section 3.1.
Implementation details: snap-research/GRID, commit
2fe3475b2d369580234093f35d52b1a2f54d0472, SemanticIDEncoderDecoder.
Training settings remain in the shared training config, independently of the
architecture. This module does not select a dataset or fit semantic IDs.
"""

from typing import Any

from genrecsys.core.checkpoint import Prepared
from genrecsys.core.config import Config
from genrecsys.core.encoders.semantic_id import Preprocessor


def validate(config: Config) -> None:
    """Check the model recipe without reading histories or constructing GPU modules."""

    _validate_objective(config)
    _validate_model(config)
    _validate_inputs(config)
    _validate_execution(config)


def _validate_objective(config: Config) -> None:
    """Require SID cross-entropy with explicit token and independent-target averaging."""

    objective = config.task.objective
    if objective.target not in config.features:
        raise ValueError("GRID objective.target must reference a declared feature")
    if objective.type != "autoregressive_semantic_id":
        raise ValueError("GRID requires the autoregressive_semantic_id objective")
    if objective.target_mode not in {
        "all_targets_independent",
        "random",
        "label_weighted",
    }:
        raise ValueError("unsupported GRID target_mode")
    reductions = (
        {
            f"{tokens}_tokens_then_targets_{scope}"
            for tokens in ("mean", "sum")
            for scope in ("per_sample", "global")
        }
        if objective.target_mode == "all_targets_independent"
        else {"mean_tokens", "sum_tokens"}
    )
    if (
        set(objective.loss) != {"type", "reduction"}
        or objective.loss["type"] != "cross_entropy"
        or objective.loss["reduction"] not in reductions
    ):
        raise ValueError(
            f"GRID requires cross_entropy with reduction in {sorted(reductions)}"
        )


def model_parameters(config: Config) -> dict[str, Any]:
    """Flatten the explicit encoder/decoder sections, using the OneRec config layout."""

    params = dict(config.model.params)
    if forbidden := set(params) & {
        "num_encoder_layers",
        "num_decoder_layers",
        "user_id_buckets",
        "item_separator",
        "bos_token",
        "output_head",
    }:
        raise ValueError(
            "model.params contains unknown fields; use encoder/decoder sections: "
            f"{sorted(forbidden)}"
        )
    encoder = params.pop("encoder", None)
    decoder = params.pop("decoder", None)
    _exact_keys(
        encoder,
        {"num_layers", "user_id_buckets", "item_separator"},
        "model.params.encoder",
    )
    _exact_keys(
        decoder,
        {"num_layers", "bos_token", "output_head"},
        "model.params.decoder",
    )
    params.update(
        feed_forward_type=params.get(
            "feed_forward_type",
            "t5" if params.get("num_feedforward_hidden_layers") == 1 else "mlp",
        ),
        weight_initialization=params.get("weight_initialization", "default"),
        decoder_activation_checkpointing=params.get(
            "decoder_activation_checkpointing", True
        ),
        num_encoder_layers=encoder["num_layers"],
        num_decoder_layers=decoder["num_layers"],
        user_id_buckets=encoder["user_id_buckets"],
        item_separator=encoder["item_separator"],
        bos_token=decoder["bos_token"],
        output_head=decoder["output_head"],
    )
    return params


def _validate_model(config: Config) -> None:
    """Check explicit architecture choices without constructing modules or reading data.

    head_dim is independent of dim / num_heads: the papers use dim=128 and six
    attention heads of width 64. num_feedforward_hidden_layers counts hidden
    layers. feed_forward_type selects native T5 or GRID's biased ReLU MLP;
    omitted settings preserve native T5 at depth one and MLP at greater depths.
    Shared output heads predict the full token vocabulary; per_level heads predict
    each SID level separately. learned BOS and item separators are GRID options.
    decoder_activation_checkpointing controls independent-target decoder chunks,
    as in OneRec; omitted settings retain the previous checkpointed behavior.
    """

    params = model_parameters(config)
    required = {
        "history_input",
        "dim",
        "num_heads",
        "head_dim",
        "dim_feedforward",
        "num_feedforward_hidden_layers",
        "feed_forward_type",
        "feed_forward_proj",
        "dropout",
        "weight_initialization",
        "decoder_activation_checkpointing",
        "num_encoder_layers",
        "num_decoder_layers",
        "user_id_buckets",
        "item_separator",
        "bos_token",
        "output_head",
    }
    optional = {
        "target_chunk_size",
        "codebook_sizes",
        "collision_codebook_size",
        "training_augmentation",
        "training_recipe",
        "max_history_items",
        "user_id_column",
        "user_id_hash",
    } & params.keys()
    _exact_keys(params, required | optional, "model.params (resolved)")
    if "codebook_sizes" in params:
        _validate_codebook_sizes(params["codebook_sizes"])
    if "collision_codebook_size" in params:
        _positive_integer(
            "model.params.collision_codebook_size", params["collision_codebook_size"]
        )
    _positive_integer(
        "model.params.target_chunk_size", params.get("target_chunk_size", 128)
    )
    if not isinstance(params["decoder_activation_checkpointing"], bool):
        raise TypeError(
            "model.params.decoder_activation_checkpointing must be a boolean"
        )
    for name in (
        "dim",
        "num_heads",
        "head_dim",
        "dim_feedforward",
        "num_feedforward_hidden_layers",
        "num_encoder_layers",
        "num_decoder_layers",
    ):
        _positive_integer(f"model.params.{name}", params[name])
    buckets = params["user_id_buckets"]
    if not isinstance(buckets, int) or isinstance(buckets, bool) or buckets < 0:
        raise ValueError(
            "model.params.encoder.user_id_buckets must be a non-negative integer"
        )
    if "user_id_column" in params and (
        not isinstance(params["user_id_column"], str)
        or not params["user_id_column"].strip()
    ):
        raise ValueError("model.params.user_id_column must be a nonempty column name")
    if "user_id_hash" in params and (
        not isinstance(params["user_id_hash"], str)
        or params["user_id_hash"] not in {"sha256", "modulo"}
    ):
        raise ValueError("model.params.user_id_hash must be sha256 or modulo")
    if not isinstance(params["item_separator"], bool):
        raise TypeError("model.params.encoder.item_separator must be a boolean")
    dropout = params["dropout"]
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not 0 <= dropout < 1
    ):
        raise ValueError("model.params.dropout must be a number in [0, 1)")
    choices = {
        "feed_forward_type": {"t5", "mlp"},
        "feed_forward_proj": {"relu", "gated-gelu"},
        "weight_initialization": {"default", "grid", "xavier_uniform"},
        "bos_token": {"pad", "learned"},
        "output_head": {"shared", "per_level"},
    }
    for name, allowed in choices.items():
        if not isinstance(params[name], str) or params[name] not in allowed:
            raise ValueError(f"model.params.{name} must be one of {sorted(allowed)}")
    if params["feed_forward_type"] == "t5":
        if params["num_feedforward_hidden_layers"] != 1:
            raise ValueError("native T5 feed-forward blocks require one hidden layer")
    elif params["feed_forward_proj"] != "relu":
        raise ValueError("GRID MLP feed-forward blocks require relu")
    augmentation_parameters(config)
    training_parameters(config)
    if "max_history_items" in params:
        _positive_integer("model.params.max_history_items", params["max_history_items"])


def augmentation_parameters(config: Config) -> dict | None:
    """Opt into the upstream contiguous-window objective, independently of architecture.

    Each input row must contain one next-item target. Historical items followed by
    this target form the training sequence; held-out rows are never augmented.
    """
    value = config.model.params.get("training_augmentation")
    if value is None:
        return None
    _exact_keys(value, {"max_windows_per_batch"}, "model.params.training_augmentation")
    _positive_integer(
        "training_augmentation.max_windows_per_batch", value["max_windows_per_batch"]
    )
    if config.task.objective.target_mode != "all_targets_independent":
        raise ValueError("GRID augmentation requires all_targets_independent")
    if not config.task.objective.loss["reduction"].endswith("_per_sample"):
        raise ValueError("GRID augmentation requires per-sample loss reduction")
    return dict(value)


def training_parameters(config: Config) -> dict:
    """GRID-local loop controls absent from the shared epoch-based Training schema."""
    value = config.model.params.get("training_recipe", {})
    keys = {"max_steps", "validation_every_n_steps", "checkpoint_metric"}
    if not isinstance(value, dict) or set(value) - keys:
        raise ValueError(f"model.params.training_recipe accepts only {sorted(keys)}")
    for key in ("max_steps", "validation_every_n_steps"):
        if value.get(key) is not None:
            _positive_integer(f"training_recipe.{key}", value[key])
    metric = value.get("checkpoint_metric")
    if metric is not None and (
        not isinstance(metric, str) or metric not in config.task.metrics
    ):
        raise ValueError(
            "training_recipe.checkpoint_metric must reference task.metrics"
        )
    return {key: value.get(key) for key in sorted(keys)}


def _exact_keys(value: Any, keys: set[str], path: str) -> None:
    """Require an exact mapping key set and identify the config path on failure."""

    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{path} must contain exactly {sorted(keys)}")


def _positive_integer(name: str, value: Any) -> None:
    """Require a positive integer for architecture dimensions and layer counts."""

    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def history_input(config: Config) -> str:
    """Resolve the explicitly selected history without relying on input order."""

    name = config.model.params.get("history_input")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("model.params.history_input must name a sequence input")
    if name not in config.inputs:
        raise ValueError(
            f"model.params.history_input references unknown input {name!r}"
        )
    if config.input(name).type != "sequence":
        raise ValueError(
            f"model.params.history_input must reference a sequence: {name!r}"
        )
    return name


def generation_constraint(config: Config) -> str:
    """Read the same decoding mode for validation and final inference."""

    value = config.inference.predict_kwargs.get("constraint", "prefix")
    if value not in {"none", "prefix"}:
        raise ValueError("inference.predict_kwargs.constraint must be none or prefix")
    return value


def generation_renormalization(config: Config) -> bool:
    """Use the same optional constrained probability normalization in both phases."""

    value = config.inference.predict_kwargs.get("renormalize_logits", False)
    if not isinstance(value, bool):
        raise TypeError("inference.predict_kwargs.renormalize_logits must be a boolean")
    return value


def _validate_inputs(config: Config) -> None:
    """Validate the selected full-SID history; event types remain selection metadata."""

    name = history_input(config)
    value = config.input(name)
    target = config.task.objective.target
    if value.type != "sequence" or target not in value.features:
        raise ValueError("GRID history must be a sequence containing objective.target")
    # A OneRec prepared bundle can retain extra feature encoders, model ports and
    # projections. GRID consumes only the target SID and the selected port's data
    # selector. These model-side fields do not alter materialized EventStore bytes.
    encoder = config.encoders[config.features[target].encoder]
    if encoder.type != "semantic_id" or not encoder.preprocessor.get("artifact"):
        raise ValueError("GRID target requires a frozen semantic_id artifact")
    if (
        model_parameters(config)["output_head"] == "per_level"
        and "codebook_sizes" not in config.model.params
    ):
        raise ValueError("GRID per_level heads require model.params.codebook_sizes")
    if not value.events:
        raise ValueError("GRID history must declare its event set")
    _positive_integer(
        f"inputs.{name}.max_events_per_sequence", value.max_events_per_sequence
    )


def _validate_execution(config: Config) -> None:
    """Validate the shared OneRec training recipe and GRID decoding controls."""

    from genrecsys.core.config import DataLoader

    for section in ("training", "inference"):
        if getattr(config, section).mixed_precision not in {"no", "bf16"}:
            raise ValueError(f"{section}.mixed_precision must be no or bf16")
    training = config.training
    for field in (
        "gradient_accumulation_steps",
        "num_epochs",
        "log_every_n_train_steps",
        "validation_every_n_epochs",
    ):
        _positive_integer(f"training.{field}", getattr(training, field))
    if not isinstance(training.gradient_checkpointing, bool):
        raise TypeError("training.gradient_checkpointing must be a boolean")
    if training.gradient_clip_norm is not None:
        _positive("training.gradient_clip_norm", training.gradient_clip_norm)
    seed = training.dataloader.seed
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("training.dataloader.seed must be an integer in [0, 2**32)")
    for section, settings in (
        ("training.dataloader", training.dataloader),
        ("training.validation_dataloader", training.validation_dataloader),
        ("inference.dataloader", config.inference.dataloader),
    ):
        if not isinstance(settings, DataLoader):
            raise TypeError(f"{section} must be a DataLoader config")
        _positive_integer(f"{section}.batch_size", settings.batch_size)
        if (
            not isinstance(settings.num_workers, int)
            or isinstance(settings.num_workers, bool)
            or settings.num_workers < 0
        ):
            raise ValueError(f"{section}.num_workers must be a non-negative integer")
    if training.optimizer.type not in {"adam", "adamw"}:
        raise ValueError("GRID optimizer must be adam or adamw")
    if "lr" in training.optimizer.params:
        _positive("training.optimizer.params.lr", training.optimizer.params["lr"])
    scheduler = training.scheduler
    if scheduler.type == "constant":
        if scheduler.params:
            raise ValueError("constant scheduler takes no parameters")
    elif scheduler.type == "cosine_warmup":
        _exact_keys(
            scheduler.params,
            {"warmup_ratio", "min_lr_ratio"},
            "training.scheduler.params",
        )
        for name, value in scheduler.params.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= value <= 1
                or (name == "warmup_ratio" and value == 1)
            ):
                raise ValueError(f"scheduler {name} is outside its valid range")
    else:
        raise ValueError("GRID scheduler must be constant or cosine_warmup")
    if not training.tracker.type:
        raise ValueError("training.tracker.type must not be empty")
    early = training.early_stopping
    if early is not None:
        if early.metric not in config.task.metrics:
            raise ValueError("early-stopping metric must reference task.metrics")
        _positive_integer("training.early_stopping.patience", early.patience)
        if (
            isinstance(early.min_delta, bool)
            or not isinstance(early.min_delta, (int, float))
            or not 0 <= early.min_delta < float("inf")
        ):
            raise ValueError(
                "training.early_stopping.min_delta must be finite and non-negative"
            )
    predict = config.inference.predict_kwargs
    if set(predict) - {"k", "beam_size", "constraint", "renormalize_logits"} or not {
        "k",
        "beam_size",
    } <= set(predict):
        raise ValueError(
            "GRID prediction settings require k, beam_size and optional "
            "constraint, renormalize_logits"
        )
    for name in ("k", "beam_size"):
        _positive_integer(f"inference.predict_kwargs.{name}", predict[name])
    required_beam = max(
        predict["k"],
        *(int(metric.params["k"]) for metric in config.task.metrics.values()),
    )
    if predict["beam_size"] < required_beam:
        raise ValueError(f"inference beam_size must be >= {required_beam}")
    generation_constraint(config)
    generation_renormalization(config)
    # Persist explicit search semantics in config.yaml, including omitted defaults.
    predict.setdefault("constraint", "prefix")
    predict.setdefault("renormalize_logits", False)


def _positive(name: str, value: Any) -> None:
    """Require a finite positive numeric value without accepting booleans."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 < value < float("inf")
    ):
        raise ValueError(f"{name} must be finite and positive")


def _validate_codebook_sizes(sizes: Any) -> None:
    """Validate compact model dimensions without enumerating token IDs."""

    if not isinstance(sizes, (list, tuple)) or not sizes:
        raise ValueError("model.params.codebook_sizes must be a nonempty sequence")
    for size in sizes:
        _positive_integer("model.params.codebook_sizes entry", size)


def _sid_vocabulary(
    item: Preprocessor,
    codebook_sizes: list[int] | None = None,
    collision_codebook_size: int | None = None,
) -> list[list[int]]:
    """Reconstruct full namespaces in the model, preserving unused codebook tokens."""

    tokens = item.mapping[item.sid_column].explode()
    if (
        tokens.is_empty()
        or not tokens.dtype.is_integer()
        or tokens.null_count()
        or bool((tokens < 2).any())
        or item.mapping[item.sid_column].list.len().n_unique() != 1
    ):
        raise ValueError("frozen SIDs must contain complete integer tokens >= 2")
    observed = [
        sorted(set(item.mapping[item.sid_column].list.get(level).to_list()))
        for level in range(item.code_length)
    ]
    if codebook_sizes is None:
        if collision_codebook_size is not None:
            raise ValueError("collision_codebook_size requires declared codebook_sizes")
        return observed
    _validate_codebook_sizes(codebook_sizes)
    if item.code_length != len(codebook_sizes) + 1:
        raise ValueError("expected full semantic IDs including the collision level")
    levels, offset = [], 2
    for size in codebook_sizes:
        levels.append(list(range(offset, offset + size)))
        offset += size
    maximum = max(observed[-1])
    if maximum < offset:
        raise ValueError("collision level is outside the declared SID namespace")
    if collision_codebook_size is not None:
        _positive_integer("collision_codebook_size", collision_codebook_size)
        if maximum >= offset + collision_codebook_size:
            raise ValueError("collision codes exceed collision_codebook_size")
        maximum = offset + collision_codebook_size - 1
    levels.append(list(range(offset, maximum + 1)))
    for index, (declared, actual) in enumerate(zip(levels, observed, strict=True)):
        if not set(actual) <= set(declared):
            raise ValueError(f"SID level {index} differs from its codebook dimensions")
    return levels


def _model_config(checkpoint: Prepared) -> dict[str, Any]:
    """Resolve architecture dimensions and full frozen SID namespaces, without fitting codes."""

    from genrecsys.core.models.grid.catalog import _item_preprocessor

    config = checkpoint.config
    params = model_parameters(config)
    item = _item_preprocessor(checkpoint)
    # Always read complete frozen SIDs. OneRec's embedding prefix is model-side
    # and may intentionally be shorter in the compatible prepared config.
    levels = _sid_vocabulary(
        item, params.get("codebook_sizes"), params.get("collision_codebook_size")
    )
    reduction = config.task.objective.loss["reduction"]
    return {
        "history_input": history_input(config),
        "max_history_items": params.get("max_history_items"),
        "num_encoder_layers": params["num_encoder_layers"],
        "num_decoder_layers": params["num_decoder_layers"],
        "d_model": params["dim"],
        "d_ff": params["dim_feedforward"],
        "num_heads": params["num_heads"],
        "d_kv": params["head_dim"],
        "dropout_rate": params["dropout"],
        "vocab_size": max(item.vocab_size, max(max(level) for level in levels) + 1),
        "pad_token_id": 0,
        "eos_token_id": 1,
        "code_len": item.code_length,
        "allowed_tokens_by_level": [list(level) for level in levels],
        "weight_initialization": params["weight_initialization"],
        "feed_forward_proj": params["feed_forward_proj"],
        "num_feedforward_hidden_layers": params["num_feedforward_hidden_layers"],
        "feed_forward_type": params["feed_forward_type"],
        "user_id_buckets": params["user_id_buckets"],
        "item_separator": params["item_separator"],
        "bos_token": params["bos_token"],
        "output_head": params["output_head"],
        "token_loss_reduction": "sum" if reduction.startswith("sum_tokens") else "mean",
        # Target aggregation uses OneRec's internal names; token reduction is explicit above.
        "multi_target_loss_reduction": "mean_tokens_then_targets_global"
        if reduction.endswith("global")
        else "mean_tokens_then_targets_per_sample",
        "multi_target_decoder_chunk_size": params.get("target_chunk_size", 128),
        "decoder_activation_checkpointing": params["decoder_activation_checkpointing"],
        # Absent keys retain the original compiled descriptions/checkpoints.
        **{
            key: params[key]
            for key in ("user_id_column", "user_id_hash")
            if key in params
        },
    }
