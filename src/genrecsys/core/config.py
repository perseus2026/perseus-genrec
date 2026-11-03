"""Typed configuration for the GenRecSys lifecycle."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from omegaconf import MISSING, ListConfig, OmegaConf

logger = logging.getLogger(__name__)
_DAY_DURATION = re.compile(r"([1-9][0-9]*)d")
_SUPPORTED_METRICS = {
    "coverage_at_k",
    "hit_rate_at_k",
    "map_at_k",
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
}
__all__ = [
    "Aggregation",
    "Attribute",
    "Baseline",
    "Compression",
    "Config",
    "DataLoader",
    "EarlyStopping",
    "Encoder",
    "Event",
    "EventEncoder",
    "Feature",
    "Inference",
    "Input",
    "LocatedIn",
    "Metric",
    "Model",
    "Objective",
    "Optimizer",
    "Scheduler",
    "Task",
    "Tracker",
    "TrainDataLoader",
    "Training",
    "Transform",
    "duration_days",
    "materialization_histories",
]


@dataclass
class Baseline:
    type: str = MISSING
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Metric:
    type: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Objective:
    type: str = "autoregressive_semantic_id"
    target: str = "item"
    target_mode: str = "all_targets_independent"
    loss: dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    type: str = MISSING
    baseline: Baseline | None = None
    metrics: dict[str, Metric | None] = field(default_factory=dict)
    objective: Objective = field(default_factory=Objective)


@dataclass
class Attribute:
    field: str | None = None


@dataclass
class Event:
    attributes: dict[str, Attribute | None] = field(default_factory=dict)
    source: str = "event_hub"
    priority: int = 0


@dataclass
class Encoder:
    type: str = MISSING
    preprocessor: dict[str, Any] = field(default_factory=dict)
    embedder: dict[str, Any] = field(default_factory=dict)


@dataclass
class LocatedIn:
    event: bool = False
    context: bool = False
    artifacts: bool = False


@dataclass
class Transform:
    type: str = MISSING
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Feature:
    located_in: LocatedIn = field(default_factory=LocatedIn)
    encoder: str = MISSING
    transform: Transform | None = None


@dataclass
class Compression:
    type: str = MISSING
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventEncoder:
    """Combine per-feature embeddings into event or SID-coordinate tokens."""

    type: str = MISSING
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Aggregation:
    """Combine context feature embeddings into model input tokens."""

    type: str = MISSING
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Input:
    """One model input; omitted compression means no sequence compression."""

    type: str = MISSING
    events: tuple[str, ...] = ()
    features: tuple[str, ...] = ()
    optional_features: tuple[str, ...] = ()
    max_events_per_sequence: int | None = None
    selection: str = "recent"
    max_priority_events_per_sequence: int | None = None
    max_duration_per_sequence: str | None = None
    event_encoder: str | None = None
    aggregation: Aggregation | None = None
    compression: Compression | None = None


@dataclass
class Model:
    type: str = MISSING
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Tracker:
    """Metric tracker declaration."""

    type: str = "logging"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataLoader:
    """Common train/inference DataLoader settings."""

    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = False


@dataclass
class TrainDataLoader(DataLoader):
    shuffle: bool = False
    seed: int = 0


@dataclass
class Optimizer:
    type: str = "adamw"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scheduler:
    type: str = "constant"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EarlyStopping:
    metric: str = MISSING
    patience: int = 1
    min_delta: float = 0.0


@dataclass
class Training:
    """Shared model-fitting settings."""

    num_epochs: int = 1
    log_every_n_train_steps: int = 10_000
    validation_every_n_epochs: int = 1
    tracker: Tracker = field(default_factory=Tracker)
    dataloader: TrainDataLoader = field(default_factory=TrainDataLoader)
    validation_dataloader: DataLoader = MISSING
    optimizer: Optimizer = field(default_factory=Optimizer)
    scheduler: Scheduler = field(default_factory=Scheduler)
    early_stopping: EarlyStopping | None = None
    gradient_clip_norm: float | None = None
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    mixed_precision: str = "no"


@dataclass
class Inference:
    """Shared prediction settings."""

    event_store_backend: str = "ram"
    mixed_precision: str = "no"
    predict_kwargs: dict[str, Any] = field(default_factory=dict)
    dataloader: DataLoader = field(default_factory=DataLoader)


@dataclass
class Config:
    """Shared lifecycle contract with model-specific validation by plugin."""

    task: Task = field(default_factory=Task)
    events: dict[str, Event] = field(default_factory=dict)
    max_duration_per_sequence: str | None = None
    encoders: dict[str, Encoder] = field(default_factory=dict)
    features: dict[str, Feature] = field(default_factory=dict)
    event_encoders: dict[str, EventEncoder] = field(default_factory=dict)
    inputs: dict[str, Input] = field(default_factory=dict)
    model: Model = field(default_factory=Model)
    training: Training = field(default_factory=Training)
    inference: Inference = field(default_factory=Inference)

    @classmethod
    def load(
        cls,
        path: str | Path,
        /,
    ) -> Self:
        """Load YAML overlays and validate the data and selected model configuration."""

        return cls._load(path, model=True)

    @classmethod
    def load_data(
        cls,
        path: str | Path,
        /,
    ) -> Self:
        """Load a recipe without making dataset stages depend on a model plugin."""

        return cls._load(path, model=False)

    @classmethod
    def _load(
        cls,
        path: str | Path,
        /,
        *,
        model: bool,
    ) -> Self:
        """Merge schema defaults with YAML, resolve interpolations, and validate."""

        schema = OmegaConf.structured(cls)
        loaded = _load_recipe(Path(path), seen=set())
        merged = OmegaConf.merge(schema, loaded)
        OmegaConf.resolve(merged)
        config = OmegaConf.to_object(merged)
        if not isinstance(config, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(config).__name__}")
        config.validate() if model else config.validate_data()
        logger.info(
            "loaded config:\n%s",
            OmegaConf.to_yaml(merged, resolve=True, sort_keys=False),
        )
        return config

    def save(self, path: str | Path, /) -> None:
        """Write the effective configuration to YAML."""

        OmegaConf.save(OmegaConf.structured(self), Path(path))

    def feature_to_encoder(
        self,
        *,
        event: bool = False,
        context: bool = False,
        artifacts: bool = False,
    ) -> dict[str, str]:
        """Return the feature/encoder view used by the shared lifecycle."""

        if not any((event, context, artifacts)):
            raise ValueError("must select at least one of event, context or artifacts")
        return {
            name: feature.encoder
            for name, feature in self.features.items()
            if (event and feature.located_in.event)
            or (context and feature.located_in.context)
            or (artifacts and feature.located_in.artifacts)
        }

    def input(self, name: str) -> Input:
        """Return a named input or raise an error identifying the missing branch."""

        try:
            return self.inputs[name]
        except KeyError as error:
            raise ValueError(f"Missing model input {name!r}") from error

    def events_for_feature(self, feature: str) -> tuple[str, ...]:
        """Events consumed by sequence inputs using a declared feature."""

        return tuple(
            dict.fromkeys(
                event
                for value in self.inputs.values()
                if value.type == "sequence" and feature in value.features
                for event in value.events
            )
        )

    def validate(self) -> None:
        """Normalize and validate data settings, then validate the selected model."""

        self.validate_data()
        self.validate_model()

    def validate_data(self) -> None:
        """Validate only the model-independent retrieval/data contract."""

        self._normalise_config_shorthand()
        self._validate_shared_contract()
        self._validate_histories()

    def validate_model(self) -> None:
        """Validate the selected model recipe after the data contract is valid."""

        from genrecsys.core.models import get

        self._validate_inputs()
        get(self.model.type).validate_config(self)

    def _normalise_config_shorthand(self) -> None:
        """Expand YAML shorthand and assign event IDs in configuration order.

        Update the config in place; event IDs do not depend on train frequencies.
        """

        normalised_metrics: dict[str, Metric | None] = {}
        for name, metric in self.task.metrics.items():
            if metric is None:
                metric = Metric()
            if metric.type is None:
                metric.type = name
            normalised_metrics[name] = metric
        self.task.metrics = normalised_metrics

        for event_name, event in self.events.items():
            normalised: dict[str, Attribute | None] = {}
            for name, attribute in event.attributes.items():
                if name == "event":
                    if attribute is None or attribute.field not in {None, "event"}:
                        raise ValueError(
                            f"events.{event_name}.attributes.event is generated by "
                            "the framework"
                        )
                    continue
                if attribute is None:
                    attribute = Attribute()
                if attribute.field is None:
                    attribute.field = name
                normalised[name] = attribute
            # Event name is an ordinary categorical feature. Keeping it in the
            # same encoder pipeline avoids a model-only pseudo-feature.
            normalised["event"] = Attribute(field="event")
            event.attributes = normalised

        # Event names are declared configuration, not categories to be learned
        # from whichever streams happened to occur in the train window.  Freeze
        # their stable 1-based segment ids in the ordinary ID preprocessor.
        self.encoders["event"] = Encoder(
            type="id", preprocessor={"categories": list(self.events)}
        )
        self.features["event"] = Feature(
            located_in=LocatedIn(event=True),
            encoder="event",
        )

        # OmegaConf materialises tuple-annotated YAML sequences as lists. Keep
        # the typed contract stable after loading and after round-trips.
        for model_input in self.inputs.values():
            model_input.events = tuple(model_input.events)
            model_input.features = tuple(model_input.features)
            model_input.optional_features = tuple(model_input.optional_features)

    def _validate_shared_contract(self) -> None:
        """Validate shared task, metric, execution, encoder, and history settings."""

        if self.inference.event_store_backend not in {"disk", "ram"}:
            raise ValueError("inference.event_store_backend must be disk or ram")
        if self.task.type != "retrieval":
            raise ValueError("Only task.type=retrieval is currently supported")
        if self.task.baseline is not None:
            if self.task.baseline.type != "global_count":
                raise ValueError("Only task.baseline.type=global_count is supported")
            if self.task.baseline.params:
                raise ValueError("task.baseline.params must be empty for global_count")
        if not self.task.metrics:
            raise ValueError("task.metrics must not be empty")
        metrics_by_k: dict[int, set[str]] = {}
        for name, metric in self.task.metrics.items():
            if metric is None or metric.type is None:
                raise AssertionError("Metric shorthand must be normalised")
            if metric.type not in _SUPPORTED_METRICS:
                raise ValueError(
                    f"Unsupported task metric {name!r}: type={metric.type!r}"
                )
            if set(metric.params) != {"k"}:
                raise ValueError(f"task.metrics.{name}.params must contain exactly k")
            if not isinstance(metric.params["k"], int) or isinstance(
                metric.params["k"], bool
            ):
                raise TypeError(f"task.metrics.{name}.params.k must be an integer")
            _require_positive(
                f"task.metrics.{name}.params.k",
                metric.params["k"],
            )
            metric_types = metrics_by_k.setdefault(metric.params["k"], set())
            if metric.type in metric_types:
                raise ValueError(
                    f"Duplicate {metric.type} declaration for k={metric.params['k']}"
                )
            metric_types.add(metric.type)

        if not self.events:
            raise ValueError("events must not be empty")
        for name, event in self.events.items():
            if not isinstance(event.priority, int) or isinstance(event.priority, bool):
                raise TypeError(f"events.{name}.priority must be an integer")
        if "position" in self.features:
            raise ValueError("feature name 'position' is reserved")
        if not self.encoders:
            raise ValueError("encoders must not be empty")
        if not self.features:
            raise ValueError("features must not be empty")
        for name, feature in self.features.items():
            transformation = feature.transform
            if transformation is not None:
                from genrecsys.core import feature_transforms

                feature_transforms.validate(
                    transformation.type,
                    transformation.params,
                )
                if not feature.located_in.event:
                    raise ValueError(
                        f"transformed feature {name!r} must be located in events"
                    )
            locations = feature.located_in
            if not any((locations.event, locations.context, locations.artifacts)):
                raise ValueError(
                    f"features.{name}.located_in must contain at least one location"
                )

        declared_encoders = set(self.encoders)
        feature_encoders = {feature.encoder for feature in self.features.values()}
        if unused_encoders := declared_encoders - feature_encoders:
            raise ValueError(
                f"encoders are declared, but unused: {sorted(unused_encoders)}"
            )
        if unknown_encoders := feature_encoders - declared_encoders:
            raise ValueError(f"unknown encoders: {sorted(unknown_encoders)}")
        if self.max_duration_per_sequence is not None:
            duration_days(
                self.max_duration_per_sequence,
                field_name="max_duration_per_sequence",
            )

    def _validate_histories(self) -> None:
        """Check only the event sets and bounds used to materialize histories."""

        histories = materialization_histories(self)
        if not histories:
            raise ValueError("config contains no sequence histories")
        for history in histories:
            if not history["events"]:
                raise ValueError("sequence history must reference events")
            if unknown := set(history["events"]) - set(self.events):
                raise ValueError(
                    f"history references unknown events: {sorted(unknown)}"
                )
            _require_positive("materialization event limit", history["max_events"])
            if history["max_duration"] is not None:
                duration_days(history["max_duration"], field_name="history duration")

    def _validate_inputs(self) -> None:
        """Check model-input references and runtime bounds; not called by preparation."""

        for name, model_input in self.inputs.items():
            if model_input.type not in {"sequence", "context"}:
                raise ValueError(f"inputs.{name}.type must be sequence or context")
            if not model_input.features:
                raise ValueError(f"inputs.{name}.features must not be empty")
            if unknown := set(model_input.features) - set(self.features):
                raise ValueError(
                    f"inputs.{name}.features: unknown references {sorted(unknown)}"
                )
            if unknown := set(model_input.optional_features) - set(
                model_input.features
            ):
                raise ValueError(
                    f"inputs.{name}.optional_features: not in features {sorted(unknown)}"
                )
            if model_input.type == "context":
                if model_input.events:
                    raise ValueError(f"context input {name!r} cannot reference events")
                continue
            maximum = model_input.max_events_per_sequence
            if maximum is None:
                raise ValueError(f"inputs.{name}.max_events_per_sequence is required")
            _require_positive(f"inputs.{name}.max_events_per_sequence", maximum)
            if model_input.selection not in {"recent", "priority"}:
                raise ValueError(f"inputs.{name}.selection must be recent or priority")
            priority_maximum = model_input.max_priority_events_per_sequence
            if priority_maximum is not None:
                if model_input.selection != "priority":
                    raise ValueError(
                        f"inputs.{name}.max_priority_events_per_sequence requires selection=priority"
                    )
                _require_positive(
                    f"inputs.{name}.max_priority_events_per_sequence",
                    priority_maximum,
                )
                if priority_maximum > maximum:
                    raise ValueError(
                        f"inputs.{name}.max_priority_events_per_sequence cannot "
                        "exceed max_events_per_sequence"
                    )


def materialization_histories(config: Config, /) -> list[dict[str, Any]]:
    """Return data-only history specifications, independent of model port names.

    Multiple ports with identical event sets/windows materialize one history at
    their largest bound. Feature order, projection and compression do not affect it.
    """

    bounds: dict[tuple[tuple[str, ...], str | None, str, int | None], int] = {}
    for value in config.inputs.values():
        if value.type != "sequence":
            continue
        maximum = value.max_events_per_sequence
        if maximum is None:
            raise ValueError("sequence history has no materialization event limit")
        key = (
            tuple(sorted(set(value.events))),
            value.max_duration_per_sequence,
            value.selection,
            value.max_priority_events_per_sequence,
        )
        bounds[key] = max(bounds.get(key, 0), maximum)
    return [
        {
            "events": list(events),
            "max_events": maximum,
            "max_duration": duration,
            **({"selection": selection} if selection != "recent" else {}),
            **(
                {"max_priority_events": priority_maximum}
                if priority_maximum is not None
                else {}
            ),
        }
        for (events, duration, selection, priority_maximum), maximum in sorted(
            bounds.items(),
            key=lambda entry: (
                entry[0][0],
                entry[0][1] or "",
                entry[0][2],
                entry[0][3] or 0,
            ),
        )
    ]


def duration_days(value: str | None, *, field_name: str) -> int:
    """Convert the currently supported whole-day duration syntax to days."""

    if not isinstance(value, str) or (match := _DAY_DURATION.fullmatch(value)) is None:
        raise ValueError(
            f"{field_name} must be a positive whole-day duration, e.g. 180d"
        )
    return int(match.group(1))


def _require_positive(name: str, value: float) -> None:
    """Reject a nonpositive value with an error naming its config field."""

    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def _load_recipe(path: Path, *, seen: set[Path]):
    """Compose explicit parent recipes in order, then apply the local overlay."""

    path = path.resolve()
    if path in seen:
        raise ValueError(f"cyclic config extends chain at {path}")
    loaded = OmegaConf.load(path)
    parents = loaded.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, (list, ListConfig)) or any(
        not isinstance(parent, str) or not parent for parent in parents
    ):
        raise TypeError("config extends must be a path string or a list of paths")
    inherited = OmegaConf.create({})
    for parent in parents:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        inherited = _merge_recipe(
            inherited, _load_recipe(parent_path, seen=seen | {path})
        )
    return _merge_recipe(inherited, loaded)


def _merge_recipe(inherited, loaded):
    """Keep architecture and decoding dictionaries owned by the selected model."""

    model_type = OmegaConf.select(loaded, "model.type")
    if model_type is not None and model_type != OmegaConf.select(
        inherited, "model.type"
    ):
        # Architecture-specific dictionaries cannot inherit another model's keys.
        inherited.model = loaded.model
        OmegaConf.update(
            inherited,
            "inference.predict_kwargs",
            OmegaConf.select(loaded, "inference.predict_kwargs", default={}),
            merge=False,
        )
    return OmegaConf.merge(inherited, loaded)
