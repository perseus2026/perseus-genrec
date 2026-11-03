"""Config-selected native model implementations."""

from importlib import import_module

from genrecsys.core.models.base import (
    Implementation,
    TrainedModel,
)
from genrecsys.core.registry import Registry

registry: Registry[Implementation] = Registry("model")


def register(name: str, implementation: Implementation, /) -> Implementation:
    """Register an implementation explicitly, primarily for external tests/plugins."""

    return registry.register(name, implementation)


def _load_builtin(name: str) -> None:
    """Load a built-in model by package convention instead of a central switch."""

    if name in registry:
        return
    try:
        module = import_module(f"genrecsys.core.models.{name}")
    except ModuleNotFoundError as error:
        if error.name == f"genrecsys.core.models.{name}":
            # Preserve the registry's consistent error with available plugins.
            registry[name]
        raise
    try:
        implementation = module.implementation
    except AttributeError as error:
        raise ValueError(
            f"Model package {module.__name__!r} must export 'implementation'"
        ) from error
    if not isinstance(implementation, Implementation):
        raise TypeError(f"{module.__name__}.implementation must be an Implementation")
    registry.register(name, implementation)


def get(name: str, /) -> Implementation:
    """Return a model plugin, lazily importing a built-in model when needed."""

    _load_builtin(name)
    return registry[name]


__all__ = [
    "Implementation",
    "TrainedModel",
    "get",
    "register",
    "registry",
]
