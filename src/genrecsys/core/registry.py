"""Small typed registries used by config-selected GenRecSys entities."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Mapping[str, T], Generic[T]):
    """A fail-fast registry whose keys are the ``type`` values from config."""

    def __init__(self, kind: str) -> None:
        """Initialize a registry with an entity name for error messages."""

        self.kind = kind
        self._values: dict[str, T] = {}

    def register(self, name: str, value: T, /) -> T:
        """Register an implementation, rejecting empty names and duplicate keys."""

        if not name:
            raise ValueError(f"{self.kind} type must not be empty")
        if name in self._values:
            raise ValueError(f"{self.kind} type {name!r} is already registered")
        self._values[name] = value
        return value

    def __getitem__(self, name: str) -> T:
        """Return an implementation or raise an error listing available types."""

        try:
            return self._values[name]
        except KeyError as error:
            available = ", ".join(sorted(self._values)) or "<none>"
            raise ValueError(
                f"Unknown {self.kind} type {name!r}; available: {available}"
            ) from error

    def __iter__(self) -> Iterator[str]:
        """Iterate over registered names in insertion order."""

        return iter(self._values)

    def __contains__(self, name: object) -> bool:
        """Return whether a name is registered."""

        return name in self._values

    def __len__(self) -> int:
        """Return the number of registered implementations."""

        return len(self._values)
