from __future__ import annotations

from typing import Callable, Dict, List

from .base import BaseAdapter


class AdapterRegistry:
    """Factory-based adapter registry with case-insensitive keys.

    Public contract (used by tests + runner/engine):
      - register(family, factory)
      - get(family) -> factory
      - make(family) -> adapter instance
      - families() -> list[str]

    Notes
    -----
    We normalize family keys with `.strip().lower()` so that schema/spec values like
    "DID"/"did"/"Did" all resolve to the same adapter family.
    """

    def __init__(self) -> None:
        self._factories: Dict[str, Callable[[], BaseAdapter]] = {}

    @staticmethod
    def _key(family: str) -> str:
        return str(family).strip().lower()

    def register(self, family: str, factory: Callable[[], BaseAdapter]) -> None:
        self._factories[self._key(family)] = factory

    def register_instance(self, family: str, adapter: BaseAdapter) -> None:
        self.register(family, lambda: adapter)

    def get(self, family: str) -> Callable[[], BaseAdapter]:
        """Return the *factory* for `family` (do not instantiate)."""
        key = self._key(family)
        if key not in self._factories:
            raise KeyError(f"Unknown adapter family: {family!r}. Available: {sorted(self._factories.keys())}")
        return self._factories[key]

    def make(self, family: str) -> BaseAdapter:
        """Instantiate and return an adapter for `family`."""
        return self.get(family)()

    # Backward/alt names
    def families(self) -> List[str]:
        return sorted(self._factories.keys())

    def list_families(self) -> List[str]:
        return self.families()
