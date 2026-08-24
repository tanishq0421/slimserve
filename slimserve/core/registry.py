"""A tiny name -> class registry (Factory pattern).

Lets configs stay declarative: a YAML says ``engine: vllm`` and the pipeline
calls ``build("engine", "vllm", cfg)`` without importing the concrete class.
This is the seam that makes the whole thing config-driven and swappable.
"""
from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")

# category -> {name -> class}
_REGISTRY: dict[str, dict[str, type]] = {}


def register(category: str, name: str) -> Callable[[type[T]], type[T]]:
    """Class decorator: ``@register("engine", "vllm")``."""
    def deco(cls: type[T]) -> type[T]:
        _REGISTRY.setdefault(category, {})[name] = cls
        return cls
    return deco


def build(category: str, name: str, *args, **kwargs):
    """Instantiate a registered class by name."""
    try:
        cls = _REGISTRY[category][name]
    except KeyError as exc:
        known = list(_REGISTRY.get(category, {}))
        raise KeyError(
            f"No {category!r} registered as {name!r}. Known: {known}"
        ) from exc
    return cls(*args, **kwargs)


def available(category: str) -> list[str]:
    return list(_REGISTRY.get(category, {}))
