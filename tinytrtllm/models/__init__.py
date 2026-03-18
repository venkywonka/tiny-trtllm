"""Model registry — maps architecture names to model classes."""

from __future__ import annotations

from typing import Type

_REGISTRY: dict[str, Type] = {}


def register_model(arch_name: str, cls: Type) -> None:
    _REGISTRY[arch_name] = cls


def get_model_class(arch_name: str) -> Type:
    if arch_name not in _REGISTRY:
        raise KeyError(
            f"Unknown architecture: {arch_name}. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[arch_name]


def list_models() -> list[str]:
    return list(_REGISTRY.keys())
