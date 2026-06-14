"""Image generation backends."""

from __future__ import annotations

from novavision.generation.base import ImageBackend, NullBackend


def get_backend(name: str = "null", **kwargs) -> ImageBackend:
    """Construct a backend by name: null, diffusers, or hf-api."""
    name = name.lower()
    if name == "null":
        return NullBackend(**kwargs)
    if name == "diffusers":
        from novavision.generation.diffusers_backend import DiffusersBackend

        return DiffusersBackend(**kwargs)
    if name == "hf-api":
        from novavision.generation.hf_api_backend import HFApiBackend

        return HFApiBackend(**kwargs)
    raise ValueError(f"Unknown backend '{name}'")


__all__ = ["ImageBackend", "NullBackend", "get_backend"]
