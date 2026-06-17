"""Provenance manifest for a benchmark run."""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata


def _version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "absent"


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def build_manifest(**config) -> dict:
    """Everything a reader needs to reconstruct the environment and run."""
    from novavision.config import CLIP_REVISION, DIFFUSION_REVISION, EMOTION_REVISION

    return {
        "git_sha": _git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            pkg: _version(pkg)
            for pkg in ("torch", "transformers", "diffusers", "numpy", "datasets", "pillow")
        },
        "model_revisions": {
            "emotion": EMOTION_REVISION,
            "diffusion": DIFFUSION_REVISION,
            "clip": CLIP_REVISION,
        },
        "config": config,
    }
