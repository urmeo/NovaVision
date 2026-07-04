"""Provenance manifest for a benchmark run."""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata


def package_version(pkg: str) -> str:
    """Installed version of a package, or ``"absent"`` if it is not installed.

    Public API: other modules (e.g. ``scripts/resummarize.py``) record provenance
    with this, so it is a supported name, not an internal helper.
    """
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "absent"


def git_sha() -> str:
    """Current commit SHA, or ``"unknown"`` outside a git checkout. Public API."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def device_info() -> dict:
    """Best-effort accelerator provenance.

    Bit-exactness depends on the device, so the accelerator name and CUDA/cuDNN
    versions are the fields most likely to explain a non-reproduction. Degrades to
    a torch-absent stub without raising, so the deterministic (no-torch) path still
    builds a manifest.
    """
    info: dict = {
        "cuda_available": False,
        "mps_available": False,
        "device_name": None,
        "cuda": None,
        "cudnn": None,
    }
    try:
        import torch
    except Exception:
        return info
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["mps_available"] = bool(torch.backends.mps.is_available())
        info["cuda"] = getattr(torch.version, "cuda", None)
        if info["cuda_available"]:
            info["device_name"] = torch.cuda.get_device_name(0)
            cudnn = torch.backends.cudnn.version()
            info["cudnn"] = int(cudnn) if cudnn else None
    except Exception:
        pass
    return info


def build_manifest(**config) -> dict:
    """Everything a reader needs to reconstruct the environment and run."""
    from novavision.config import CLIP_REVISION, DIFFUSION_REVISION, EMOTION_REVISION

    return {
        "git_sha": git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device_info": device_info(),
        "packages": {
            pkg: package_version(pkg)
            for pkg in ("torch", "transformers", "diffusers", "numpy", "datasets", "pillow")
        },
        "model_revisions": {
            "emotion": EMOTION_REVISION,
            "diffusion": DIFFUSION_REVISION,
            "clip": CLIP_REVISION,
        },
        "config": config,
    }
