"Cross-module invariants."

from __future__ import annotations

import inspect

import numpy as np

from novavision.taxonomy import (
    EMOTION_PRIORS,
    EMOTION_PROMPTS,
    EMOTIONS,
    GOEMOTIONS_TO_EKMAN,
)


def test_emotion_prompts_match_emotions_in_order():
    assert tuple(EMOTION_PROMPTS) == EMOTIONS


def test_emotion_priors_cover_every_emotion():
    assert set(EMOTION_PRIORS) == set(EMOTIONS)


def test_goemotions_maps_only_into_ekman():
    assert set(GOEMOTIONS_TO_EKMAN.values()) <= set(EMOTIONS)


class _FakeArr:
    """A torch-free stand-in exposing just what CLIPProbe.recover calls."""

    def __init__(self, a):
        self.a = np.asarray(a, dtype=float)

    def __matmul__(self, other):
        return _FakeArr(self.a @ other.a)

    @property
    def T(self):
        return _FakeArr(self.a.T)

    def squeeze(self, axis=None):
        return _FakeArr(np.squeeze(self.a, axis))

    def cpu(self):
        return self

    def numpy(self):
        return self.a


def test_clip_probe_maps_argmax_to_canonical_label(monkeypatch):

    from novavision.eval import probes

    probe = probes.CLIPProbe()
    probe._scale = 1.0
    eye = _FakeArr(np.eye(len(EMOTIONS)))
    monkeypatch.setattr(probe, "_fixed_features", lambda: (eye, None, None))
    monkeypatch.setattr(probe, "_expected", lambda img, feats, ladder: 0.0)

    for idx, emotion in enumerate(EMOTIONS):
        onehot = _FakeArr(np.eye(len(EMOTIONS))[idx : idx + 1])
        monkeypatch.setattr(probe, "_image_features", lambda image, v=onehot: v)
        rec = probe.recover(image=None)
        assert tuple(rec.scores.keys()) == EMOTIONS
        assert rec.emotion == emotion


def test_all_backends_default_to_same_size():
    from novavision.generation.base import ImageBackend, NullBackend
    from novavision.generation.diffusers_backend import DiffusersBackend
    from novavision.generation.hf_api_backend import HFApiBackend

    sizes = {}
    for backend in (ImageBackend, NullBackend, DiffusersBackend, HFApiBackend):
        params = inspect.signature(backend.generate).parameters
        sizes[backend.__name__] = (params["width"].default, params["height"].default)
    assert set(sizes.values()) == {(512, 512)}, sizes


def test_seed_is_independent_of_tier():
    from novavision.experiments import run

    assert "tier" not in inspect.signature(run._seed).parameters

    assert run._seed(0, 3, 4, 1) != run._seed(0, 3, 5, 1)


def test_seed_bound_assumption_is_still_valid():

    assert len(EMOTIONS) == 7


def test_report_conditions_are_producible_by_run():
    import report

    from novavision.experiments import run

    producible = set(run.CONDITIONS["content"]) | set(run.CONDITIONS["text"])

    assert set(report.CONDITIONS) == producible
    assert tuple(report.CONDITIONS) == run.ALL_CONDITIONS


def test_manifest_exposes_public_provenance_api():
    from novavision.experiments import manifest

    assert callable(manifest.git_sha)
    assert callable(manifest.package_version)
    assert manifest.package_version("definitely-not-a-real-package") == "absent"


def test_manifest_records_device_provenance():

    from novavision.experiments import manifest

    m = manifest.build_manifest(backend="null")
    assert "device_info" in m
    assert set(m["device_info"]) >= {"cuda_available", "device_name", "cuda"}


def test_manifest_pins_only_default_models():

    from novavision.config import CLIP_REVISION, DIFFUSION_REVISION
    from novavision.experiments import manifest

    default = manifest.build_manifest(backend="null")["model_revisions"]
    assert default["diffusion"] == DIFFUSION_REVISION
    assert default["clip"] == CLIP_REVISION

    swapped = manifest.build_manifest(
        backend="diffusers", diffusion_model="org/other-model", clip_model="org/other-probe"
    )["model_revisions"]
    assert swapped["diffusion"] is None
    assert swapped["clip"] is None


def test_device_info_degrades_without_torch(monkeypatch):

    import sys

    from novavision.experiments import manifest

    monkeypatch.setitem(sys.modules, "torch", None)
    info = manifest.device_info()
    assert info["cuda_available"] is False
    assert info["device_name"] is None
    assert info["cuda"] is None


def test_version_synced_across_metadata():

    import re
    from pathlib import Path

    import novavision

    root = Path(novavision.__file__).resolve().parents[1]
    cff = (root / "CITATION.cff").read_text()
    assert re.search(rf"^version: {re.escape(novavision.__version__)}$", cff, re.M)
