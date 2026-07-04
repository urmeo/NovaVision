"""Cross-module invariants.

The rest of the suite pins behavior *through* each module on its happy path.
These pin behavior *across* module boundaries: the silent couplings (label
ordering, backend-default parity, seed pairing, public provenance API, tier
vocabulary) that no single-module test observes and that a refactor can break
with every other test still green.
"""

from __future__ import annotations

import inspect

import numpy as np

from novavision.taxonomy import (
    EMOTION_PRIORS,
    EMOTION_PROMPTS,
    EMOTIONS,
    GOEMOTIONS_TO_EKMAN,
)

# --- Label ordering (finding 1): the CLIP probe zips stacked text features
#     against EMOTIONS order; any drift silently mis-maps every score. ---


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
    # Behavioral contract (finding 1): recover() must key scores by EMOTIONS order
    # and map the argmax to that label, so the metrics layer (called with EMOTIONS)
    # agrees. Torch-free so it runs in the dev CI job; catches a revert to dict
    # order and survives equivalent refactors (list(EMOTIONS), [*EMOTIONS], ...).
    from novavision.eval import probes

    probe = probes.CLIPProbe()
    probe._scale = 1.0
    eye = _FakeArr(np.eye(len(EMOTIONS)))  # emo feature i lights up on input i
    monkeypatch.setattr(probe, "_fixed_features", lambda: (eye, None, None))
    monkeypatch.setattr(probe, "_expected", lambda img, feats, ladder: 0.0)

    for idx, emotion in enumerate(EMOTIONS):
        onehot = _FakeArr(np.eye(len(EMOTIONS))[idx : idx + 1])
        monkeypatch.setattr(probe, "_image_features", lambda image, v=onehot: v)
        rec = probe.recover(image=None)
        assert tuple(rec.scores.keys()) == EMOTIONS
        assert rec.emotion == emotion


# --- Backend-default parity (finding 19): the same pipeline must render the
#     same size whichever backend runs, when no explicit size is passed. ---


def test_all_backends_default_to_same_size():
    from novavision.generation.base import ImageBackend, NullBackend
    from novavision.generation.diffusers_backend import DiffusersBackend
    from novavision.generation.hf_api_backend import HFApiBackend

    sizes = {}
    for backend in (ImageBackend, NullBackend, DiffusersBackend, HFApiBackend):
        params = inspect.signature(backend.generate).parameters
        sizes[backend.__name__] = (params["width"].default, params["height"].default)
    assert set(sizes.values()) == {(512, 512)}, sizes


# --- Seed pairing (finding 4): tiers are paired on generation noise because the
#     seed is a function of (item, emotion, seed) only, never the tier. ---


def test_seed_is_independent_of_tier():
    from novavision.experiments import run

    # Same (base, ci, ei, sk) must give one seed regardless of which tier renders.
    assert run._seed(0, 3, 4, 1) == run._seed(0, 3, 4, 1)
    # Distinct coordinates must give distinct seeds inside the guarded domain.
    assert run._seed(0, 3, 4, 1) != run._seed(0, 3, 5, 1)


def test_seed_bound_assumption_is_still_valid():
    # SEED_MAX_ITEMS/SEED_MAX_SEEDS were derived for 7 emotions (the ei*13 term
    # must not overflow the 97 stride). If the taxonomy grows, the bounds must be
    # re-derived; this fires as the reminder. Exhaustive injectivity is covered by
    # test_experiment.py::test_seed_injective_within_guarded_domain, not repeated here.
    assert len(EMOTIONS) == 7


# --- Tier vocabulary (report vs run): report.py must not describe a tier the
#     experiment cannot produce, or a run's tables silently omit/mislabel it. ---


def test_report_conditions_are_producible_by_run():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import report

    from novavision.experiments import run

    producible = set(run.CONDITIONS["content"]) | set(run.CONDITIONS["text"])
    assert set(report.CONDITIONS) <= producible, set(report.CONDITIONS) - producible


# --- Public provenance API (finding 34): resummarize depends on manifest's
#     public names, not private helpers. ---


def test_manifest_exposes_public_provenance_api():
    from novavision.experiments import manifest

    assert callable(manifest.git_sha)
    assert callable(manifest.package_version)
    assert manifest.package_version("definitely-not-a-real-package") == "absent"


def test_manifest_records_device_provenance():
    # Bit-exactness depends on the device, so the manifest must carry accelerator
    # provenance.
    from novavision.experiments import manifest

    m = manifest.build_manifest(backend="null")
    assert "device_info" in m
    assert set(m["device_info"]) >= {"cuda_available", "device_name", "cuda"}


def test_device_info_degrades_without_torch(monkeypatch):
    # The deterministic (no-torch) path must still build a manifest; exercise the
    # degradation branch directly rather than relying on torch being absent.
    import sys

    from novavision.experiments import manifest

    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise
    info = manifest.device_info()
    assert info["cuda_available"] is False
    assert info["device_name"] is None
    assert info["cuda"] is None
