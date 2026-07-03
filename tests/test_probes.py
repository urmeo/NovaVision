import math

import pytest
from PIL import Image

from novavision.eval.probes import HFImageClassifierProbe


def _img():
    return Image.new("RGB", (8, 8), (10, 20, 30))


def test_hf_probe_maps_and_aggregates():
    probe = HFImageClassifierProbe("fake/model", label_map={"happy": "joy"})
    probe._pipe = lambda img: [
        {"label": "happy", "score": 0.6},
        {"label": "joy", "score": 0.2},
        {"label": "anger", "score": 0.2},
    ]
    rec = probe.recover(_img())
    assert rec.emotion == "joy"  # happy->joy aggregates to 0.8
    assert round(rec.scores["joy"], 4) == 0.8


def test_hf_probe_drops_unknown_labels():
    probe = HFImageClassifierProbe("fake/model")
    probe._pipe = lambda img: [{"label": "cat", "score": 0.9}, {"label": "fear", "score": 0.1}]
    assert probe.recover(_img()).emotion == "fear"


def test_clip_t_default_is_nan():
    probe = HFImageClassifierProbe("fake/model")
    assert math.isnan(probe.clip_t(_img(), "a city street"))


def test_probe_name():
    assert HFImageClassifierProbe("org/emotion-vit").name == "img:emotion-vit"


def test_hf_probe_raises_when_nothing_maps():
    probe = HFImageClassifierProbe("fake/model")
    probe._pipe = lambda img: [{"label": "cat", "score": 0.9}]
    with pytest.raises(RuntimeError):
        probe.recover(_img())


def test_hf_probe_label_coverage_check():
    probe = HFImageClassifierProbe("fake/model", label_map={"happy": "joy"})
    assert probe._check_label_coverage({0: "happy", 1: "cat"}) == ["joy"]
    with pytest.raises(ValueError):
        HFImageClassifierProbe("fake/model")._check_label_coverage({0: "cat"})
