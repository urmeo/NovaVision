import pytest

from novavision.prompting import EMOTION_SCENES, TIERS, build_prompt, va_descriptors


def test_tiers():
    assert TIERS == ("raw", "emotion", "affect")


def test_raw_keeps_content_and_skips_emotion():
    prompt = build_prompt("a red bicycle", emotion="joy", tier="raw")
    assert "a red bicycle" in prompt
    assert "mood" not in prompt


def test_emotion_tier_keeps_content_and_adds_mood():
    prompt = build_prompt("a city street", emotion="joy", tier="emotion")
    assert "a city street" in prompt
    assert "joyful uplifting mood" in prompt


def test_affect_tier_adds_palette():
    prompt = build_prompt("a city street", emotion="joy", valence=0.8, arousal=0.7, tier="affect")
    assert "a city street" in prompt
    assert "warm vibrant palette" in prompt
    assert "dramatic high-contrast" in prompt


def test_scene_floor_drops_content():
    prompt = build_prompt("a city street", emotion="joy", tier="scene")
    assert "a city street" not in prompt
    assert EMOTION_SCENES["joy"] in prompt


def test_unknown_tier_raises():
    with pytest.raises(ValueError):
        build_prompt("x", tier="bogus")


def test_unknown_style_falls_back():
    assert build_prompt("x", style="does-not-exist", tier="raw")


def test_va_descriptors_low_low():
    desc = va_descriptors(-0.8, 0.1)
    assert "cool desaturated" in desc
    assert "soft gentle" in desc
