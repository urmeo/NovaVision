import pytest

from novavision.prompting import TIERS, build_prompt, va_descriptors


def test_tiers():
    assert TIERS == ("raw", "emotion", "affect")


def test_raw_keeps_text_and_skips_scene():
    prompt = build_prompt("a red bicycle", emotion="joy", tier="raw")
    assert "a red bicycle" in prompt
    assert "meadow" not in prompt


def test_emotion_tier_injects_scene():
    prompt = build_prompt("good news", emotion="joy", tier="emotion")
    assert "meadow" in prompt


def test_affect_tier_adds_palette():
    prompt = build_prompt("good news", emotion="joy", valence=0.8, arousal=0.7, tier="affect")
    assert "warm vibrant palette" in prompt
    assert "dramatic high-contrast" in prompt


def test_unknown_tier_raises():
    with pytest.raises(ValueError):
        build_prompt("x", tier="bogus")


def test_unknown_style_falls_back():
    assert build_prompt("x", style="does-not-exist", tier="raw")


def test_va_descriptors_low_low():
    desc = va_descriptors(-0.8, 0.1)
    assert "cool desaturated" in desc
    assert "soft gentle" in desc
