"""Affect-conditioned prompt synthesis with ablation tiers."""

from __future__ import annotations

from novavision.taxonomy import NEUTRAL

TIERS = ("raw", "emotion", "affect")

STYLE_PRESETS = {
    "photorealistic": "hyperrealistic photograph, 85mm lens, natural light, ultra sharp, 8k",
    "artistic": "digital painting, expressive brushwork, rich color, dramatic lighting",
    "abstract": "abstract art, geometric forms, bold composition, modern gallery piece",
    "nature": "landscape photography, golden hour, atmospheric depth, fine detail",
    "dreamscape": "surreal dreamscape, ethereal glow, volumetric light, imaginative",
}

EMOTION_SCENES = {
    "joy": "a radiant sunlit meadow full of blooming wildflowers",
    "sadness": "a misty rain-soaked forest at dusk",
    "anger": "a dramatic thunderstorm with lightning over a dark sky",
    "fear": "a fog-shrouded landscape under a pale moon",
    "surprise": "a sudden burst of colorful aurora across the night sky",
    "disgust": "a murky overgrown swamp with sickly green hues",
    "neutral": "a quiet zen garden at first light",
}

QUALITY = "masterpiece, highly detailed, sharp focus, professional"

NEGATIVE_PROMPT = (
    "watermark, text, signature, logo, blurry, low quality, distorted, "
    "deformed, bad anatomy, cropped, jpeg artifacts, lowres"
)


def va_descriptors(valence: float, arousal: float) -> str:
    """Map valence/arousal to palette and lighting cues."""
    if valence >= 0.33:
        palette = "warm vibrant palette, golden tones"
    elif valence <= -0.33:
        palette = "cool desaturated palette, muted blue-grey tones"
    else:
        palette = "balanced natural palette"

    if arousal >= 0.66:
        light = "dramatic high-contrast lighting, dynamic composition"
    elif arousal <= 0.33:
        light = "soft gentle lighting, calm still composition"
    else:
        light = "even lighting, steady composition"

    return f"{palette}, {light}"


def build_prompt(
    text: str,
    *,
    emotion: str = NEUTRAL,
    valence: float = 0.0,
    arousal: float = 0.5,
    style: str = "artistic",
    tier: str = "affect",
) -> str:
    """Compose the generation prompt for the given conditioning tier."""
    if tier not in TIERS:
        raise ValueError(f"Unknown tier '{tier}', expected one of {TIERS}")

    style_desc = STYLE_PRESETS.get(style.lower(), STYLE_PRESETS["artistic"])
    scene = EMOTION_SCENES.get(emotion, EMOTION_SCENES[NEUTRAL])

    if tier == "raw":
        parts = [text, style_desc, QUALITY]
    elif tier == "emotion":
        parts = [scene, f"inspired by '{text}'", style_desc, QUALITY]
    else:
        parts = [scene, va_descriptors(valence, arousal), f"inspired by '{text}'", style_desc, QUALITY]

    return ", ".join(parts)
