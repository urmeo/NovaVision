"""Prompt synthesis: emotion as a modifier on independent content.

The conditioning tiers are the ablation. Critically, the image *content* comes
from the input (text or a content-bank subject), and emotion is layered on as a
mood/affect modifier; content is never chosen by the emotion label. That
decoupling is what lets recovery be attributed to the conditioning. The old
fixed per-emotion scenes survive only as a ``scene`` floor that measures how
much recovery is pure template recognition.
"""

from __future__ import annotations

from novavision.taxonomy import NEUTRAL

TIERS = ("raw", "naive", "emotion", "affect")
FLOORS = ("scene",)

STYLE_PRESETS = {
    "photorealistic": "hyperrealistic photograph, 85mm lens, natural light, ultra sharp, 8k",
    "artistic": "digital painting, expressive brushwork, rich color, dramatic lighting",
    "abstract": "abstract art, geometric forms, bold composition, modern gallery piece",
    "nature": "landscape photography, golden hour, atmospheric depth, fine detail",
    "dreamscape": "surreal dreamscape, ethereal glow, volumetric light, imaginative",
}

# Affect modifiers, applied to any content (not scene replacements).
EMOTION_MOODS = {
    "joy": "joyful uplifting mood, bright cheerful atmosphere",
    "sadness": "sad melancholic mood, somber wistful atmosphere",
    "anger": "angry intense mood, hostile aggressive atmosphere",
    "fear": "fearful anxious mood, tense ominous atmosphere",
    "surprise": "surprised startled mood, sudden astonishing atmosphere",
    "disgust": "disgusted repulsed mood, sickly off-putting atmosphere",
    "neutral": "calm neutral mood, plain everyday atmosphere",
}

# Fixed per-emotion scenes, used ONLY by the `scene` floor.
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

# Valence/arousal bucket boundaries for the palette/lighting mapping (circumplex thirds).
VALENCE_POSITIVE, VALENCE_NEGATIVE = 0.33, -0.33
AROUSAL_HIGH, AROUSAL_LOW = 0.66, 0.33

NEGATIVE_PROMPT = (
    "watermark, text, signature, logo, blurry, low quality, distorted, "
    "deformed, bad anatomy, cropped, jpeg artifacts, lowres"
)


def va_descriptors(valence: float, arousal: float) -> str:
    """Map valence/arousal to palette and lighting cues.

    On the content track the valence/arousal passed here is the per-emotion prior,
    so these cues are a deterministic function of the emotion, the ``affect`` vs
    ``emotion`` contrast there measures palette/lighting strength only. Independent,
    text-grounded valence/arousal is exercised on the text track.
    """
    if valence >= VALENCE_POSITIVE:
        palette = "warm vibrant palette, golden tones"
    elif valence <= VALENCE_NEGATIVE:
        palette = "cool desaturated palette, muted blue-grey tones"
    else:
        palette = "balanced natural palette"

    if arousal >= AROUSAL_HIGH:
        light = "dramatic high-contrast lighting, dynamic composition"
    elif arousal <= AROUSAL_LOW:
        light = "soft gentle lighting, calm still composition"
    else:
        light = "even lighting, steady composition"

    return f"{palette}, {light}"


def build_prompt(
    content: str,
    *,
    emotion: str = NEUTRAL,
    valence: float = 0.0,
    arousal: float = 0.5,
    style: str = "artistic",
    tier: str = "affect",
) -> str:
    """Compose the generation prompt for a conditioning tier or floor."""
    if tier not in TIERS and tier not in FLOORS:
        raise ValueError(f"Unknown tier '{tier}', expected one of {TIERS + FLOORS}")

    style_desc = STYLE_PRESETS.get(style.lower(), STYLE_PRESETS["artistic"])
    mood = EMOTION_MOODS.get(emotion, EMOTION_MOODS[NEUTRAL])

    if tier == "raw":
        parts = [content, style_desc, QUALITY]
    elif tier == "naive":  # bare emotion word
        parts = [content, emotion, style_desc, QUALITY]
    elif tier == "emotion":
        parts = [content, mood, style_desc, QUALITY]
    elif tier == "affect":
        parts = [content, mood, va_descriptors(valence, arousal), style_desc, QUALITY]
    else:  # scene floor
        scene = EMOTION_SCENES.get(emotion, EMOTION_SCENES[NEUTRAL])
        parts = [scene, style_desc, QUALITY]

    return ", ".join(p for p in parts if p)
