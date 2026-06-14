"""Emotion taxonomy, affect priors, and label mappings."""

from __future__ import annotations

EMOTIONS: tuple[str, ...] = (
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
)

NEUTRAL = "neutral"

# Circumplex priors (Russell, 1980): valence -1..1, arousal 0..1
EMOTION_PRIORS: dict[str, tuple[float, float]] = {
    "joy": (0.80, 0.70),
    "sadness": (-0.70, 0.30),
    "anger": (-0.60, 0.85),
    "fear": (-0.75, 0.80),
    "surprise": (0.35, 0.85),
    "disgust": (-0.60, 0.50),
    "neutral": (0.00, 0.30),
}

# Official GoEmotions -> Ekman grouping (Demszky et al., 2020)
GOEMOTIONS_TO_EKMAN: dict[str, str] = {
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "nervousness": "fear",
    "joy": "joy",
    "amusement": "joy",
    "approval": "joy",
    "excitement": "joy",
    "gratitude": "joy",
    "love": "joy",
    "optimism": "joy",
    "relief": "joy",
    "pride": "joy",
    "admiration": "joy",
    "desire": "joy",
    "caring": "joy",
    "sadness": "sadness",
    "disappointment": "sadness",
    "embarrassment": "sadness",
    "grief": "sadness",
    "remorse": "sadness",
    "surprise": "surprise",
    "realization": "surprise",
    "confusion": "surprise",
    "curiosity": "surprise",
    "neutral": "neutral",
}

# Zero-shot recovery prompts for CLIP
EMOTION_PROMPTS: dict[str, str] = {
    "anger": "an image that feels angry and intense",
    "disgust": "an image that feels disgusting and repulsive",
    "fear": "an image that feels fearful and threatening",
    "joy": "an image that feels joyful and happy",
    "neutral": "an image that feels calm and neutral",
    "sadness": "an image that feels sad and melancholic",
    "surprise": "an image that feels surprising and astonishing",
}

# Anchor prompts for valence/arousal probing
VALENCE_ANCHORS = ("a pleasant, positive, beautiful scene", "an unpleasant, negative, ugly scene")
AROUSAL_ANCHORS = ("an energetic, intense, exciting scene", "a calm, quiet, low-energy scene")


def to_ekman(label: str) -> str:
    """Map a GoEmotions label to an Ekman emotion."""
    return GOEMOTIONS_TO_EKMAN.get(label.lower().strip(), NEUTRAL)


def prior(emotion: str) -> tuple[float, float]:
    """Return the (valence, arousal) prior for an emotion."""
    return EMOTION_PRIORS.get(emotion.lower().strip(), (0.0, 0.5))
