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

# Zero-shot recovery prompts for CLIP (ensemble per emotion, averaged)
EMOTION_PROMPTS: dict[str, tuple[str, ...]] = {
    "anger": (
        "an angry, intense image",
        "a furious, rage-filled scene",
        "a hostile, aggressive picture",
    ),
    "disgust": (
        "a disgusting, repulsive image",
        "a revolting, sickening scene",
        "a gross, nauseating picture",
    ),
    "fear": (
        "a fearful, threatening image",
        "a scary, terrifying scene",
        "an anxious, dread-filled picture",
    ),
    "joy": (
        "a joyful, happy image",
        "a cheerful, uplifting scene",
        "a delighted, blissful picture",
    ),
    "neutral": (
        "a calm, neutral image",
        "a plain, ordinary scene",
        "an emotionless, balanced picture",
    ),
    "sadness": (
        "a sad, melancholic image",
        "a sorrowful, gloomy scene",
        "a lonely, grieving picture",
    ),
    "surprise": (
        "a surprising, astonishing image",
        "a shocking, unexpected scene",
        "an amazed, startled picture",
    ),
}

# Ordered anchor ladders for graded valence/arousal (expected value over softmax)
VALENCE_LADDER: tuple[tuple[str, float], ...] = (
    ("a very unpleasant, miserable, repulsive scene", -1.0),
    ("an unpleasant, negative, gloomy scene", -0.5),
    ("a neutral, ordinary, plain scene", 0.0),
    ("a pleasant, positive, nice scene", 0.5),
    ("a very pleasant, beautiful, delightful scene", 1.0),
)
AROUSAL_LADDER: tuple[tuple[str, float], ...] = (
    ("a very calm, still, sleepy scene", 0.0),
    ("a calm, quiet, relaxed scene", 0.25),
    ("a moderate, steady scene", 0.5),
    ("an energetic, lively, active scene", 0.75),
    ("a very intense, frenzied, high-energy scene", 1.0),
)


def to_ekman(label: str) -> str:
    """Map a GoEmotions label to an Ekman emotion."""
    return GOEMOTIONS_TO_EKMAN.get(label.lower().strip(), NEUTRAL)


def prior(emotion: str) -> tuple[float, float]:
    """Return the (valence, arousal) prior for an emotion."""
    return EMOTION_PRIORS.get(emotion.lower().strip(), (0.0, 0.5))
