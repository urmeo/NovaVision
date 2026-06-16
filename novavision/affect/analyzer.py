"""Discrete emotion classification with grounded valence/arousal."""

from __future__ import annotations

from dataclasses import dataclass

from novavision.affect.lexicon import AffectLexicon
from novavision.config import EMOTION_REVISION
from novavision.taxonomy import prior

DEFAULT_MODEL = "j-hartmann/emotion-english-distilroberta-base"


@dataclass(frozen=True)
class EmotionAnalysis:
    primary: str
    confidence: float
    valence: float
    arousal: float
    coverage: float
    scores: dict[str, float]


class EmotionAnalyzer:
    """DistilRoBERTa emotion classifier + lexicon-grounded affect."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        lexicon: AffectLexicon | None = None,
        revision: str | None = EMOTION_REVISION,
    ):
        self.model_name = model_name
        self.revision = revision
        self._lexicon = lexicon
        self._classifier = None

    @property
    def lexicon(self) -> AffectLexicon:
        if self._lexicon is None:
            self._lexicon = AffectLexicon.load()
        return self._lexicon

    @property
    def classifier(self):
        if self._classifier is None:
            from transformers import pipeline

            self._classifier = pipeline(
                "text-classification",
                model=self.model_name,
                revision=self.revision,
                top_k=None,
                device=-1,
            )
        return self._classifier

    def analyze(self, text: str) -> EmotionAnalysis:
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        raw = self.classifier(text)[0]
        scores = {r["label"].lower(): float(r["score"]) for r in raw}
        primary = max(scores, key=lambda k: scores[k])

        affect = self.lexicon.score(text)
        pv, pa = prior(primary)
        c = affect.coverage
        valence = c * affect.valence + (1 - c) * pv
        arousal = c * affect.arousal + (1 - c) * pa

        return EmotionAnalysis(
            primary=primary,
            confidence=scores[primary],
            valence=round(valence, 4),
            arousal=round(arousal, 4),
            coverage=c,
            scores=scores,
        )
