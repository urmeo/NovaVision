"""Discrete emotion classification with grounded valence/arousal."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from novavision.affect.lexicon import AffectLexicon
from novavision.config import EMOTION_MODEL, EMOTION_REVISION
from novavision.taxonomy import prior

DEFAULT_MODEL = EMOTION_MODEL

# A fully in-lexicon input ("happy") must not discard the classifier prior
# entirely: the lexicon reads words, not context, so it never gets full weight.
MAX_LEXICON_BLEND = 0.8


@dataclass(frozen=True)
class EmotionAnalysis:
    primary: str
    confidence: float
    valence: float
    arousal: float
    coverage: float
    scores: Mapping[str, float]

    def __post_init__(self):
        # frozen=True does not deep-freeze; make the scores read-only too.
        object.__setattr__(self, "scores", MappingProxyType(dict(self.scores)))


class EmotionAnalyzer:
    """DistilRoBERTa emotion classifier + lexicon-grounded affect."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        lexicon: AffectLexicon | None = None,
        revision: str | None = None,
        coverage_override: float | None = None,
    ):
        self.model_name = model_name
        # The pinned revision is a commit of DEFAULT_MODEL only; applying it to a
        # swapped model would request a commit that does not exist in that repo.
        self.revision = revision or (EMOTION_REVISION if model_name == DEFAULT_MODEL else None)
        # Ablation hook: force the lexicon/prior blend weight (0 = prior only,
        # 1 = lexicon only) instead of using measured coverage. None = normal.
        if coverage_override is not None and not 0.0 <= coverage_override <= 1.0:
            raise ValueError("coverage_override must be in [0, 1]")
        self.coverage_override = coverage_override
        self._lexicon = lexicon
        self._classifier = None
        # Independent resources get independent locks: neither load can ever
        # wait on, or re-enter through, the other (Lock is not reentrant).
        self._lex_lock = threading.Lock()
        self._clf_lock = threading.Lock()

    @property
    def lexicon(self) -> AffectLexicon:
        if self._lexicon is None:
            with self._lex_lock:  # double-checked: one load under concurrent first calls
                if self._lexicon is None:
                    self._lexicon = AffectLexicon.load()
        return self._lexicon

    @property
    def classifier(self):
        if self._classifier is None:
            with self._clf_lock:
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

        # Truncate at the model limit: MAX_TEXT (2000 chars) can exceed 512 tokens,
        # and an untruncated overflow raises inside transformers.
        raw = self.classifier(text, truncation=True)[0]
        scores = {r["label"].lower(): float(r["score"]) for r in raw}
        primary = max(scores, key=lambda k: scores[k])

        affect = self.lexicon.score(text)
        pv, pa = prior(primary)
        c = (
            self.coverage_override
            if self.coverage_override is not None
            else min(affect.coverage, MAX_LEXICON_BLEND)
        )
        valence = c * affect.valence + (1 - c) * pv
        arousal = c * affect.arousal + (1 - c) * pa

        return EmotionAnalysis(
            primary=primary,
            confidence=scores[primary],
            valence=round(valence, 4),
            arousal=round(arousal, 4),
            coverage=affect.coverage,  # raw diagnostic; the blend weight is capped
            scores=scores,
        )
