"""Valence/arousal scoring of text from an affect lexicon."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

_TOKEN = re.compile(r"[a-z][a-z']+")
_SUFFIXES = ("ing", "edly", "ed", "es", "s", "ly", "ness", "er", "est")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PATH = _REPO_ROOT / "data" / "lexicon" / "affect_lexicon.tsv"


@dataclass(frozen=True)
class AffectScore:
    valence: float
    arousal: float
    coverage: float  # fraction of tokens matched


def _variants(token: str):
    yield token
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) > len(suf) + 2:
            yield token[: -len(suf)]


class AffectLexicon:
    """Maps words to valence (-1..1) and arousal (0..1)."""

    def __init__(self, entries: dict[str, tuple[float, float]]):
        if not entries:
            raise ValueError("Lexicon is empty")
        self._entries = entries

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, word: str) -> bool:
        return word.lower() in self._entries

    def lookup(self, word: str) -> tuple[float, float] | None:
        for v in _variants(word.lower()):
            if v in self._entries:
                return self._entries[v]
        return None

    def score(self, text: str) -> AffectScore:
        tokens = _TOKEN.findall(text.lower())
        if not tokens:
            return AffectScore(0.0, 0.5, 0.0)

        hits = [self.lookup(t) for t in tokens]
        matched = [h for h in hits if h is not None]
        if not matched:
            return AffectScore(0.0, 0.5, 0.0)

        valence = sum(v for v, _ in matched) / len(matched)
        arousal = sum(a for _, a in matched) / len(matched)
        return AffectScore(
            round(valence, 4), round(arousal, 4), round(len(matched) / len(tokens), 4)
        )

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> AffectLexicon:
        path = Path(path or os.getenv("NOVAVISION_LEXICON") or _DEFAULT_PATH)
        entries: dict[str, tuple[float, float]] = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                word, valence, arousal = line.split("\t")
                if word == "word":  # header
                    continue
                entries[word.lower()] = (float(valence), float(arousal))
        return cls(entries)
