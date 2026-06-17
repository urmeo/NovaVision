"""Valence/arousal scoring of text from an affect lexicon."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

_TOKEN = re.compile(r"[a-z][a-z']+")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PATH = _REPO_ROOT / "data" / "lexicon" / "affect_lexicon.tsv"


@dataclass(frozen=True)
class AffectScore:
    valence: float
    arousal: float
    coverage: float  # fraction of tokens matched


def _variants(token: str):
    """Candidate lemmas in priority order.

    Only reliable inflections are stripped, with spelling restored (caring ->
    care, running -> run). Derivational suffixes that change meaning (-est,
    -er, -ness, -less) are deliberately left alone: missing a word is harmless,
    but mapping `hopeless` to `hope` or `honest` to `hon` corrupts the score.
    """
    yield token
    if token.endswith(("ies", "ied")) and len(token) > 4:
        yield token[:-3] + "y"  # cities -> city, tried -> try
        yield token[:-1]  # movies -> movie, lies -> lie
    elif token.endswith("ing") and len(token) > 5:
        stem = token[:-3]
        yield stem + "e"  # caring -> care
        yield stem  # playing -> play
        if len(stem) > 2 and stem[-1] == stem[-2]:
            yield stem[:-1]  # running -> run
    elif token.endswith("ed") and len(token) > 4:
        stem = token[:-2]
        yield stem + "e"  # closed -> close
        yield stem  # played -> play
        if len(stem) > 2 and stem[-1] == stem[-2]:
            yield stem[:-1]  # stopped -> stop
    elif token.endswith("es") and len(token) > 4:
        yield token[:-2]  # wishes -> wish
        yield token[:-1]  # likes -> like
    elif token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        yield token[:-1]  # dogs -> dog
    elif token.endswith("ly") and len(token) > 4:
        yield token[:-2]  # sadly -> sad


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
            for n, line in enumerate(fh, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split("\t")]
                if len(parts) < 3 or parts[0].lower() == "word":
                    continue
                word, valence, arousal = parts[:3]
                try:
                    entries[word.lower()] = (float(valence), float(arousal))
                except ValueError:
                    raise ValueError(f"{path}:{n} has non-numeric valence/arousal") from None
        return cls(entries)
