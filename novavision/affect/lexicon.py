"""Valence/arousal scoring of text from an affect lexicon."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

_TOKEN = re.compile(r"[a-z][a-z']+")

# Function words are never scored: with a research lexicon (e.g. Warriner) they
# match ("does" even stems to "doe", the deer) and dilute or distort the score.
# Kept out of the coverage denominator too, so coverage means "fraction of
# content words matched". Negators are excluded the same way but drive the
# negation flip below.
STOPWORDS = frozenset(
    """
a an the and or but if because as of at by for with about into onto over under
again then once here there when where why how all any both each few more most
other some such than too very just own same so that this these those me my mine
we us our ours you your yours he him his she her hers it its they them their
theirs what which who whom am is are was were be been being have has had having
do does did doing done will would shall should can could may might must ought
to from in on out off it's i'm i've i'll you're we're they're he's she's that's
there's
""".split()
)

NEGATORS = frozenset(
    "not no never nor cannot can't don't doesn't didn't isn't wasn't aren't "
    "weren't won't wouldn't couldn't shouldn't ain't without".split()
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PATH = _REPO_ROOT / "data" / "lexicon" / "affect_lexicon.tsv"


@dataclass(frozen=True)
class AffectScore:
    valence: float
    arousal: float
    coverage: float  # fraction of content words matched (stopwords excluded)


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
            if v in STOPWORDS or v in NEGATORS:
                continue  # "wills" must not stem into a scored "will"
            if v in self._entries:
                return self._entries[v]
        return None

    def score(self, text: str) -> AffectScore:
        # Smart quotes (U+2019) would otherwise split "can’t" into can|t and
        # silently defeat the negation flip on text typed from phones.
        tokens = _TOKEN.findall(text.lower().replace("’", "'"))
        matched: list[tuple[float, float]] = []
        n_content = 0
        for i, tok in enumerate(tokens):
            if tok in STOPWORDS or tok in NEGATORS:
                continue
            n_content += 1
            hit = self.lookup(tok)
            if hit is None:
                continue
            v, a = hit
            # Two-token lookback negation: "not happy" must not score as happy.
            # Scope and degree ("hardly", "barely") are deliberately unmodeled.
            if any(t in NEGATORS for t in tokens[max(0, i - 2) : i]):
                v = -v
            matched.append((v, a))
        if not matched:
            return AffectScore(0.0, 0.5, 0.0)

        valence = sum(v for v, _ in matched) / len(matched)
        arousal = sum(a for _, a in matched) / len(matched)
        return AffectScore(round(valence, 4), round(arousal, 4), round(len(matched) / n_content, 4))

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
                # Only the exact header row is skipped: a lexicon ENTRY for the
                # word "word" (present in research norms) must not be dropped.
                if [p.lower() for p in parts[:3]] == ["word", "valence", "arousal"]:
                    continue
                if len(parts) < 3:
                    raise ValueError(f"{path}:{n} has fewer than 3 tab-separated columns")
                word, valence, arousal = parts[:3]
                try:
                    entries[word.lower()] = (float(valence), float(arousal))
                except ValueError:
                    raise ValueError(f"{path}:{n} has non-numeric valence/arousal") from None
        return cls(entries)
