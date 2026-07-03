import math

import pytest

from novavision.affect.analyzer import EmotionAnalyzer
from novavision.affect.lexicon import AffectLexicon
from novavision.taxonomy import prior


class _Lex(AffectLexicon):
    """A tiny lexicon with controllable coverage for testing the blend."""

    def __init__(self, valence, arousal, coverage):
        self._v, self._a, self._c = valence, arousal, coverage

    def score(self, text):
        from novavision.affect.lexicon import AffectScore

        return AffectScore(self._v, self._a, self._c)


def _analyzer(scores, lex):
    a = EmotionAnalyzer(lexicon=lex)
    # Inject a fake classifier so no model is loaded (kwargs absorb truncation=True).
    a._classifier = lambda text, **kw: [[{"label": k, "score": v} for k, v in scores.items()]]
    return a


def test_blend_is_coverage_weighted():
    # coverage c=0.5 -> halfway between lexicon and the emotion prior.
    lex = _Lex(valence=1.0, arousal=1.0, coverage=0.5)
    a = _analyzer({"joy": 0.9, "anger": 0.1}, lex)
    res = a.analyze("anything")
    pv, pa = prior("joy")
    assert res.primary == "joy"
    assert math.isclose(res.valence, 0.5 * 1.0 + 0.5 * pv, abs_tol=1e-4)
    assert math.isclose(res.arousal, 0.5 * 1.0 + 0.5 * pa, abs_tol=1e-4)
    assert res.coverage == 0.5


def test_zero_coverage_falls_back_to_prior():
    # No lexicon hits -> affect is the emotion prior exactly (c=0).
    lex = _Lex(valence=0.0, arousal=0.0, coverage=0.0)
    a = _analyzer({"sadness": 0.8, "joy": 0.2}, lex)
    res = a.analyze("zzzz oovword")
    pv, pa = prior("sadness")
    assert (res.valence, res.arousal) == (round(pv, 4), round(pa, 4))


def test_full_coverage_is_capped():
    # A fully in-lexicon input must not discard the classifier prior entirely.
    lex = _Lex(valence=-0.4, arousal=0.9, coverage=1.0)
    a = _analyzer({"fear": 0.7, "joy": 0.3}, lex)
    res = a.analyze("all words known")
    pv, pa = prior("fear")
    assert math.isclose(res.valence, 0.8 * -0.4 + 0.2 * pv, abs_tol=1e-4)
    assert math.isclose(res.arousal, 0.8 * 0.9 + 0.2 * pa, abs_tol=1e-4)
    assert res.coverage == 1.0  # raw coverage is still reported


def test_scores_are_read_only():
    a = _analyzer({"joy": 1.0}, _Lex(0, 0, 0))
    res = a.analyze("hello")
    with pytest.raises(TypeError):
        res.scores["joy"] = 0.0


def test_empty_text_raises():
    a = _analyzer({"joy": 1.0}, _Lex(0, 0, 0))
    with pytest.raises(ValueError):
        a.analyze("   ")
