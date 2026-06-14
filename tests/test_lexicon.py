import pytest

from novavision.affect.lexicon import AffectLexicon


@pytest.fixture(scope="module")
def lex():
    return AffectLexicon.load()


def test_loads_entries(lex):
    assert len(lex) > 50


def test_positive_text_has_positive_valence(lex):
    assert lex.score("I feel so happy and joyful today").valence > 0.3


def test_negative_text_has_negative_valence(lex):
    assert lex.score("I am sad lonely and hopeless").valence < -0.3


def test_anger_has_high_arousal(lex):
    assert lex.score("furious rage outraged").arousal > 0.7


def test_empty_text_is_neutral(lex):
    score = lex.score("")
    assert score.coverage == 0.0
    assert score.valence == 0.0


def test_coverage_bounds(lex):
    score = lex.score("the happy dog ran quickly")
    assert 0.0 < score.coverage <= 1.0


def test_suffix_variant_lookup(lex):
    assert lex.lookup("loves") == lex.lookup("love")
