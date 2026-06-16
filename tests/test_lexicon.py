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


def test_inflections_restore_spelling():
    lex = AffectLexicon({"love": (0.9, 0.7), "care": (0.6, 0.4), "run": (0.1, 0.8)})
    assert lex.lookup("loving") == (0.9, 0.7)
    assert lex.lookup("caring") == (0.6, 0.4)
    assert lex.lookup("running") == (0.1, 0.8)


def test_ies_of_ie_nouns():
    lex = AffectLexicon({"movie": (0.3, 0.5), "city": (0.1, 0.4)})
    assert lex.lookup("movies") == (0.3, 0.5)  # not 'movy'
    assert lex.lookup("cities") == (0.1, 0.4)


def test_meaning_changing_suffixes_not_stripped():
    # `hon`/`inter` exist; honest/interest must not collapse onto them.
    lex = AffectLexicon({"hon": (0.5, 0.5), "inter": (0.0, 0.5), "hope": (0.8, 0.5)})
    assert lex.lookup("honest") is None
    assert lex.lookup("interest") is None
    assert lex.lookup("hopeless") is None


def test_malformed_lexicon_line_raises(tmp_path):
    bad = tmp_path / "bad.tsv"
    bad.write_text("word\tvalence\tarousal\nhappy\thigh\t0.5\n")
    with pytest.raises(ValueError):
        AffectLexicon.load(bad)
