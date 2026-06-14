from collections import Counter

from novavision.data import load_benchmark
from novavision.taxonomy import EMOTIONS


def test_sample_is_balanced():
    rows = load_benchmark()
    counts = Counter(row["emotion"] for row in rows)
    assert set(counts) == set(EMOTIONS)
    assert set(counts.values()) == {8}


def test_rows_have_text():
    for row in load_benchmark():
        assert row["text"].strip()
        assert row["emotion"] in EMOTIONS
