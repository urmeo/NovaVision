import pytest

from novavision.eval.metrics import accuracy, confusion_matrix, macro_f1, pearson


def test_accuracy():
    assert accuracy(["a", "b", "c"], ["a", "b", "x"]) == 2 / 3
    assert accuracy([], []) == 0.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        accuracy(["a", "b"], ["a"])


def test_macro_f1_perfect():
    labels = ["a", "b"]
    assert macro_f1(["a", "b", "a"], ["a", "b", "a"], labels) == 1.0


def test_confusion_matrix():
    cm = confusion_matrix(["a", "a", "b"], ["a", "b", "b"], ["a", "b"])
    assert cm.tolist() == [[1, 1], [0, 1]]


def test_pearson():
    assert round(pearson([1, 2, 3], [2, 4, 6]), 4) == 1.0
    assert pearson([1, 1, 1], [1, 2, 3]) == 0.0
