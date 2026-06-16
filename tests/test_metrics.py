import math

import pytest

from novavision.eval.metrics import accuracy, confusion_matrix, macro_f1, pearson, spearman


def test_accuracy():
    assert accuracy(["a", "b", "c"], ["a", "b", "x"]) == 2 / 3
    assert math.isnan(accuracy([], []))


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        accuracy(["a", "b"], ["a"])


def test_macro_f1_perfect():
    labels = ["a", "b"]
    assert macro_f1(["a", "b", "a"], ["a", "b", "a"], labels) == 1.0


def test_macro_f1_only_present_labels():
    # 'c' never appears in y_true, so it does not drag the average to 0.
    assert macro_f1(["a", "b"], ["a", "b"], ["a", "b", "c"]) == 1.0


def test_confusion_matrix():
    cm = confusion_matrix(["a", "a", "b"], ["a", "b", "b"], ["a", "b"])
    assert cm.tolist() == [[1, 1], [0, 1]]


def test_out_of_label_raises():
    with pytest.raises(ValueError):
        confusion_matrix(["a", "z"], ["a", "a"], ["a", "b"])


def test_pearson():
    assert round(pearson([1, 2, 3], [2, 4, 6]), 4) == 1.0
    assert math.isnan(pearson([1, 1, 1], [1, 2, 3]))
    assert math.isnan(pearson([1.0], [2.0]))


def test_spearman_monotonic():
    assert round(spearman([1, 2, 3, 4], [1, 4, 9, 16]), 4) == 1.0
