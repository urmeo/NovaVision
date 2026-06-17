import math

import pytest

from novavision.eval.metrics import (
    accuracy,
    bootstrap_ci,
    bootstrap_corr_ci,
    confusion_matrix,
    macro_f1,
    majority_baseline,
    paired_bootstrap_test,
    pearson,
    permutation_test,
    prediction_collapse,
    spearman,
)


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


def test_bootstrap_ci_brackets_mean():
    lo, hi = bootstrap_ci([0, 1, 1, 1, 0, 1, 1, 0, 1, 1])
    assert lo <= 0.7 <= hi
    assert 0.0 <= lo <= hi <= 1.0


def test_paired_test_detects_difference():
    a = [1, 1, 1, 1, 1, 1, 1, 1]
    b = [0, 0, 0, 0, 0, 0, 0, 0]
    res = paired_bootstrap_test(a, b)
    assert res["mean_diff"] == 1.0
    assert 0 < res["p_value"] < 0.05  # +1 smoothing: never exactly 0


def test_paired_test_no_difference():
    a = [1, 0, 1, 0, 1, 0]
    res = paired_bootstrap_test(a, a)
    assert res["mean_diff"] == 0.0
    assert res["p_value"] >= 0.05


def test_majority_baseline():
    # 3 of 5 are 'neutral' -> a probe always saying 'neutral' scores 0.6.
    assert majority_baseline(["neutral", "neutral", "neutral", "joy", "anger"]) == 0.6
    # Balanced labels collapse the baseline onto chance.
    assert majority_baseline(["a", "b", "c", "d"]) == 0.25
    assert math.isnan(majority_baseline([]))


def test_prediction_collapse_flags_degenerate_probe():
    # Probe that says 'neutral' for 9 of 10 images.
    c = prediction_collapse(["neutral"] * 9 + ["joy"])
    assert c["label"] == "neutral"
    assert c["rate"] == 0.9
    assert c["distinct"] == 2
    # A healthy, varied probe: 2 of each over 6 -> rate 1/3, all 3 labels used.
    healthy = prediction_collapse(["a", "b", "c", "a", "b", "c"])
    assert round(healthy["rate"], 4) == 0.3333 and healthy["distinct"] == 3


def test_permutation_test_detects_real_signal():
    labels = ["a", "b", "c", "d", "e", "f", "g"] * 4
    # Perfect recovery -> far above the shuffled-label null, tiny p.
    perfect = permutation_test(labels, labels, n=500)
    assert perfect["accuracy"] == 1.0
    assert perfect["p_value"] < 0.01
    assert perfect["null_mean"] < 0.3  # random targets ~ chance


def test_permutation_test_flags_circular_chance():
    # Predictions ignore the target (always 'a') -> recovery == shuffled baseline.
    truth = ["a", "b", "c", "d", "e", "f", "g"] * 4
    pred = ["a"] * len(truth)
    res = permutation_test(truth, pred, n=500)
    assert res["p_value"] > 0.2  # indistinguishable from random targets
    assert math.isnan(permutation_test(["a"], ["a"])["p_value"])


def test_bootstrap_corr_ci_brackets_and_widens_at_small_n():
    # Perfect monotonic relation -> CI sits high and is valid.
    lo, hi = bootstrap_corr_ci([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6])
    assert -1.0 <= lo <= hi <= 1.0
    assert hi == 1.0
    # Too few points to bootstrap a correlation -> nan, not a fake number.
    nlo, nhi = bootstrap_corr_ci([1, 2], [2, 1])
    assert math.isnan(nlo) and math.isnan(nhi)
