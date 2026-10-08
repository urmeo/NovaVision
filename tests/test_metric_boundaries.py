"""Reject incomplete paired observations before statistical calculations."""

import pytest

from novavision.eval.metrics import bootstrap_corr_ci, mae, pearson, permutation_test, spearman


@pytest.mark.parametrize("metric", [mae, pearson, spearman, bootstrap_corr_ci])
@pytest.mark.parametrize("x,y", [([0.1, 0.2, 0.3], [0.1]), ([0.1], [0.1, 0.2, 0.3]), ([], [0.1])])
def test_numeric_metrics_reject_unpaired_observations(metric, x, y):
    with pytest.raises(ValueError, match="same length"):
        metric(x, y)


@pytest.mark.parametrize(
    "truth,predicted",
    [(["joy", "sadness", "fear"], ["joy"]), (["joy"], ["joy", "fear"]), ([], ["joy"])],
)
def test_permutation_test_rejects_unpaired_labels(truth, predicted):
    with pytest.raises(ValueError, match="same length"):
        permutation_test(truth, predicted, n=10)
