"""Classification and correlation metrics (numpy only).

Undefined cases return ``nan`` rather than ``0.0`` so a degenerate input
(empty tier, constant signal) is never mistaken for a genuine zero result.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np


def _check(y_true: Sequence[str], y_pred: Sequence[str]) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")


def _check_labels(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> None:
    known = set(labels)
    unknown = (set(y_true) | set(y_pred)) - known
    if unknown:
        raise ValueError(f"labels {sorted(unknown)} not in {sorted(known)}")


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    _check(y_true, y_pred)
    if not y_true:
        return float("nan")
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return correct / len(y_true)


def permutation_test(
    y_true: Sequence[str], y_pred: Sequence[str], *, n: int = 2000, seed: int = 0
) -> dict[str, float | list[float]]:
    """Shuffled-label control for circularity: is recovery above random targets?

    The recovery metric writes an emotion into the prompt and reads one back, so a
    high score could reflect prompt/probe agreement rather than image content. This
    permutes the intended labels many times and recomputes accuracy to build the
    null distribution of recovery under *random* targets, the circularity
    baseline. Returns the observed accuracy, the null mean and 95% interval, and a
    one-sided p-value ``P(null >= observed)``. A small p means recovery tracks the
    real image/label correspondence, not chance agreement; ``p`` near 1 means the
    headline is indistinguishable from shuffled labels.

    The fixed default ``seed`` is deliberate (common random numbers): every tier
    is scored against the same permutation draws of the shared ``y_true``, so
    tier p-values differ only through the predictions, not Monte-Carlo noise.
    """
    yt = np.asarray(list(y_true))
    yp = np.asarray(list(y_pred))
    if len(yt) < 2:
        nan = float("nan")
        return {"accuracy": nan, "null_mean": nan, "null_ci": [nan, nan], "p_value": nan}
    observed = float(np.mean(yt == yp))
    rng = np.random.default_rng(seed)
    null = np.array([float(np.mean(rng.permutation(yt) == yp)) for _ in range(n)])
    lo, hi = np.quantile(null, [0.025, 0.975])
    p = float((1 + np.sum(null >= observed)) / (n + 1))  # +1 smoothing
    return {
        "accuracy": round(observed, 4),
        "null_mean": round(float(null.mean()), 4),
        "null_ci": [round(float(lo), 4), round(float(hi), 4)],
        "p_value": round(p, 4),
    }


def confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> np.ndarray:
    _check(y_true, y_pred)
    _check_labels(y_true, y_pred, labels)
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[index[t], index[p]] += 1
    return matrix


def majority_baseline(y_true: Sequence[str]) -> float:
    """Accuracy of the degenerate classifier that always predicts one label.

    A probe that has collapsed to a single output (the failure mode the floors
    are meant to expose) scores exactly this. Recovery is only evidence of signal
    if it clears this baseline, not merely chance. On balanced labels the two
    coincide, which is precisely why a chance-level result is uninformative.
    """
    if not y_true:
        return float("nan")
    counts = Counter(y_true)
    return max(counts.values()) / len(y_true)


def prediction_collapse(y_pred: Sequence[str]) -> dict:
    """How concentrated a probe's predictions are, a degeneracy diagnostic.

    Returns the probe's most frequent output, the fraction of items it was
    assigned (``rate``: 1.0 means the probe predicted one label for everything),
    and the count of ``distinct`` labels used. A high rate / low distinct count
    means a reported accuracy reflects the probe's pathology, not the generator.
    """
    if not y_pred:
        return {"label": "", "rate": float("nan"), "distinct": 0}
    counts = Counter(y_pred)
    label, top = counts.most_common(1)[0]
    return {"label": label, "rate": top / len(y_pred), "distinct": len(counts)}


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
    """Macro-F1 averaged over the labels actually present in y_true."""
    _check(y_true, y_pred)
    _check_labels(y_true, y_pred, labels)
    present = [label for label in labels if label in set(y_true)]
    if not present:
        return float("nan")
    scores = []
    for label in present:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(f1)
    return float(np.mean(scores))


def cohen_kappa(a: Sequence[str], b: Sequence[str], labels: Sequence[str]) -> float:
    """Chance-corrected agreement between two label sets (e.g. human vs probe)."""
    _check(a, b)
    _check_labels(a, b, labels)
    if not a:
        return float("nan")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in labels)
    if pe == 1.0:
        return float("nan")
    return float((po - pe) / (1 - pe))


def mae(x: Sequence[float], y: Sequence[float]) -> float:
    """Mean absolute error, an interpretable companion to the VA correlations."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if len(xa) == 0:
        return float("nan")
    return float(np.mean(np.abs(xa - ya)))


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if len(xa) < 2 or xa.std() == 0 or ya.std() == 0:
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Rank correlation, robust to the compressed VA scale."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if len(xa) < 2:
        return float("nan")
    return pearson(_rank(xa).tolist(), _rank(ya).tolist())


def _rank(a: np.ndarray) -> np.ndarray:
    order = a.argsort()
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    # average ties
    _, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    return (sums / counts)[inverse]


def bootstrap_corr_ci(
    x: Sequence[float],
    y: Sequence[float],
    *,
    method: str = "spearman",
    n: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a correlation, resampling item pairs.

    A point correlation on a handful of items (e.g. n=14) is noise dressed as a
    measurement; the CI makes that uncertainty explicit so the reader is not
    invited to over-read three reported decimals.
    """
    corr = spearman if method == "spearman" else pearson
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if len(xa) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(xa), size=len(xa))
        r = corr(xa[idx].tolist(), ya[idx].tolist())
        if r == r:  # drop degenerate resamples (constant signal -> nan)
            vals.append(r)
    if len(vals) < 2:
        return float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def bootstrap_ci(
    values: Sequence[float], *, n: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean, resampling over items."""
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, len(arr), size=(n, len(arr)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap_test(
    a: Sequence[float], b: Sequence[float], *, n: int = 2000, seed: int = 0
) -> dict[str, float]:
    """Paired bootstrap on the per-item difference a - b.

    Returns the observed mean difference, its 95% CI, and a two-sided p-value
    for H0: mean difference = 0. Items are paired, so the two conditions must
    be aligned element-wise (same item, same seed).
    """
    da = np.asarray(a, dtype=float)
    db = np.asarray(b, dtype=float)
    if len(da) != len(db):
        raise ValueError("a and b must be the same length")
    diff = da - db
    if len(diff) < 2:
        return {
            "mean_diff": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value": float("nan"),
        }
    rng = np.random.default_rng(seed)
    resampled = diff[rng.integers(0, len(diff), size=(n, len(diff)))].mean(axis=1)
    lo, hi = np.quantile(resampled, [0.025, 0.975])
    centered = resampled - resampled.mean()
    # +1 smoothing
    p = float((1 + np.sum(np.abs(centered) >= abs(diff.mean()))) / (n + 1))
    return {
        "mean_diff": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": p,
    }
