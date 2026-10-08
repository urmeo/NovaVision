"Classification and correlation metrics (numpy only)."

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np


def _check(y_true: Sequence[object], y_pred: Sequence[object]) -> None:
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
    "Shuffled-label control for circularity: is recovery above random targets?"
    _check(y_true, y_pred)
    yt = np.asarray(list(y_true))
    yp = np.asarray(list(y_pred))
    if len(yt) < 2:
        nan = float("nan")
        return {"accuracy": nan, "null_mean": nan, "null_ci": [nan, nan], "p_value": nan}
    observed = float(np.mean(yt == yp))
    rng = np.random.default_rng(seed)
    null = np.array([float(np.mean(rng.permutation(yt) == yp)) for _ in range(n)])
    lo, hi = np.quantile(null, [0.025, 0.975])
    p = float((1 + np.sum(null >= observed)) / (n + 1))
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
    "Accuracy of the degenerate classifier that always predicts one label."
    if not y_true:
        return float("nan")
    counts = Counter(y_true)
    return max(counts.values()) / len(y_true)


def prediction_collapse(y_pred: Sequence[str]) -> dict:
    "How concentrated a probe's predictions are, a degeneracy diagnostic."
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


def rogan_gladen(apparent: float, sensitivity: float, specificity: float) -> float:
    "Prevalence corrected for an imperfect test (Rogan & Gladen, 1978)."
    denom = sensitivity + specificity - 1.0

    if not np.isfinite(denom) or denom <= 0 or not np.isfinite(apparent):
        return float("nan")
    return float(np.clip((apparent + specificity - 1.0) / denom, 0.0, 1.0))


def cohens_h(p1: float, p2: float) -> float:
    """Effect size for two proportions (arcsine-transformed difference)."""
    if not (np.isfinite(p1) and np.isfinite(p2)):
        return float("nan")
    phi = lambda p: 2.0 * np.arcsin(np.sqrt(np.clip(p, 0.0, 1.0)))  # noqa: E731
    return float(phi(p1) - phi(p2))


def holm_bonferroni(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    "Family-wise correction over a set of named p-values (Holm, 1979)."
    finite = {k: v for k, v in pvalues.items() if isinstance(v, (int, float)) and v == v}
    ordered = sorted(finite, key=lambda k: finite[k])
    m = len(ordered)
    out: dict[str, dict] = {}
    running = 0.0
    for i, name in enumerate(ordered):
        adj = min(1.0, (m - i) * finite[name])
        running = max(running, adj)
        out[name] = {"p_adjusted": round(running, 4), "reject": running <= alpha}
    for name in pvalues:
        if name not in out:
            out[name] = {"p_adjusted": None, "reject": False}
    return out


def mae(x: Sequence[float], y: Sequence[float]) -> float:
    """Mean absolute error, an interpretable companion to the VA correlations."""
    _check(x, y)
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if len(xa) == 0:
        return float("nan")
    return float(np.mean(np.abs(xa - ya)))


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    _check(x, y)
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if len(xa) < 2 or xa.std() == 0 or ya.std() == 0:
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Rank correlation, robust to the compressed VA scale."""
    _check(x, y)
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if len(xa) < 2:
        return float("nan")
    return pearson(_rank(xa).tolist(), _rank(ya).tolist())


def _rank(a: np.ndarray) -> np.ndarray:
    order = a.argsort()
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)

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
    "Percentile bootstrap CI for a correlation, resampling item pairs."
    if method not in ("spearman", "pearson"):
        raise ValueError(f"unknown method '{method}', expected 'spearman' or 'pearson'")
    corr = spearman if method == "spearman" else pearson
    _check(x, y)
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if len(xa) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(xa), size=len(xa))
        r = corr(xa[idx].tolist(), ya[idx].tolist())
        if r == r:
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
    "Paired bootstrap on the per-item difference a - b."
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

    p = float((1 + np.sum(np.abs(centered) >= abs(diff.mean()))) / (n + 1))
    return {
        "mean_diff": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": p,
    }
