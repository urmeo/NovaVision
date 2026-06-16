"""Classification and correlation metrics (numpy only).

Undefined cases return ``nan`` rather than ``0.0`` so a degenerate input
(empty tier, constant signal) is never mistaken for a genuine zero result.
"""

from __future__ import annotations

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
    return pearson(_rank(xa), _rank(ya))


def _rank(a: np.ndarray) -> np.ndarray:
    order = a.argsort()
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    # Average ties so equal values share a rank.
    _, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    return (sums / counts)[inverse]
