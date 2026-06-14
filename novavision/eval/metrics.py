"""Classification and correlation metrics (numpy only)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _check(y_true: Sequence[str], y_pred: Sequence[str]) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    _check(y_true, y_pred)
    if not y_true:
        return 0.0
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return correct / len(y_true)


def confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> np.ndarray:
    _check(y_true, y_pred)
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in index and p in index:
            matrix[index[t], index[p]] += 1
    return matrix


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
    _check(y_true, y_pred)
    scores = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(f1)
    return float(np.mean(scores)) if scores else 0.0


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
