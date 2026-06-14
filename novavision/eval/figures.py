"""Plots for the affect-recovery experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_confusion(matrix: np.ndarray, labels, path: str | Path, title: str = "") -> None:
    import matplotlib.pyplot as plt

    norm = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("recovered")
    ax.set_ylabel("intended")
    if title:
        ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_accuracy(tier_acc: dict[str, float], path: str | Path) -> None:
    import matplotlib.pyplot as plt

    tiers = list(tier_acc)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(tiers, [tier_acc[t] for t in tiers], color="#4c72b0")
    ax.set_ylabel("affect-recovery accuracy")
    ax.set_ylim(0, 1)
    for i, t in enumerate(tiers):
        ax.text(i, tier_acc[t] + 0.02, f"{tier_acc[t]:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_va_scatter(records, tier: str, path: str | Path) -> None:
    import matplotlib.pyplot as plt

    sub = [r for r in records if r["tier"] == tier]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter([r["intended_valence"] for r in sub], [r["recovered_valence"] for r in sub],
               alpha=0.6, label="valence", color="#c44e52")
    ax.scatter([r["intended_arousal"] for r in sub], [r["recovered_arousal"] for r in sub],
               alpha=0.6, label="arousal", color="#55a868")
    ax.plot([-1, 1], [-1, 1], "k--", linewidth=0.8)
    ax.set_xlabel("intended")
    ax.set_ylabel("recovered")
    ax.set_title(f"valence/arousal ({tier})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
