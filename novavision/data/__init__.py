"""Benchmark data: builder and loader."""

from __future__ import annotations

import csv
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PATH = _REPO_ROOT / "data" / "affectbench_sample.csv"


def load_benchmark(path: str | Path | None = None) -> list[dict[str, str]]:
    """Load (text, emotion) rows from a benchmark CSV."""
    path = Path(path or SAMPLE_PATH)
    with open(path, encoding="utf-8") as fh:
        return [row for row in csv.DictReader(fh)]
