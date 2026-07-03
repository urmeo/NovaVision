"""Benchmark loading, content bank, and provenance checks."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from novavision.taxonomy import EMOTIONS

_REPO_ROOT = Path(__file__).resolve().parents[2]
CONTENT_BANK_PATH = _REPO_ROOT / "data" / "content_bank.txt"

_EMOTIONS = set(EMOTIONS)


def load_benchmark(path: str | Path) -> list[dict[str, str]]:
    """Load validated (text, emotion) rows from a benchmark CSV.

    The path is required: there is no default, so a run can never silently
    fall back to a sample. Every row is checked for a non-empty text and a
    known emotion, so a malformed benchmark fails loudly here, not mid-run.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark not found: {path}")

    rows: list[dict[str, str]] = []
    with open(path, encoding="utf-8-sig") as fh:  # tolerate an Excel/Sheets BOM
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or {"text", "emotion"} - set(reader.fieldnames):
            raise ValueError(f"{path} must have 'text' and 'emotion' columns")
        for i, row in enumerate(reader, start=2):
            text = (row.get("text") or "").strip()
            emotion = (row.get("emotion") or "").strip().lower()
            if not text:
                raise ValueError(f"{path}:{i} has empty text")
            if emotion not in _EMOTIONS:
                raise ValueError(f"{path}:{i} unknown emotion '{emotion}'")
            rows.append({"text": text, "emotion": emotion})
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def load_content_bank(path: str | Path | None = None) -> list[str]:
    """Load neutral, emotion-independent content prompts.

    These subjects carry no affect on their own, so the same content can be
    rendered under every intended emotion. That decoupling is what lets the
    benchmark attribute recovered emotion to the conditioning, not the scene.
    """
    path = Path(path or CONTENT_BANK_PATH)
    items: list[str] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key = " ".join(line.lower().split())
            if key in seen:
                # A duplicate silently double-weights that scene in every tier.
                raise ValueError(f"{path}:{n} duplicate subject: '{line}'")
            seen.add(key)
            items.append(line)
    if not items:
        raise ValueError(f"{path} has no content prompts")
    return items


def sha256(path: str | Path) -> str:
    """Content hash of a benchmark, for the run manifest."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
