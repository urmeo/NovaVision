"""Turn results.json into a markdown metrics table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COLUMNS = [
    ("accuracy", "Accuracy"),
    ("macro_f1", "Macro-F1"),
    ("valence_r", "Valence r"),
    ("arousal_r", "Arousal r"),
    ("clip_t", "CLIP-T"),
]


def to_markdown(metrics: dict) -> str:
    header = "| Tier | " + " | ".join(label for _, label in COLUMNS) + " |"
    rule = "|" + "---|" * (len(COLUMNS) + 1)
    lines = [header, rule]
    for tier, values in metrics.items():
        cells = " | ".join(f"{values[key]:.3f}" for key, _ in COLUMNS)
        lines.append(f"| {tier} | {cells} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render results as a table")
    parser.add_argument("--results", default="results/results.json")
    parser.add_argument("--out", default="paper/tables.md")
    args = parser.parse_args()

    metrics = json.loads(Path(args.results).read_text())["metrics"]
    table = to_markdown(metrics)
    Path(args.out).write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
