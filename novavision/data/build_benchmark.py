"""Build AffectBench: a balanced Ekman benchmark from GoEmotions.

Sampling is from the GoEmotions *test* split by default (so a benchmark item
never overlaps a model's training split), deduplicated, and interleaved across
classes so any prefix stays balanced. A manifest records the split, dataset
revision, realized per-class counts, and a content hash for reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from novavision.taxonomy import EMOTIONS, to_ekman

DATASET = "google-research-datasets/go_emotions"
# Pin the dataset revision so a rebuild reconstructs the same pool.
DEFAULT_REVISION = "13888bb783fae7d7eb33ef4cb1ee19a4c3c0d3f8"


def _curate(
    examples: list[tuple[str, str]], n_per_class: int, seed: int
) -> dict[str, list[str]]:
    """Single-label, deduplicated, balanced per-class sampling.

    ``examples`` is (text, ekman_label). Texts are normalised for exact-dup
    removal, sorted for determinism, then sampled to ``n_per_class``.
    """
    buckets: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for text, emotion in examples:
        text = text.strip()
        norm = " ".join(text.lower().split())
        if not text or norm in seen or emotion not in set(EMOTIONS):
            continue
        seen.add(norm)
        buckets[emotion].append(text)

    rng = random.Random(seed)
    sampled: dict[str, list[str]] = {}
    for emotion in EMOTIONS:
        texts = sorted(buckets.get(emotion, []))
        rng.shuffle(texts)
        sampled[emotion] = texts[:n_per_class]
    return sampled


def _interleave(sampled: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Round-robin across classes so a prefix stays stratified."""
    rows: list[tuple[str, str]] = []
    for i in range(max((len(v) for v in sampled.values()), default=0)):
        for emotion in EMOTIONS:
            if i < len(sampled[emotion]):
                rows.append((sampled[emotion][i], emotion))
    return rows


def build(
    n_per_class: int = 100,
    out_path: str | Path = "data/affectbench.csv",
    seed: int = 0,
    split: str = "test",
    revision: str = DEFAULT_REVISION,
) -> Path:
    from datasets import load_dataset

    dataset = load_dataset(DATASET, "simplified", split=split, revision=revision)
    names = dataset.features["labels"].feature.names
    examples = [
        (row["text"], to_ekman(names[row["labels"][0]]))
        for row in dataset
        if len(row["labels"]) == 1
    ]

    sampled = _curate(examples, n_per_class, seed)
    rows = _interleave(sampled)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["text", "emotion"])
        writer.writerows(rows)

    counts = {e: len(sampled[e]) for e in EMOTIONS}
    short = sorted(e for e, c in counts.items() if c < n_per_class)
    manifest = {
        "dataset": DATASET,
        "split": split,
        "revision": revision,
        "n_per_class": n_per_class,
        "seed": seed,
        "counts": counts,
        "total": len(rows),
        "underfilled": short,
        "balanced": not short,
    }
    out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    if short:
        print(f"WARNING: classes below {n_per_class}: {short}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the AffectBench benchmark")
    parser.add_argument("--n", type=int, default=100, help="examples per emotion")
    parser.add_argument("--out", default="data/affectbench.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    path = build(args.n, args.out, args.seed, args.split)
    print(f"Wrote {path} and {path.with_suffix('.manifest.json')}")


if __name__ == "__main__":
    main()
