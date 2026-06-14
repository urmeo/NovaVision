"""Build a balanced Ekman benchmark from GoEmotions."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from novavision.taxonomy import EMOTIONS, to_ekman


def build(
    n_per_class: int = 100, out_path: str | Path = "data/affectbench.csv", seed: int = 0
) -> Path:
    from datasets import load_dataset

    dataset = load_dataset("google-research-datasets/go_emotions", "simplified", split="train")
    names = dataset.features["labels"].feature.names

    buckets: dict[str, list[str]] = defaultdict(list)
    for row in dataset:
        if len(row["labels"]) != 1:  # single-label only
            continue
        ekman = to_ekman(names[row["labels"][0]])
        text = row["text"].strip()
        if text:
            buckets[ekman].append(text)

    rng = random.Random(seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["text", "emotion"])
        for emotion in EMOTIONS:
            texts = buckets.get(emotion, [])
            rng.shuffle(texts)
            for text in texts[:n_per_class]:
                writer.writerow([text, emotion])
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the AffectBench benchmark")
    parser.add_argument("--n", type=int, default=100, help="examples per emotion")
    parser.add_argument("--out", default="data/affectbench.csv")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    path = build(args.n, args.out, args.seed)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
