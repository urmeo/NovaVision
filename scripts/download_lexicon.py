"""Download the Warriner et al. (2013) norms as a NovaVision lexicon."""

from __future__ import annotations

import argparse
import csv
import io
import urllib.request
from pathlib import Path

URL = "http://crr.ugent.be/papers/Ratings_Warriner_et_al.csv"


def convert(raw: str) -> list[tuple[str, float, float]]:
    rows = []
    for row in csv.DictReader(io.StringIO(raw)):
        word = row["Word"].strip().lower()
        valence = (float(row["V.Mean.Sum"]) - 5.0) / 4.0  # 1..9 -> -1..1
        arousal = (float(row["A.Mean.Sum"]) - 1.0) / 8.0  # 1..9 -> 0..1
        rows.append((word, round(valence, 4), round(arousal, 4)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the Warriner affect norms")
    parser.add_argument("--out", default="data/lexicon/warriner.tsv")
    args = parser.parse_args()

    with urllib.request.urlopen(URL) as resp:
        raw = resp.read().decode("utf-8", errors="replace")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("word\tvalence\tarousal\n")
        for word, valence, arousal in convert(raw):
            fh.write(f"{word}\t{valence}\t{arousal}\n")

    print(f"Wrote {out}. Set NOVAVISION_LEXICON={out} to use it.")


if __name__ == "__main__":
    main()
