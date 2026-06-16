"""Human-study harness: sample images for rating, then score agreement.

The probe is only a proxy for perceived emotion until that proxy is checked
against people. ``build_sheet`` regenerates a stratified sample of images
deterministically (so no large blobs live in the repo) and writes a blank
rating sheet plus a hidden key; ``analyze`` reports human-vs-probe agreement
(Cohen's kappa) once raters fill the sheet.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from novavision.data import load_content_bank
from novavision.eval.metrics import accuracy, cohen_kappa
from novavision.generation import get_backend
from novavision.prompting import NEGATIVE_PROMPT, build_prompt
from novavision.taxonomy import EMOTIONS, prior


def sample_records(records: list[dict], n: int, seed: int) -> list[dict]:
    """A class-stratified sample from the conditioned tiers."""
    rng = random.Random(seed)
    pool = [r for r in records if r["tier"] in ("emotion", "affect")]
    by_emotion: dict[str, list[dict]] = {e: [] for e in EMOTIONS}
    for r in pool:
        by_emotion[r["intended"]].append(r)
    for rows in by_emotion.values():
        rng.shuffle(rows)

    picked: list[dict] = []
    i = 0
    while len(picked) < min(n, len(pool)):
        for e in EMOTIONS:
            if i < len(by_emotion[e]) and len(picked) < n:
                picked.append(by_emotion[e][i])
        i += 1
    return picked


def build_sheet(results_dir: str | Path, n: int = 60, seed: int = 0, gen=None) -> Path:
    results_dir = Path(results_dir)
    data = json.loads((results_dir / "results.json").read_text())
    cfg = data["manifest"]["config"]
    style = cfg.get("style", "artistic")
    bank = load_content_bank()

    from novavision.experiments.run import _seed

    gen = gen or get_backend(cfg["backend"], model_id=cfg["diffusion_model"])
    picked = sample_records(data["records"], n, seed)

    study = results_dir / "human_study"
    images = study / "images"
    images.mkdir(parents=True, exist_ok=True)

    sheet, key = [], []
    for i, r in enumerate(picked):
        ci, ei = bank.index(r["content"]), EMOTIONS.index(r["intended"])
        pv, pa = prior(r["intended"])
        prompt = build_prompt(
            r["content"], emotion=r["intended"], valence=pv, arousal=pa, style=style, tier=r["tier"]
        )
        image = gen.generate(
            prompt,
            width=cfg["width"],
            height=cfg["height"],
            seed=_seed(cfg["base_seed"], ci, ei, r["seed"]),
            negative_prompt=NEGATIVE_PROMPT,
        )
        rel = f"images/{i:03d}.png"
        image.save(study / rel)
        sheet.append({"id": i, "image": rel, "emotion": ""})
        key.append({"id": i, "intended": r["intended"], "probe": r["predicted"]})

    _write_csv(study / "ratings_template.csv", ["id", "image", "emotion"], sheet)
    _write_csv(study / "key.csv", ["id", "intended", "probe"], key)
    (study / "README.md").write_text(_INSTRUCTIONS.format(labels=", ".join(EMOTIONS)))
    return study


def analyze(ratings_csv: str | Path, key_csv: str | Path) -> dict:
    ratings = {int(r["id"]): r["emotion"].strip().lower() for r in _read_csv(ratings_csv)}
    key = {int(r["id"]): r for r in _read_csv(key_csv)}

    ids = [i for i in key if ratings.get(i)]
    human = [ratings[i] for i in ids]
    probe = [key[i]["probe"] for i in ids]
    intended = [key[i]["intended"] for i in ids]
    return {
        "n_rated": len(ids),
        "human_vs_probe_kappa": round(cohen_kappa(human, probe, EMOTIONS), 4),
        "human_vs_intended_acc": round(accuracy(human, intended), 4),
        "probe_vs_intended_acc": round(accuracy(probe, intended), 4),
    }


def _write_csv(path, fields, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


_INSTRUCTIONS = """# Human study

Open each image in `images/` and, in `ratings_template.csv`, fill the `emotion`
column with the single best-fitting label from: {labels}.

Use three or more independent raters (one sheet each). Then score agreement:

    python -m novavision.eval.human_study analyze --ratings rater1.csv --key key.csv

`key.csv` holds the intended and probe labels; keep it hidden from raters.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Human-study harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--results", default="results/paper")
    b.add_argument("--n", type=int, default=60)
    b.add_argument("--seed", type=int, default=0)

    a = sub.add_parser("analyze")
    a.add_argument("--ratings", required=True)
    a.add_argument("--key", required=True)

    args = parser.parse_args()
    if args.cmd == "build":
        path = build_sheet(args.results, n=args.n, seed=args.seed)
        print(f"Wrote rating sheet to {path}")
    else:
        print(json.dumps(analyze(args.ratings, args.key), indent=2))


if __name__ == "__main__":
    main()
