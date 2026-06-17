"""Validate a recovery probe against a labelled image-emotion dataset.

The benchmark trusts the probe to read emotion from an image. This measures how
much that trust is warranted: run the probe on images with known emotion labels
(e.g. FI or EmoSet, mapped to the Ekman set) and report its accuracy and
confusion — the instrument's known error, to be propagated to every result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from novavision.eval.metrics import accuracy, confusion_matrix, macro_f1
from novavision.taxonomy import EMOTIONS

_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Aliases from common image-emotion datasets to the Ekman set.
EKMAN_ALIASES = {
    "happy": "joy",
    "happiness": "joy",
    "amusement": "joy",
    "contentment": "joy",
    "excitement": "joy",
    "joy": "joy",
    "sad": "sadness",
    "sadness": "sadness",
    "anger": "anger",
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "surprise": "surprise",
    "awe": "surprise",
    "neutral": "neutral",
}


def load_hf_dataset(
    name: str,
    n: int = 200,
    split: str = "train",
    seed: int = 0,
    revision: str | None = None,
    image_key: str = "image",
    label_key: str = "label",
):
    """Sample (image, ekman_label) pairs from a labelled HF image dataset.

    The revision is pinnable and the seed fixes the shuffle, so the exact sample
    behind a reported number is reproducible. Labels are mapped to the Ekman set
    via ``EKMAN_ALIASES``; rows whose label has no Ekman home are skipped.
    """
    import random

    from datasets import load_dataset

    ds = load_dataset(name, split=split, revision=revision)
    feat = ds.features.get(label_key)
    names = getattr(feat, "names", None)

    order = list(range(len(ds)))
    random.Random(seed).shuffle(order)
    pairs: list = []
    for i in order:
        if len(pairs) >= n:
            break
        ex = ds[i]
        raw = ex[label_key]
        label = names[raw] if names and isinstance(raw, int) else raw
        ekman = EKMAN_ALIASES.get(str(label).lower())
        if ekman in set(EMOTIONS):
            pairs.append((ex[image_key].convert("RGB"), ekman))
    return pairs


def load_image_folder(root: str | Path) -> list[tuple[Path, str]]:
    """Images under ``root/<ekman_emotion>/*`` → (path, label) pairs."""
    root = Path(root)
    pairs: list[tuple[Path, str]] = []
    for sub in sorted(root.iterdir()):
        label = sub.name.lower()
        if not sub.is_dir() or label not in set(EMOTIONS):
            continue
        for img in sorted(sub.iterdir()):
            if img.suffix.lower() in _EXTS:
                pairs.append((img, label))
    if not pairs:
        raise ValueError(f"No labelled images found under {root}/<emotion>/")
    return pairs


def validate(probe, pairs: list) -> dict:
    """pairs are (image-or-path, ekman_label)."""
    from PIL import Image

    y_true, y_pred = [], []
    for item, label in pairs:
        img = item if isinstance(item, Image.Image) else Image.open(item).convert("RGB")
        rec = probe.recover(img)
        y_true.append(label)
        y_pred.append(rec.emotion)

    cm = confusion_matrix(y_true, y_pred, EMOTIONS)
    return {
        "probe": probe.name,
        "n": len(pairs),
        "accuracy": round(accuracy(y_true, y_pred), 4),
        "macro_f1": round(macro_f1(y_true, y_pred, EMOTIONS), 4),
        "labels": list(EMOTIONS),
        "confusion": cm.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a probe on labelled images")
    parser.add_argument("--images-dir", default=None, help="root/<emotion>/*.jpg")
    parser.add_argument("--hf-dataset", default=None, help="labelled HF image dataset")
    parser.add_argument("--n", type=int, default=200, help="sample size for --hf-dataset")
    parser.add_argument("--split", default="train", help="HF dataset split")
    parser.add_argument("--revision", default=None, help="HF dataset revision to pin")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--out", default="results/probe_validation.json")
    args = parser.parse_args()

    from novavision.config import CLIP_REVISION
    from novavision.eval.probes import CLIPProbe

    if args.hf_dataset:
        pairs = load_hf_dataset(
            args.hf_dataset, n=args.n, split=args.split, seed=args.seed, revision=args.revision
        )
    elif args.images_dir:
        pairs = load_image_folder(args.images_dir)
    else:
        parser.error("pass --hf-dataset or --images-dir")

    probe = CLIPProbe(model_id=args.clip_model, revision=CLIP_REVISION)
    report = validate(probe, pairs)
    report["source"] = args.hf_dataset or args.images_dir
    # Provenance: make the exact sample behind the number reproducible.
    if args.hf_dataset:
        report["split"] = args.split
        report["seed"] = args.seed
        report["revision"] = args.revision
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "confusion"}, indent=2))


if __name__ == "__main__":
    main()
