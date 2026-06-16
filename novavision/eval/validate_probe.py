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


def validate(probe, pairs: list[tuple[Path, str]]) -> dict:
    from PIL import Image

    y_true, y_pred = [], []
    for path, label in pairs:
        with Image.open(path) as im:
            rec = probe.recover(im.convert("RGB"))
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
    parser.add_argument("--images-dir", required=True, help="root/<emotion>/*.jpg")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--out", default="results/probe_validation.json")
    args = parser.parse_args()

    from novavision.config import CLIP_REVISION
    from novavision.eval.probes import CLIPProbe

    probe = CLIPProbe(model_id=args.clip_model, revision=CLIP_REVISION)
    report = validate(probe, load_image_folder(args.images_dir))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "confusion"}, indent=2))


if __name__ == "__main__":
    main()
