"Human-study harness: sample images for rating, then score agreement."

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from novavision.data import load_content_bank
from novavision.eval.metrics import accuracy, cohen_kappa
from novavision.generation import get_backend
from novavision.prompting import NEGATIVE_PROMPT, build_prompt
from novavision.taxonomy import EMOTIONS


def _sample_records(records: list[dict], n: int, seed: int) -> list[dict]:
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


def _record_index(r: dict, bank: list[str]) -> int:
    "The content/row position used as the seed salt, for reproducing the image."
    if "index" in r:
        return int(r["index"])
    if r["content"] in bank:
        return bank.index(r["content"])
    raise ValueError(
        "Cannot reproduce this image: the record has no 'index' and its content is not a "
        f"content-bank subject ({r['content']!r}). Rebuild from a run produced by the current code."
    )


def _verify_rebuild_provenance(data: dict) -> None:
    """Reject unverified replay before constructing or loading a model backend."""
    from novavision.experiments.manifest import build_manifest, package_version
    from novavision.experiments.run import _source_fingerprint

    manifest = data["manifest"]
    cfg = manifest["config"]
    if cfg["backend"] not in ("diffusers", "null"):
        raise ValueError("Backend is non-deterministic; supply verified saved original images")
    required = ("git_sha", "source_sha256", "python", "platform", "packages", "device_info")
    if any(key not in manifest for key in required):
        raise ValueError("Cannot rebuild images: incomplete provenance; use saved original images")
    current = build_manifest(**cfg)
    recorded_sha = manifest["git_sha"]
    if (
        not isinstance(recorded_sha, str)
        or len(recorded_sha) != 40
        or any(c not in "0123456789abcdef" for c in recorded_sha)
        or manifest["source_sha256"] != _source_fingerprint()
        or any(manifest[key] != current[key] for key in ("python", "platform", "device_info"))
        or not set(current["packages"]).issubset(manifest["packages"])
        or any(package_version(pkg) != ver for pkg, ver in manifest["packages"].items())
    ):
        raise ValueError(
            "Cannot rebuild images: source/environment provenance differs from the run"
        )
    if cfg["backend"] == "diffusers":
        revision = manifest.get("model_revisions", {}).get("diffusion")
        if not revision or any(
            cfg.get(k) in (None, "n/a")
            for k in (
                "device",
                "dtype",
                "generation_steps",
                "style",
                "base_seed",
                "width",
                "height",
            )
        ):
            raise ValueError("Cannot rebuild images: generator provenance is incomplete")


def build_sheet(
    results_dir: str | Path, n: int = 60, seed: int = 0, gen=None, *, out: str | Path | None = None
) -> Path:
    "Build blinded ratings from verified images; ``out`` selects the study directory."
    if n < 1:
        raise ValueError("n must be positive")
    results_dir = Path(results_dir)
    data = json.loads((results_dir / "results.json").read_text())
    cfg = data["manifest"]["config"]
    style = cfg.get("style", "artistic")
    picked = _sample_records(data["records"], n, seed)
    if not picked:
        raise ValueError("No conditioned records available for a human study")
    if any(not r.get("image_pixel_sha256") for r in picked):
        raise ValueError(
            "Cannot rebuild or verify images: missing pixel digests; use recorded originals"
        )
    rebuild = [r for r in picked if not r.get("image_path")]
    bank = load_content_bank() if rebuild else []

    from novavision.determinism import set_determinism
    from novavision.experiments.run import _seed

    if rebuild:
        _verify_rebuild_provenance(data)

        for r in rebuild:
            _record_index(r, bank)
        set_determinism(cfg["base_seed"])
        kwargs = {"model_id": cfg["diffusion_model"]}
        if cfg["backend"] == "diffusers":
            kwargs.update(
                device=cfg["device"],
                steps=cfg["generation_steps"],
                revision=data["manifest"]["model_revisions"]["diffusion"],
            )
        gen = gen or get_backend(cfg["backend"], **kwargs)
        if cfg["backend"] == "diffusers" and any(
            getattr(gen, attr, None) != expected
            for attr, expected in (
                ("device", cfg["device"]),
                ("dtype", cfg["dtype"]),
                ("steps", cfg["generation_steps"]),
                ("revision", data["manifest"]["model_revisions"]["diffusion"]),
                ("model_id", cfg["diffusion_model"]),
            )
        ):
            raise ValueError(
                "Cannot rebuild images: supplied backend differs from recorded generator"
            )
    counts = Counter(r["intended"] for r in picked)
    per_class = {e: counts.get(e, 0) for e in EMOTIONS}

    print(f"[human-study] realized per-class counts: {per_class}", flush=True)

    study = Path(out) if out is not None else results_dir / "human_study"
    images = study / "images"
    images.mkdir(parents=True, exist_ok=True)

    sheet, key = [], []
    for i, r in enumerate(picked):
        from PIL import Image

        from novavision.experiments.run import _image_digest

        if r.get("image_path"):
            source = Path(r["image_path"])
            if not source.is_absolute():
                source = results_dir / source
            with Image.open(source) as saved:
                image = saved.convert("RGB")
        else:
            image = _rebuild_image(r, bank, gen, cfg, style, _seed)
        if _image_digest(image) != r["image_pixel_sha256"]:
            raise ValueError("Image pixel digest differs from the image the probe scored")
        rel = f"images/{i:03d}.png"
        image.save(study / rel)
        sheet.append({"id": i, "image": rel, "emotion": ""})
        key.append({"id": i, "intended": r["intended"], "probe": r["predicted"]})

    _write_csv(study / "ratings_template.csv", ["id", "image", "emotion"], sheet)
    _write_csv(study / "key.csv", ["id", "intended", "probe"], key)
    _write_csv(
        study / "counts.csv",
        ["emotion", "n"],
        [{"emotion": e, "n": count} for e, count in per_class.items()],
    )
    return study


def _rebuild_image(r, bank, gen, cfg, style, seed_fn):
    idx = _record_index(r, bank)
    ei = EMOTIONS.index(r["intended"])

    prompt = build_prompt(
        r["content"],
        emotion=r["intended"],
        valence=r["intended_valence"],
        arousal=r["intended_arousal"],
        style=style,
        tier=r["tier"],
    )
    return gen.generate(
        prompt,
        width=cfg["width"],
        height=cfg["height"],
        seed=seed_fn(cfg["base_seed"], idx, ei, r["seed"]),
        negative_prompt=NEGATIVE_PROMPT,
    )


def analyze(ratings_csv: str | Path, key_csv: str | Path) -> dict:
    from novavision.eval.validate_probe import EKMAN_ALIASES

    raw = {int(r["id"]): r["emotion"].strip().lower() for r in _read_csv(ratings_csv)}

    ratings = {i: EKMAN_ALIASES.get(v, v) for i, v in raw.items() if v}
    key = {int(r["id"]): r for r in _read_csv(key_csv)}
    known = set(EMOTIONS)

    ids = [i for i in key if ratings.get(i) in known]
    unscored = sorted(i for i in key if ratings.get(i) and ratings[i] not in known)
    human = [ratings[i] for i in ids]
    probe = [key[i]["probe"] for i in ids]
    intended = [key[i]["intended"] for i in ids]
    return {
        "n_rated": len(ids),
        "n_unscored": len(unscored),
        "unscored_ids": unscored,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Human-study harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--results", default="outputs/results")
    b.add_argument("--n", type=int, default=60)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument(
        "--out",
        default="outputs/generated/human_study",
        help="destination directory for ratings, images and counts",
    )

    a = sub.add_parser("analyze")
    a.add_argument("--ratings", required=True)
    a.add_argument("--key", required=True)

    args = parser.parse_args()
    if args.cmd == "build":
        path = build_sheet(args.results, n=args.n, seed=args.seed, out=args.out)
        print(f"Wrote rating sheet to {path}")
    else:
        from novavision.experiments.run import json_safe

        print(json.dumps(json_safe(analyze(args.ratings, args.key)), indent=2))


if __name__ == "__main__":
    main()
