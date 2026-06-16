"""Affect-recovery benchmark across conditioning tiers, with floors and stats.

The primary track renders emotion-neutral content under each intended emotion,
so recovered emotion is attributable to the conditioning, not the scene. Two
floors bound the claim: ``raw`` (no emotion → chance) and ``scene`` (fixed
template, no content → pure template recognition). Every tier is run over
several seeds, and tier differences are reported with bootstrap CIs and a
paired significance test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from novavision.config import CLIP_REVISION
from novavision.data import load_content_bank, sha256
from novavision.determinism import set_determinism
from novavision.eval import figures
from novavision.eval.metrics import (
    accuracy,
    bootstrap_ci,
    confusion_matrix,
    macro_f1,
    paired_bootstrap_test,
    pearson,
    spearman,
)
from novavision.eval.probes import CLIPProbe
from novavision.experiments.manifest import build_manifest
from novavision.generation import get_backend
from novavision.prompting import NEGATIVE_PROMPT, build_prompt
from novavision.taxonomy import EMOTIONS, prior

CONDITIONS = ("raw", "emotion", "affect", "scene")
CONTRASTS = (("emotion", "raw"), ("affect", "emotion"), ("affect", "raw"))


def _seed(base: int, ci: int, ei: int, sk: int) -> int:
    """Shared per (content, emotion, seed) so tiers are paired on noise."""
    return base + ((ci * 97 + ei * 13 + sk) % 100_000)


def run_experiment(
    backend: str = "diffusers",
    benchmark: str | None = None,
    *,
    contents: int | None = None,
    seeds: int = 3,
    base_seed: int = 0,
    style: str = "artistic",
    out: str = "results",
    width: int = 512,
    height: int = 512,
    diffusion_model: str = "stabilityai/sd-turbo",
    clip_model: str = "openai/clip-vit-base-patch32",
) -> dict:
    set_determinism(base_seed)
    bank = load_content_bank()
    if contents:
        bank = bank[:contents]

    gen = get_backend(backend, model_id=diffusion_model)
    probe = CLIPProbe(model_id=clip_model, revision=CLIP_REVISION)

    records = _content_records(bank, gen, probe, style, seeds, base_seed, width, height)

    metrics = _summarize(records)
    contrasts = _contrasts(records)
    manifest = build_manifest(
        backend=backend,
        diffusion_model=diffusion_model,
        clip_model=clip_model,
        device=getattr(gen, "device", "n/a"),
        dtype=getattr(gen, "dtype", "n/a"),
        style=style,
        contents=len(bank),
        seeds=seeds,
        base_seed=base_seed,
        width=width,
        height=height,
        benchmark=benchmark,
        benchmark_sha256=sha256(benchmark) if benchmark else None,
    )
    _write(out, records, metrics, contrasts, manifest)
    return {"metrics": metrics, "contrasts": contrasts}


def _content_records(bank, gen, probe, style, seeds, base_seed, width, height) -> list[dict]:
    records: list[dict] = []
    total = len(bank) * len(EMOTIONS) * seeds
    step = 0
    for ci, content in enumerate(bank):
        for ei, emotion in enumerate(EMOTIONS):
            pv, pa = prior(emotion)
            for sk in range(seeds):
                seed = _seed(base_seed, ci, ei, sk)
                step += 1
                print(f"[{step}/{total}] {emotion} :: {content[:30]}", flush=True)
                for tier in ("raw", "emotion", "affect"):
                    prompt = build_prompt(
                        content, emotion=emotion, valence=pv, arousal=pa, style=style, tier=tier
                    )
                    image = gen.generate(
                        prompt,
                        width=width,
                        height=height,
                        seed=seed,
                        negative_prompt=NEGATIVE_PROMPT,
                    )
                    rec = probe.recover(image)
                    records.append(
                        _record(
                            probe.name,
                            tier,
                            content,
                            emotion,
                            sk,
                            rec,
                            pv,
                            pa,
                            probe.clip_t(image, content),
                        )
                    )

    # Scene floor: fixed template, no content — depends only on (emotion, seed).
    for ei, emotion in enumerate(EMOTIONS):
        pv, pa = prior(emotion)
        for sk in range(seeds):
            seed = _seed(base_seed, 0, ei, sk)
            prompt = build_prompt("", emotion=emotion, style=style, tier="scene")
            image = gen.generate(
                prompt, width=width, height=height, seed=seed, negative_prompt=NEGATIVE_PROMPT
            )
            rec = probe.recover(image)
            # No input content, so CLIP-T against content is undefined here.
            records.append(_record(probe.name, "scene", "", emotion, sk, rec, pv, pa, float("nan")))
    return records


def _record(probe, tier, content, emotion, sk, rec, pv, pa, clip_t) -> dict:
    return {
        "probe": probe,
        "tier": tier,
        "content": content,
        "intended": emotion,
        "seed": sk,
        "predicted": rec.emotion,
        "intended_valence": pv,
        "intended_arousal": pa,
        "recovered_valence": rec.valence,
        "recovered_arousal": rec.arousal,
        "clip_t": clip_t,
    }


def _summarize(records) -> dict:
    summary: dict = {}
    for tier in CONDITIONS:
        sub = [r for r in records if r["tier"] == tier]
        if not sub:
            continue
        y_true = [r["intended"] for r in sub]
        y_pred = [r["predicted"] for r in sub]
        correct = [int(t == p) for t, p in zip(y_true, y_pred)]
        lo, hi = bootstrap_ci(correct)
        iv, rv = _col(sub, "intended_valence"), _col(sub, "recovered_valence")
        ia, ra = _col(sub, "intended_arousal"), _col(sub, "recovered_arousal")
        clip = [r["clip_t"] for r in sub if r["clip_t"] == r["clip_t"]]  # drop nan
        summary[tier] = {
            "accuracy": round(accuracy(y_true, y_pred), 4),
            "accuracy_ci": [round(lo, 4), round(hi, 4)],
            "macro_f1": round(macro_f1(y_true, y_pred, EMOTIONS), 4),
            "valence_r": round(pearson(iv, rv), 4),
            "valence_rho": round(spearman(iv, rv), 4),
            "arousal_r": round(pearson(ia, ra), 4),
            "arousal_rho": round(spearman(ia, ra), 4),
            "clip_t": round(sum(clip) / len(clip), 4) if clip else float("nan"),
            "n": len(sub),
        }
    summary["chance"] = round(1 / len(EMOTIONS), 4)
    return summary


def _contrasts(records) -> dict:
    """Paired bootstrap on per-item recovery correctness between tiers."""
    by_key: dict[tuple, dict[str, int]] = {}
    for r in records:
        key = (r["content"], r["intended"], r["seed"])
        by_key.setdefault(key, {})[r["tier"]] = int(r["intended"] == r["predicted"])

    out = {}
    for hi, lo in CONTRASTS:
        a, b = [], []
        for tiers in by_key.values():
            if hi in tiers and lo in tiers:
                a.append(tiers[hi])
                b.append(tiers[lo])
        if a:
            res = paired_bootstrap_test(a, b)
            out[f"{hi}_vs_{lo}"] = {k: round(v, 4) for k, v in res.items()}
    return out


def _col(rows, key):
    return [r[key] for r in rows]


def _write(out, records, metrics, contrasts, manifest) -> None:
    out_dir = Path(out)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "manifest": manifest,
        "metrics": metrics,
        "contrasts": contrasts,
        "records": records,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))

    acc = {t: metrics[t]["accuracy"] for t in CONDITIONS if t in metrics}
    figures.plot_accuracy(acc, fig_dir / "accuracy.png", chance=metrics["chance"])
    for tier in CONDITIONS:
        sub = [r for r in records if r["tier"] == tier]
        if not sub:
            continue
        cm = confusion_matrix(_col(sub, "intended"), _col(sub, "predicted"), EMOTIONS)
        figures.plot_confusion(cm, EMOTIONS, fig_dir / f"confusion_{tier}.png", title=tier)
    figures.plot_va_scatter(records, "affect", fig_dir / "va_affect.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Affect-recovery benchmark")
    parser.add_argument("--backend", default="diffusers", choices=["diffusers", "hf-api", "null"])
    parser.add_argument("--benchmark", default=None, help="optional text benchmark for provenance")
    parser.add_argument("--contents", type=int, default=None, help="limit content-bank subjects")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--style", default="artistic")
    parser.add_argument("--diffusion-model", default="stabilityai/sd-turbo")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    result = run_experiment(
        backend=args.backend,
        benchmark=args.benchmark,
        contents=args.contents,
        seeds=args.seeds,
        base_seed=args.base_seed,
        style=args.style,
        out=args.out,
        width=args.width,
        height=args.height,
        diffusion_model=args.diffusion_model,
        clip_model=args.clip_model,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
