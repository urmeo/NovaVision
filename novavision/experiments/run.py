"""Run the affect-recovery benchmark across conditioning tiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from novavision.affect.analyzer import EmotionAnalyzer
from novavision.data import load_benchmark
from novavision.eval import figures
from novavision.eval.clip_affect import CLIPAffect
from novavision.eval.metrics import accuracy, confusion_matrix, macro_f1, pearson
from novavision.generation import get_backend
from novavision.prompting import NEGATIVE_PROMPT, TIERS, build_prompt
from novavision.taxonomy import EMOTIONS


def run_experiment(
    backend: str = "diffusers",
    benchmark: str | None = None,
    tiers=TIERS,
    style: str = "artistic",
    limit: int | None = None,
    seed: int = 0,
    width: int = 512,
    height: int = 512,
    out: str = "results",
    diffusion_model: str = "stabilityai/sd-turbo",
    emotion_model: str = "j-hartmann/emotion-english-distilroberta-base",
    clip_model: str = "openai/clip-vit-base-patch32",
) -> dict:
    rows = load_benchmark(benchmark)
    if limit:
        rows = rows[:limit]

    analyzer = EmotionAnalyzer(model_name=emotion_model)
    gen = (
        get_backend(backend, model_id=diffusion_model)
        if backend == "diffusers"
        else get_backend(backend)
    )
    clip = CLIPAffect(model_id=clip_model)

    records = []
    for i, row in enumerate(rows):
        text, intended = row["text"], row["emotion"]
        print(f"[{i + 1}/{len(rows)}] {intended}", flush=True)
        analysis = analyzer.analyze(text)
        # Condition on the gold emotion to isolate controllability from
        # classifier error; ground valence/arousal in the text.
        for tier in tiers:
            prompt = build_prompt(
                text,
                emotion=intended,
                valence=analysis.valence,
                arousal=analysis.arousal,
                style=style,
                tier=tier,
            )
            image = gen.generate(
                prompt, width=width, height=height, seed=seed + i, negative_prompt=NEGATIVE_PROMPT
            )
            rec = clip.recover(image)
            records.append(
                {
                    "text": text,
                    "intended": intended,
                    "classified": analysis.primary,
                    "tier": tier,
                    "predicted": rec.emotion,
                    "intended_valence": analysis.valence,
                    "intended_arousal": analysis.arousal,
                    "recovered_valence": rec.valence,
                    "recovered_arousal": rec.arousal,
                    "clip_t": clip.clip_t(image, text),
                }
            )

    metrics = _summarize(records, tiers)
    classification = _classification_accuracy(records)
    _write(out, backend, style, seed, records, metrics, classification)
    return {"metrics": metrics, "classification_accuracy": classification}


def _summarize(records, tiers) -> dict:
    summary = {}
    for tier in tiers:
        sub = [r for r in records if r["tier"] == tier]
        y_true = [r["intended"] for r in sub]
        y_pred = [r["predicted"] for r in sub]
        summary[tier] = {
            "accuracy": round(accuracy(y_true, y_pred), 4),
            "macro_f1": round(macro_f1(y_true, y_pred, EMOTIONS), 4),
            "valence_r": round(
                pearson(
                    [r["intended_valence"] for r in sub], [r["recovered_valence"] for r in sub]
                ),
                4,
            ),
            "arousal_r": round(
                pearson(
                    [r["intended_arousal"] for r in sub], [r["recovered_arousal"] for r in sub]
                ),
                4,
            ),
            "clip_t": round(sum(r["clip_t"] for r in sub) / len(sub), 4) if sub else 0.0,
            "n": len(sub),
        }
    return summary


def _classification_accuracy(records) -> float:
    seen, true, pred = set(), [], []
    for r in records:
        if r["text"] in seen:
            continue
        seen.add(r["text"])
        true.append(r["intended"])
        pred.append(r["classified"])
    return round(accuracy(true, pred), 4)


def _write(out, backend, style, seed, records, metrics, classification) -> None:
    out_dir = Path(out)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "config": {"backend": backend, "style": style, "seed": seed},
        "metrics": metrics,
        "classification_accuracy": classification,
        "records": records,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))

    figures.plot_accuracy({t: metrics[t]["accuracy"] for t in metrics}, fig_dir / "accuracy.png")
    for tier in metrics:
        sub = [r for r in records if r["tier"] == tier]
        cm = confusion_matrix([r["intended"] for r in sub], [r["predicted"] for r in sub], EMOTIONS)
        figures.plot_confusion(cm, EMOTIONS, fig_dir / f"confusion_{tier}.png", title=tier)
        figures.plot_va_scatter(records, tier, fig_dir / f"va_{tier}.png")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Affect-recovery benchmark")
    parser.add_argument("--backend", default="diffusers", choices=["diffusers", "hf-api", "null"])
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--style", default="artistic")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    result = run_experiment(
        backend=args.backend,
        benchmark=args.benchmark,
        style=args.style,
        limit=args.limit,
        seed=args.seed,
        out=args.out,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
