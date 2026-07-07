"""Affect-recovery benchmark across conditioning tiers, with floors and stats.

Two tracks share one pipeline. The **content** track renders emotion-neutral
subjects under each intended emotion, so recovery is attributable to the
conditioning, not the scene; its floors are ``raw`` (no emotion → chance) and
``scene`` (fixed template, no content). The **text** track conditions on
AffectBench sentences, grounds valence/arousal in the text, and uses a
``shuffled`` floor (condition on a wrong emotion) since the sentence already
carries affect. Every tier runs over several seeds; tier differences are
reported with bootstrap CIs and a paired significance test.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from novavision.affect.analyzer import EmotionAnalyzer
from novavision.config import CLIP_REVISION
from novavision.data import load_benchmark, load_content_bank, sha256
from novavision.determinism import set_determinism
from novavision.eval import figures
from novavision.eval.metrics import (
    accuracy,
    bootstrap_ci,
    bootstrap_corr_ci,
    confusion_matrix,
    macro_f1,
    mae,
    majority_baseline,
    paired_bootstrap_test,
    pearson,
    permutation_test,
    prediction_collapse,
    spearman,
)
from novavision.eval.probes import CLIPProbe, HFImageClassifierProbe
from novavision.experiments.manifest import build_manifest
from novavision.generation import get_backend
from novavision.prompting import NEGATIVE_PROMPT, TIERS, build_prompt
from novavision.taxonomy import EMOTIONS, prior

CONDITIONS = {
    "content": (*TIERS, "scene"),
    "text": (*TIERS, "shuffled"),
}
# The full ordered vocabulary, for consumers that render every condition
# (report tables, benchmark submissions). Owned here so a new tier cannot
# silently vanish from downstream artifacts.
ALL_CONDITIONS = (*TIERS, "scene", "shuffled")
CONTRASTS = (
    ("naive", "raw"),
    ("emotion", "naive"),
    ("emotion", "raw"),
    ("affect", "emotion"),
    ("affect", "raw"),
)


# _seed is injective only while ei*13 + sk < 97 and ci*97 + 90 < 100_000, i.e.
# items <= 1030 and seeds <= 13; beyond that, distinct (item, emotion, seed)
# triples silently share generator noise. Enforced before every run.
SEED_MAX_ITEMS = 1030
SEED_MAX_SEEDS = 13


def _seed(base: int, ci: int, ei: int, sk: int) -> int:
    """Shared per (content, emotion, seed) so tiers are paired on noise.

    The formula is frozen: committed records reproduce their images from it
    (see ``_record``), so collisions are excluded by bounding the domain
    (``_check_seed_domain``) rather than by changing the mixing.
    """
    return base + ((ci * 97 + ei * 13 + sk) % 100_000)


def _check_seed_domain(n_items: int, seeds: int) -> None:
    """Refuse configurations where _seed would stop being collision-free."""
    if n_items > SEED_MAX_ITEMS or seeds > SEED_MAX_SEEDS:
        raise ValueError(
            f"seed mixing is collision-free only up to {SEED_MAX_ITEMS} items and "
            f"{SEED_MAX_SEEDS} seeds (got {n_items} items, {seeds} seeds); "
            "shard the run over --base-seed values at least 100000 apart instead"
        )


def _shuffle_emotion(gold: str, seed: int) -> str:
    """A deterministic wrong emotion, for the shuffled floor."""
    others = [e for e in EMOTIONS if e != gold]
    return others[seed % len(others)]


class _Checkpoint:
    """Stream records to JSONL so a long run resumes instead of restarting.

    Each image's record is keyed by (tier, intended, index, seed); an interrupted
    run reloads the completed records and skips their regeneration. A no-op when
    ``path`` is None, so the default (non-resumable) path is unchanged.
    """

    def __init__(self, path: Path | None):
        self.path = path
        self.done: dict = {}
        if path and path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.done[(r["tier"], r["intended"], r["index"], r["seed"])] = r

    def cached(self, tier: str, intended: str, index: int, seed: int):
        return self.done.get((tier, intended, index, seed))

    def record(self, rec: dict) -> dict:
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(json_safe(rec)) + "\n")
        return rec


def _make_probe(kind: str, probe_model: str | None, clip_model: str, device: str | None):
    """CLIP (default) or an independent HF image-emotion classifier."""
    if kind == "hf":
        if not probe_model:
            raise ValueError("--probe hf requires --probe-model")
        return HFImageClassifierProbe(model_id=probe_model, device=device)
    if kind != "clip":
        raise ValueError(f"unknown probe kind '{kind}', expected 'clip' or 'hf'")
    model_id = probe_model or clip_model
    # The pinned revision is a ViT-B/32 commit; it must not leak onto other models.
    revision = CLIP_REVISION if model_id == "openai/clip-vit-base-patch32" else None
    return CLIPProbe(model_id=model_id, device=device, revision=revision)


def run_experiment(
    backend: str = "diffusers",
    *,
    track: str = "content",
    benchmark: str | None = None,
    contents: int | None = None,
    limit: int | None = None,
    seeds: int = 3,
    base_seed: int = 0,
    style: str = "artistic",
    out: str = "results",
    width: int = 512,
    height: int = 512,
    device: str | None = None,
    diffusion_model: str = "stabilityai/sd-turbo",
    probe: str = "clip",
    clip_model: str = "openai/clip-vit-base-patch32",
    probe_model: str | None = None,
    emotion_model: str = "j-hartmann/emotion-english-distilroberta-base",
    coverage_override: float | None = None,
    resume: bool = False,
) -> dict:
    set_determinism(base_seed)
    kwargs = {"model_id": diffusion_model}
    if device:
        kwargs["device"] = device
    gen = get_backend(backend, **kwargs)
    probe_obj = _make_probe(probe, probe_model, clip_model, device)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = _Checkpoint(out_dir / "records.jsonl" if resume else None)

    if track == "text":
        if not benchmark:
            raise ValueError("The text track requires --benchmark.")
        rows = load_benchmark(benchmark)
        if limit is not None:
            rows = rows[:limit]
        n_items = len(rows)
        _check_seed_domain(n_items, seeds)
        analyzer = EmotionAnalyzer(model_name=emotion_model, coverage_override=coverage_override)
        records = _text_records(
            rows, gen, probe_obj, analyzer, style, seeds, base_seed, width, height, ckpt
        )
    else:
        bank = load_content_bank()
        if contents is not None:
            bank = bank[:contents]
        n_items = len(bank)
        _check_seed_domain(n_items, seeds)
        records = _content_records(
            bank, gen, probe_obj, style, seeds, base_seed, width, height, ckpt
        )

    conditions = CONDITIONS[track]
    metrics = _summarize(records, conditions)
    contrasts = _contrasts(records)
    manifest = build_manifest(
        backend=backend,
        track=track,
        diffusion_model=diffusion_model,
        probe=probe_obj.name,
        emotion_model=emotion_model if track == "text" else None,
        device=getattr(gen, "device", "n/a"),
        dtype=getattr(gen, "dtype", "n/a"),
        style=style,
        items=n_items,
        seeds=seeds,
        base_seed=base_seed,
        width=width,
        height=height,
        benchmark=benchmark,
        benchmark_sha256=sha256(benchmark) if benchmark else None,
        coverage_override=coverage_override,
    )
    _write(out, records, metrics, contrasts, manifest, conditions)
    if ckpt.path and ckpt.path.exists():
        ckpt.path.unlink()  # run completed; results.json supersedes the checkpoint
    return {"metrics": metrics, "contrasts": contrasts}


def _content_records(bank, gen, probe, style, seeds, base_seed, width, height, ckpt) -> list[dict]:
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
                for tier in TIERS:
                    done = ckpt.cached(tier, emotion, ci, sk)
                    if done is not None:
                        records.append(done)
                        continue
                    image = _render(gen, content, emotion, pv, pa, style, tier, seed, width, height)
                    rec = probe.recover(image)
                    clip_t = probe.clip_t(image, content)
                    records.append(
                        ckpt.record(
                            _record(
                                probe.name,
                                tier,
                                content,
                                emotion,
                                sk,
                                rec,
                                pv,
                                pa,
                                clip_t,
                                index=ci,
                            )
                        )
                    )

    # scene floor
    for ei, emotion in enumerate(EMOTIONS):
        pv, pa = prior(emotion)
        for sk in range(seeds):
            done = ckpt.cached("scene", emotion, 0, sk)
            if done is not None:
                records.append(done)
                continue
            seed = _seed(base_seed, 0, ei, sk)
            image = _render(gen, "", emotion, pv, pa, style, "scene", seed, width, height)
            rec = probe.recover(image)
            records.append(
                ckpt.record(
                    _record(probe.name, "scene", "", emotion, sk, rec, pv, pa, float("nan"))
                )
            )
    return records


def _text_records(rows, gen, probe, analyzer, style, seeds, base_seed, width, height, ckpt):
    records: list[dict] = []
    total = len(rows) * seeds
    step = 0
    for ri, row in enumerate(rows):
        text, gold = row["text"], row["emotion"]
        a = analyzer.analyze(text)
        iv, ia = a.valence, a.arousal
        gi = EMOTIONS.index(gold)
        for sk in range(seeds):
            step += 1
            print(f"[{step}/{total}] {gold} :: {text[:30]}", flush=True)
            seed = _seed(base_seed, ri, gi, sk)
            for tier in TIERS:
                done = ckpt.cached(tier, gold, ri, sk)
                if done is not None:
                    records.append(done)
                    continue
                image = _render(gen, text, gold, iv, ia, style, tier, seed, width, height)
                rec = probe.recover(image)
                clip_t = probe.clip_t(image, text)
                records.append(
                    ckpt.record(
                        _record(
                            probe.name,
                            tier,
                            text,
                            gold,
                            sk,
                            rec,
                            iv,
                            ia,
                            clip_t,
                            a.primary,
                            index=ri,
                        )
                    )
                )

            # shuffled floor: condition on a wrong emotion, score against it
            wrong = _shuffle_emotion(gold, seed)
            done = ckpt.cached("shuffled", wrong, ri, sk)
            if done is not None:
                records.append(done)
                continue
            wv, wa = prior(wrong)
            wseed = _seed(base_seed, ri, EMOTIONS.index(wrong), sk)
            image = _render(gen, text, wrong, wv, wa, style, "emotion", wseed, width, height)
            rec = probe.recover(image)
            clip_t = probe.clip_t(image, text)
            records.append(
                ckpt.record(
                    _record(
                        probe.name,
                        "shuffled",
                        text,
                        wrong,
                        sk,
                        rec,
                        wv,
                        wa,
                        clip_t,
                        a.primary,
                        index=ri,
                    )
                )
            )
    return records


def _render(gen, content, emotion, v, a, style, tier, seed, width, height):
    prompt = build_prompt(content, emotion=emotion, valence=v, arousal=a, style=style, tier=tier)
    return gen.generate(
        prompt, width=width, height=height, seed=seed, negative_prompt=NEGATIVE_PROMPT
    )


def _record(
    probe, tier, content, emotion, sk, rec, pv, pa, clip_t, classified=None, index=0
) -> dict:
    return {
        "probe": probe,
        "tier": tier,
        "content": content,
        "index": index,  # content/row position; the seed salt, so any image can be reproduced
        "intended": emotion,
        "classified": classified,
        "seed": sk,
        "predicted": rec.emotion,
        "intended_valence": pv,
        "intended_arousal": pa,
        "recovered_valence": rec.valence,
        "recovered_arousal": rec.arousal,
        "clip_t": clip_t,
    }


def _summarize(records, conditions) -> dict:
    summary: dict = {}
    for tier in conditions:
        sub = [r for r in records if r["tier"] == tier]
        if not sub:
            continue
        y_true = [r["intended"] for r in sub]
        y_pred = [r["predicted"] for r in sub]
        correct = [int(t == p) for t, p in zip(y_true, y_pred)]
        lo, hi = bootstrap_ci(correct)
        iv, rv = _col(sub, "intended_valence"), _col(sub, "recovered_valence")
        ia, ra = _col(sub, "intended_arousal"), _col(sub, "recovered_arousal")
        # drop nan and null (sanitized nan in a re-read results.json)
        clip = [r["clip_t"] for r in sub if _finite(r["clip_t"])]
        vlo, vhi = bootstrap_corr_ci(iv, rv)
        alo, ahi = bootstrap_corr_ci(ia, ra)
        summary[tier] = {
            "accuracy": round(accuracy(y_true, y_pred), 4),
            "accuracy_ci": [round(lo, 4), round(hi, 4)],
            "macro_f1": round(macro_f1(y_true, y_pred, EMOTIONS), 4),
            "valence_r": round(pearson(iv, rv), 4),
            "valence_rho": round(spearman(iv, rv), 4),
            "valence_rho_ci": [round(vlo, 4), round(vhi, 4)],
            "valence_mae": round(mae(iv, rv), 4),
            "arousal_r": round(pearson(ia, ra), 4),
            "arousal_rho": round(spearman(ia, ra), 4),
            "arousal_rho_ci": [round(alo, 4), round(ahi, 4)],
            "arousal_mae": round(mae(ia, ra), 4),
            "clip_t": round(sum(clip) / len(clip), 4) if clip else float("nan"),
            "majority_baseline": round(majority_baseline(y_true), 4),
            "collapse": _round_collapse(prediction_collapse(y_pred)),
            "shuffled_control": permutation_test(y_true, y_pred),
            "n": len(sub),
        }
    summary["chance"] = round(1 / len(EMOTIONS), 4)
    summary["probe_health"] = _probe_health(records)
    cls = _classification_accuracy(records)
    if cls is not None:
        summary["classification_accuracy"] = cls
    return summary


def _round_collapse(c: dict) -> dict:
    return {"label": c["label"], "rate": round(float(c["rate"]), 4), "distinct": c["distinct"]}


def _probe_health(records) -> dict:
    """Probe-degeneracy diagnostic over the conditioning tiers (floors excluded).

    If the probe collapses onto one label, recovery at chance is the trivial
    consequence of that collapse, not evidence the floors discriminate, so this
    is reported next to every headline number.
    """
    preds = [r["predicted"] for r in records if r["tier"] in TIERS]
    c = prediction_collapse(preds)
    return {
        "majority_label": c["label"],
        "majority_rate": round(float(c["rate"]), 4),
        "distinct_labels": c["distinct"],
        "n_labels": len(EMOTIONS),
    }


def _classification_accuracy(records) -> float | None:
    """Upstream classifier vs gold, once per text (text track only)."""
    seen, true, pred = set(), [], []
    for r in records:
        if r.get("classified") is None or r["content"] in seen:
            continue
        seen.add(r["content"])
        true.append(r["intended"])
        pred.append(r["classified"])
    return round(accuracy(true, pred), 4) if true else None


def _contrasts(records) -> dict:
    """Paired bootstrap on per-item recovery correctness between tiers."""
    # shared seed pairs tiers
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


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def json_safe(obj):
    """Replace non-finite floats (nan/inf) with null so output is valid RFC-8259 JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


def dump_results(payload: dict, path) -> None:
    """Write results.json as standards-compliant JSON (no bare NaN tokens)."""
    Path(path).write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False))


def _write(out, records, metrics, contrasts, manifest, conditions) -> None:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": manifest,
        "metrics": metrics,
        "contrasts": contrasts,
        "records": records,
    }
    dump_results(payload, out_dir / "results.json")
    _write_figures(out_dir, records, metrics, conditions)


def _write_figures(out_dir, records, metrics, conditions) -> None:
    """Render the accuracy, per-tier confusion, and VA-scatter figures.

    Pure function of the records and summary, so it is reused to refresh figures
    from an existing run without regenerating any images (see scripts/resummarize).
    """
    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    acc = {t: metrics[t]["accuracy"] for t in conditions if t in metrics}
    figures.plot_accuracy(acc, fig_dir / "accuracy.png", chance=metrics["chance"])
    for tier in conditions:
        sub = [r for r in records if r["tier"] == tier]
        if not sub:
            continue
        cm = confusion_matrix(_col(sub, "intended"), _col(sub, "predicted"), EMOTIONS)
        figures.plot_confusion(cm, EMOTIONS, fig_dir / f"confusion_{tier}.png", title=tier)
    if any(r["tier"] == "affect" for r in records):
        figures.plot_va_scatter(records, "affect", fig_dir / "va_affect.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Affect-recovery benchmark")
    parser.add_argument("--backend", default="diffusers", choices=["diffusers", "hf-api", "null"])
    parser.add_argument("--track", default="content", choices=["content", "text"])
    parser.add_argument(
        "--benchmark", default=None, help="AffectBench CSV (required for text track)"
    )
    parser.add_argument("--contents", type=int, default=None, help="limit content-bank subjects")
    parser.add_argument("--limit", type=int, default=None, help="limit text-benchmark rows")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--style", default="artistic")
    parser.add_argument("--diffusion-model", default="stabilityai/sd-turbo")
    parser.add_argument("--probe", default="clip", choices=["clip", "hf"])
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--probe-model", default=None, help="probe model id (CLIP or HF)")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--device", default=None, help="cpu, cuda, or mps; auto if unset")
    parser.add_argument("--out", default="results")
    parser.add_argument(
        "--force-coverage",
        type=float,
        default=None,
        help="ablation: force the lexicon/prior blend weight (0=prior only, 1=lexicon only)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="stream records to <out>/records.jsonl and resume an interrupted run",
    )
    args = parser.parse_args()

    result = run_experiment(
        backend=args.backend,
        track=args.track,
        benchmark=args.benchmark,
        contents=args.contents,
        limit=args.limit,
        seeds=args.seeds,
        base_seed=args.base_seed,
        style=args.style,
        out=args.out,
        width=args.width,
        height=args.height,
        device=args.device,
        diffusion_model=args.diffusion_model,
        probe=args.probe,
        clip_model=args.clip_model,
        probe_model=args.probe_model,
        coverage_override=args.force_coverage,
        resume=args.resume,
    )
    print(json.dumps(json_safe(result), indent=2))


if __name__ == "__main__":
    main()
