"""Bias-correct recovery for the probe's measured error (Rogan-Gladen).

The benchmark reports *apparent* recovery: the rate at which the probe reads back
the intended emotion. The probe is imperfect, and probe validation measures how
imperfect (its per-class sensitivity and specificity on labelled images). This
deconvolves that known error from recovery, so the probe's weakness becomes a
correction rather than only a caveat.

Use the SAME probe for both inputs: the run's ``results.json`` (apparent recovery)
and that probe's ``probe_validation`` confusion (the error model).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from novavision.eval.metrics import rogan_gladen
from novavision.taxonomy import EMOTIONS


def _sensitivity_specificity(confusion: list[list[int]]) -> dict[str, tuple[float, float]]:
    """Per-class (sensitivity, specificity) from a true x predicted count matrix."""
    cm = np.asarray(confusion, dtype=float)
    out = {}
    for i, e in enumerate(EMOTIONS):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        out[e] = (sens, spec)
    return out


def _apparent_recovery(records: list[dict], tier: str) -> dict[str, float]:
    """Per-class apparent recovery rate for one tier."""
    out = {}
    for e in EMOTIONS:
        sub = [r for r in records if r["tier"] == tier and r["intended"] == e]
        out[e] = np.mean([r["predicted"] == e for r in sub]) if sub else float("nan")
    return out


def correct(results: dict, validation: dict, tier: str) -> dict:
    ss = _sensitivity_specificity(validation["confusion"])
    apparent = _apparent_recovery(results["records"], tier)
    per_class = {}
    corrected_vals = []
    for e in EMOTIONS:
        a, (sens, spec) = apparent[e], ss[e]
        c = rogan_gladen(a, sens, spec) if a == a else float("nan")
        per_class[e] = {
            "apparent": _round(a),
            "sensitivity": round(sens, 4),
            "corrected": _round(c),
        }
        if c == c:
            corrected_vals.append(c)
    apparent_macro = np.nanmean(list(apparent.values()))
    return {
        "tier": tier,
        "probe": validation.get("model", validation.get("probe")),
        "apparent_recovery": _round(float(apparent_macro)),
        "corrected_recovery": _round(float(np.mean(corrected_vals)) if corrected_vals else np.nan),
        "per_class": per_class,
    }


def _round(x: float) -> float | None:
    return None if x != x else round(float(x), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rogan-Gladen recovery correction")
    parser.add_argument("--results", default="results/paper/results.json")
    parser.add_argument("--probe-validation", default="results/paper/probe_validation_scene.json")
    parser.add_argument("--tier", default="emotion")
    parser.add_argument("--out", default="results/paper/corrected_recovery.json")
    args = parser.parse_args()

    results = json.loads(Path(args.results).read_text())
    validation = json.loads(Path(args.probe_validation).read_text())
    report = correct(results, validation, args.tier)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_class"}, indent=2))


if __name__ == "__main__":
    main()
