"""Render results.json into the paper's generated tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from novavision.eval.metrics import cohens_h, holm_bonferroni
from novavision.experiments.run import ALL_CONDITIONS as CONDITIONS


def _fmt(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.3f}"


def _rho(m: dict, key: str) -> str:
    """Correlation with its bootstrap CI when available, never a bare 3-decimal."""
    rho = m.get(key)
    ci = m.get(f"{key}_ci")
    lo = ci[0] if ci else None
    # A degenerate correlation writes NaN, which becomes null after a results.json
    # round-trip; render the CI only when the lower bound is a real number.
    if lo is not None and not (isinstance(lo, float) and lo != lo):
        return f"{_fmt(rho)} [{ci[0]:.2f}, {ci[1]:.2f}]"
    return _fmt(rho)


def _shuffled_note(metrics: dict) -> str:
    """Circularity check: does any conditioning tier beat the shuffled-label null?"""
    parts = []
    for cond in ("naive", "emotion", "affect"):
        sc = (metrics.get(cond) or {}).get("shuffled_control") or {}
        # An older/edited results.json may carry the control dict without a
        # p-value; skip that tier rather than KeyError out of report generation.
        p = sc.get("p_value")
        if p is not None:
            parts.append(f"{cond} p={p:.2f}")
    if not parts:
        return ""
    # The null mean is shared across tiers; take it from whichever tier actually
    # carries it, so a partial dict on one tier does not drop the annotation.
    null_mean = next(
        (
            nm
            for cond in ("naive", "emotion", "affect")
            for nm in [((metrics.get(cond) or {}).get("shuffled_control") or {}).get("null_mean")]
            if nm is not None
        ),
        None,
    )
    base = f" (null mean {null_mean:.3f})" if null_mean is not None else ""
    return (
        "**Shuffled-label control:** one-sided permutation test of recovery vs randomly "
        f"reassigned target emotions{base}: {', '.join(parts)}. A p near 1 means recovery is "
        "indistinguishable from the circularity baseline, i.e. not above chance label agreement."
    )


def metrics_table(metrics: dict) -> str:
    head = "| Condition | Accuracy [95% CI] | Macro-F1 | Valence ρ†‡ | Arousal ρ†‡ | CLIP-T | n |"
    rule = "|" + "---|" * 7
    lines = [head, rule]
    for cond in CONDITIONS:
        m = metrics.get(cond)
        if not m:
            continue
        lo, hi = m["accuracy_ci"]
        # A degenerate (n<2) condition writes null CI bounds; _fmt renders those.
        acc = f"{_fmt(m['accuracy'])} [{_fmt(lo)}, {_fmt(hi)}]"
        lines.append(
            f"| {cond} | {acc} | {_fmt(m['macro_f1'])} | {_rho(m, 'valence_rho')} | "
            f"{_rho(m, 'arousal_rho')} | {_fmt(m['clip_t'])} | {m['n']} |"
        )
    chance = metrics.get("chance")
    base = metrics.get("raw", {}).get("majority_baseline")
    if chance is not None:
        note = f"\nChance = {chance:.3f} (1/7)"
        if base is not None:
            note += f"; majority-class baseline = {base:.3f}"
        note += (
            ". A probe collapsed onto one label scores here, so recovery is only "
            "informative *above* it."
        )
        lines.append(note)
    health = metrics.get("probe_health")
    if health:
        lines.append(
            f"**Probe health:** the probe used {health['distinct_labels']}/{health['n_labels']} "
            f"emotion labels across the conditioning tiers, predicting "
            f"'{health['majority_label']}' for {health['majority_rate']:.0%} of items; every "
            "recovery number must be read against this degeneracy."
        )
    shuffled = _shuffled_note(metrics)
    if shuffled:
        lines.append(shuffled)
    lines.append(
        "† On neutral content the intended valence/arousal is the per-emotion prior, so these "
        "correlations reflect between-emotion separation, not within-emotion grounding."
    )
    lines.append(
        "‡ ρ is shown with its bootstrap 95% CI; at small n the interval spans most of [-1, 1], "
        "so the point estimate should not be over-read."
    )
    return "\n".join(lines)


def contrasts_table(contrasts: dict, metrics: dict | None = None) -> str:
    head = "| Contrast | Δ accuracy | 95% CI | Cohen's h | p |"
    rule = "|" + "---|" * 5
    lines = [head, rule]
    metrics = metrics or {}
    for name, c in contrasts.items():
        ci = f"[{_fmt(c['ci_low'])}, {_fmt(c['ci_high'])}]"
        diff = c["mean_diff"]
        diff_s = f"{diff:+.3f}" if isinstance(diff, (int, float)) else _fmt(diff)
        label = name.replace("_", " ")
        lines.append(
            f"| {label} | {diff_s} | {ci} | {_cohens_h(name, metrics)} | {_fmt(c['p_value'])} |"
        )
    return "\n".join(lines)


def _cohens_h(contrast_name: str, metrics: dict) -> str:
    """Effect size for a tier-vs-tier contrast, from the two tiers' accuracies."""
    hi, _, lo = contrast_name.partition("_vs_")
    a_hi = (metrics.get(hi) or {}).get("accuracy")
    a_lo = (metrics.get(lo) or {}).get("accuracy")
    if a_hi is None or a_lo is None:
        return "n/a"
    return f"{cohens_h(a_hi, a_lo):+.3f}"


def _holm_note(metrics: dict) -> str:
    """Family-wise Holm correction over the conditioning tiers' shuffled-label p-values."""
    pvals = {}
    for cond in ("naive", "emotion", "affect"):
        sc = (metrics.get(cond) or {}).get("shuffled_control")
        if sc and sc.get("p_value") is not None:
            pvals[cond] = sc["p_value"]
    if not pvals:
        return ""
    adjusted = holm_bonferroni(pvals)
    parts = [f"{cond} p_adj={adjusted[cond]['p_adjusted']:.2f}" for cond in pvals]
    survivors = [cond for cond in pvals if adjusted[cond]["reject"]]
    # The conclusion is derived from the adjusted decisions, never hardcoded, so a
    # future run where a tier clears the correction reports that truthfully.
    verdict = (
        "no tier survives at 0.05, so the null holds under multiple-comparison control"
        if not survivors
        else f"the null is rejected for {', '.join(survivors)} under multiple-comparison control"
    )
    return (
        "**Family-wise correction (Holm):** across the conditioning tiers, "
        f"{', '.join(parts)}; {verdict}."
    )


def render(results: dict) -> str:
    metrics = results["metrics"]
    holm = _holm_note(metrics)
    parts = [
        "<!-- generated by scripts/report.py; do not edit by hand -->",
        "**Table 1.** Affect recovery by conditioning condition.",
        "",
        metrics_table(metrics),
        *([holm] if holm else []),
        "",
        "**Table 2.** Paired contrasts on per-item recovery (bootstrap).",
        "",
        contrasts_table(results["contrasts"], metrics),
        "",
    ]
    return "\n".join(parts) + "\n"


def inject(paper_path: str | Path, tables: str) -> bool:
    """Replace the content between the TABLES markers in the paper."""
    path = Path(paper_path)
    if not path.exists():
        return False
    text = path.read_text()
    start, end = "<!--TABLES-->", "<!--/TABLES-->"
    if start not in text or end not in text:
        return False
    head = text.split(start)[0]
    tail = text.split(end, 1)[1]
    path.write_text(f"{head}{start}\n{tables}{end}{tail}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Render results as markdown tables")
    parser.add_argument("--results", default="results/paper/results.json")
    parser.add_argument("--out", default="paper/tables.md")
    parser.add_argument("--paper", default="paper/paper.md", help="inject tables into this file")
    args = parser.parse_args()

    results = json.loads(Path(args.results).read_text())
    table = render(results)
    Path(args.out).write_text(table)
    if inject(args.paper, table):
        print(f"Injected tables into {args.paper}")
    print(table)


if __name__ == "__main__":
    main()
