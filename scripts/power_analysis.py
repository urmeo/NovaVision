"""How large must the powered run be to detect a controllability effect?

A null is only meaningful if the study was powered to detect a real effect. This
turns the measured probe ceiling into a sample-size answer: given that the probe
recovers the intended emotion at only ``ceiling`` on real scenes (``make
validate-probe-scene``), what per-tier ``n`` is needed to distinguish a
conditioning effect from the chance/shuffled-label baseline?

Model. A conditioning effect of strength ``s`` (the fraction of generated images
that genuinely convey the intended emotion) yields recovery accuracy

    p1 = s * ceiling + (1 - s) * chance,

because a conveyed emotion is recovered at the probe's ceiling rate and a
non-conveyed one at chance. The test mirrors the benchmark's own control: reject
when observed accuracy clears the one-sided 95th percentile of the chance
(shuffled-label) null. Power is estimated by simulation, so it needs no scipy and
matches the permutation test the harness actually runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CHANCE = 1.0 / 7.0
ALPHA = 0.05
TARGET_POWER = 0.80
EFFECTS = (0.2, 0.3, 0.4, 0.5, 0.7, 1.0)  # conditioning strengths s
_SIM = 4000


def _critical_count(n: int, chance: float, rng: np.random.Generator) -> int:
    """Smallest correct-count that clears the one-sided chance null at ALPHA."""
    null = rng.binomial(n, chance, _SIM)
    return int(np.quantile(null, 1 - ALPHA)) + 1


def power_at(n: int, p1: float, chance: float, rng: np.random.Generator) -> float:
    crit = _critical_count(n, chance, rng)
    return float(np.mean(rng.binomial(n, p1, _SIM) >= crit))


def min_n_for_power(
    p1: float, chance: float, rng: np.random.Generator, n_max: int = 5000
) -> int | None:
    """Smallest n reaching TARGET_POWER, or None if unreachable within n_max."""
    if p1 <= chance:
        return None
    for n in range(10, n_max + 1, 10):
        if power_at(n, p1, chance, rng) >= TARGET_POWER:
            return n
    return None


def analyze(ceiling: float, planned_n: int) -> dict:
    rng = np.random.default_rng(0)
    rows = []
    for s in EFFECTS:
        p1 = s * ceiling + (1 - s) * CHANCE
        rows.append(
            {
                "effect_strength": s,
                "recovery_accuracy": round(p1, 4),
                "n_for_80pct_power": min_n_for_power(p1, CHANCE, rng),
                "power_at_planned_n": round(power_at(planned_n, p1, CHANCE, rng), 3),
            }
        )
    return {
        "chance": round(CHANCE, 4),
        "probe_ceiling": round(ceiling, 4),
        "alpha": ALPHA,
        "target_power": TARGET_POWER,
        "planned_n": planned_n,
        "rows": rows,
    }


def _format(report: dict) -> str:
    lines = [
        f"Probe ceiling {report['probe_ceiling']} (real-scene recovery); "
        f"chance {report['chance']}; alpha {report['alpha']}; "
        f"target power {report['target_power']}.",
        "",
        "| Effect s | Recovery acc | n for 80% power | Power at planned n={} |".format(
            report["planned_n"]
        ),
        "|---|---|---|---|",
    ]
    for r in report["rows"]:
        n = r["n_for_80pct_power"]
        lines.append(
            f"| {r['effect_strength']:.1f} | {r['recovery_accuracy']:.3f} | "
            f"{n if n is not None else 'unreachable'} | {r['power_at_planned_n']:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample-size analysis for the powered run")
    parser.add_argument(
        "--probe-validation",
        default="results/paper/probe_validation_scene_l14.json",
        help="probe validation JSON; its accuracy is the recovery ceiling",
    )
    parser.add_argument("--planned-n", type=int, default=420, help="per-tier n of the planned run")
    parser.add_argument("--out", default="results/paper/power_analysis.json")
    args = parser.parse_args()

    ceiling = json.loads(Path(args.probe_validation).read_text())["accuracy"]
    report = analyze(ceiling, args.planned_n)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(_format(report))


if __name__ == "__main__":
    main()
