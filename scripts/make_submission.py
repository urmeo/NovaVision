"""Turn a run's results.json into a schema-valid benchmark submission.

Reads only committed, script-produced fields, so a submission cannot claim a
number the run did not measure. Validate the output against
``benchmark/submission.schema.json`` before opening a leaderboard PR.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from novavision.experiments.run import ALL_CONDITIONS


def build_submission(results: dict, system: str) -> dict:
    manifest = results["manifest"]
    cfg = manifest["config"]
    metrics = results["metrics"]

    conditions = {}
    for tier in ALL_CONDITIONS:
        m = metrics.get(tier)
        if not m:
            continue
        sc = m.get("shuffled_control") or {}
        conditions[tier] = {
            "accuracy": m["accuracy"],
            "accuracy_ci": m["accuracy_ci"],
            "shuffled_p": sc.get("p_value"),
            "n": m["n"],
        }

    submission = {
        "system": system,
        "track": cfg.get("track", "content"),
        "generator": cfg.get("diffusion_model", "unknown"),
        "probe": cfg.get("probe", "unknown"),
        "conditions": conditions,
        "provenance": {
            "git_sha": manifest.get("git_sha", "unknown"),
            "seeds": cfg.get("seeds", 1),
            "base_seed": cfg.get("base_seed", 0),
            "benchmark_sha256": cfg.get("benchmark_sha256"),
        },
    }
    # probe_health is optional in the schema; include it only when the run carries
    # a valid diagnostic, never a schema-invalid 0 placeholder.
    health = metrics.get("probe_health") or {}
    if health.get("distinct_labels") and health.get("n_labels"):
        submission["probe_health"] = {
            "distinct_labels": health["distinct_labels"],
            "n_labels": health["n_labels"],
        }
    return submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a benchmark submission from a run")
    parser.add_argument("--results", default="results/paper/results.json")
    parser.add_argument("--system", required=True, help="name of the submitted system")
    parser.add_argument("--out", default="submission.json")
    args = parser.parse_args()

    results = json.loads(Path(args.results).read_text())
    submission = build_submission(results, args.system)
    Path(args.out).write_text(json.dumps(submission, indent=2))
    print(f"Wrote {args.out}. Validate against benchmark/submission.schema.json before submitting.")


if __name__ == "__main__":
    main()
