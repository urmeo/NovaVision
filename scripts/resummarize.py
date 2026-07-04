"""Recompute metrics, contrasts, and figures from an existing run's records.

Refreshes a ``results.json`` produced by an earlier analysis version with the
current diagnostics (``probe_health``, majority-class baseline, prediction
collapse, valence/arousal bootstrap CIs) **without regenerating any images**;
the per-image records are the original measured outputs, so the headline numbers
are preserved and merely re-derived. A tier that was never measured (e.g. one
added to the code after the run) stays absent. Provenance stays honest: the
original manifest is kept and a ``reanalysis`` stamp records that only the
summary was recomputed, at the current commit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from novavision.experiments import run as run_mod
from novavision.experiments.manifest import git_sha, package_version


def _conditions_for(records: list[dict]) -> tuple[str, ...]:
    tiers = {r["tier"] for r in records}
    track = "text" if "shuffled" in tiers else "content"
    return run_mod.CONDITIONS[track]


def resummarize(results_path: str | Path) -> dict:
    path = Path(results_path)
    payload = json.loads(path.read_text())
    records = payload["records"]
    conditions = _conditions_for(records)

    payload["metrics"] = run_mod._summarize(records, conditions)
    payload["contrasts"] = run_mod._contrasts(records)
    payload.setdefault("manifest", {})["reanalysis"] = {
        "git_sha": git_sha(),
        "note": "metrics/diagnostics recomputed from the original records; no images regenerated",
        "packages": {pkg: package_version(pkg) for pkg in ("numpy",)},
    }
    run_mod.dump_results(payload, path)
    run_mod._write_figures(path.parent, records, payload["metrics"], conditions)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute summary from existing records")
    parser.add_argument("--results", default="results/paper/results.json")
    args = parser.parse_args()
    payload = resummarize(args.results)
    health = payload["metrics"].get("probe_health", {})
    print(json.dumps({"reanalyzed": args.results, "probe_health": health}, indent=2))


if __name__ == "__main__":
    main()
