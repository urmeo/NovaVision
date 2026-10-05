"Recompute metrics, contrasts, and figures from an existing run's records."

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


def resummarize(results_path: str | Path, *, figures_dir: str | Path | None = None) -> dict:
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
    run_mod._write_figures(
        path.parent, records, payload["metrics"], conditions, figures_dir=figures_dir
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute summary from existing records")
    parser.add_argument("--results", default="outputs/results/results.json")
    parser.add_argument("--figures", default=None, help="override the figure directory")
    args = parser.parse_args()
    figures_dir = args.figures
    if (
        figures_dir is None
        and Path(args.results).resolve() == Path("outputs/results/results.json").resolve()
    ):
        figures_dir = "outputs/figures"
    payload = resummarize(args.results, figures_dir=figures_dir)
    health = payload["metrics"].get("probe_health", {})
    print(json.dumps({"reanalyzed": args.results, "probe_health": health}, indent=2))


if __name__ == "__main__":
    main()
