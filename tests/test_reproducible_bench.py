"The committed benchmark numbers must regenerate from the committed records."

import json
from pathlib import Path

from novavision.experiments import run as run_mod

RESULTS = Path(__file__).resolve().parents[1] / "outputs" / "results" / "results.json"


def _conditions_for(records):
    tiers = {r["tier"] for r in records}
    track = "text" if "shuffled" in tiers else "content"
    return run_mod.CONDITIONS[track]


def _canonical(obj):
    """RFC-8259-safe, order-independent form for exact comparison."""
    return json.loads(json.dumps(run_mod.json_safe(obj), sort_keys=True))


def test_committed_metrics_regenerate_from_records():
    payload = json.loads(RESULTS.read_text())
    records = payload["records"]
    conditions = _conditions_for(records)

    metrics = run_mod._summarize(records, conditions)
    contrasts = run_mod._contrasts(records)

    assert _canonical(metrics) == _canonical(payload["metrics"])
    assert _canonical(contrasts) == _canonical(payload["contrasts"])


def test_committed_pilot_is_an_honest_null():
    """Guardrail on the headline reading: no tier clears the shuffled-label control."""
    payload = json.loads(RESULTS.read_text())
    metrics = payload["metrics"]
    for tier in ("raw", "emotion", "affect"):
        assert metrics[tier]["shuffled_control"]["p_value"] > 0.05

    assert metrics["probe_health"]["distinct_labels"] <= 2
    assert metrics["raw"]["accuracy"] == metrics["chance"]
