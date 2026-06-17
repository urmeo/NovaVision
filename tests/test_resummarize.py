import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import resummarize  # noqa: E402

from novavision.taxonomy import EMOTIONS  # noqa: E402


def _run_with_records(tmp_path):
    # A degenerate probe: predicts 'neutral' for everything (the real failure mode).
    records = []
    for tier in ("raw", "emotion", "affect", "scene"):
        for sk in range(1):
            for emotion in EMOTIONS:
                records.append(
                    {
                        "probe": "clip:test",
                        "tier": tier,
                        "content": "" if tier == "scene" else "a city street",
                        "intended": emotion,
                        "classified": None,
                        "seed": sk,
                        "predicted": "neutral",
                        "intended_valence": 0.2,
                        "intended_arousal": 0.6,
                        "recovered_valence": 0.05,
                        "recovered_arousal": 0.5,
                        "clip_t": 0.25,
                    }
                )
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"manifest": {"git_sha": "old"}, "records": records}))
    return path


def test_resummarize_adds_collapse_diagnostic_without_touching_records(tmp_path):
    path = _run_with_records(tmp_path)
    before = json.loads(path.read_text())["records"]

    payload = resummarize.resummarize(path)

    # Records are untouched — no images were regenerated.
    assert payload["records"] == before
    # The probe-collapse diagnostic is surfaced from the real records.
    health = payload["metrics"]["probe_health"]
    assert health["majority_label"] == "neutral"
    assert health["distinct_labels"] == 1
    assert health["majority_rate"] == 1.0
    # Provenance stays honest: original manifest kept, reanalysis stamped.
    assert "reanalysis" in payload["manifest"]
    assert payload["manifest"]["git_sha"] == "old"
    # Figures were refreshed from the records.
    assert (tmp_path / "figures" / "accuracy.png").exists()
