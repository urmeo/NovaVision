import json
from pathlib import Path

import make_submission  # scripts/ on sys.path via conftest.py

_RESULTS = Path(__file__).resolve().parents[1] / "results" / "paper" / "results.json"


def test_submission_from_committed_results_matches_schema():
    results = json.loads(_RESULTS.read_text())
    sub = make_submission.build_submission(results, "committed pilot")

    schema = json.loads(
        (
            Path(make_submission.__file__).resolve().parents[1]
            / "benchmark"
            / "submission.schema.json"
        ).read_text()
    )
    for key in schema["required"]:
        assert key in sub
    assert len(sub["conditions"]) >= 2
    for cond in sub["conditions"].values():
        assert 0.0 <= cond["accuracy"] <= 1.0
        assert len(cond["accuracy_ci"]) == 2
        assert cond["n"] >= 1
    assert sub["provenance"]["git_sha"]  # copied from the manifest, not invented


def test_submission_never_invents_a_condition():
    # Only tiers actually present in the run may appear in the submission.
    results = {
        "manifest": {"git_sha": "abc", "config": {"seeds": 1, "base_seed": 0}},
        "metrics": {"raw": {"accuracy": 0.1, "accuracy_ci": [0.0, 0.3], "n": 7}, "chance": 0.14},
    }
    sub = make_submission.build_submission(results, "sparse")
    assert set(sub["conditions"]) == {"raw"}
