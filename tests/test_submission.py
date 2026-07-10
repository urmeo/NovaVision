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


def test_submission_propagates_benchmark_sha256_from_config():
    # benchmark_sha256 is nested under manifest.config (build_manifest puts every
    # run kwarg there); the submission must read it from config, not the top level.
    results = {
        "manifest": {
            "git_sha": "abc",
            "config": {"seeds": 1, "base_seed": 0, "benchmark_sha256": "DEADBEEF"},
        },
        "metrics": {"raw": {"accuracy": 0.1, "accuracy_ci": [0.0, 0.3], "n": 7}},
    }
    sub = make_submission.build_submission(results, "hashed")
    assert sub["provenance"]["benchmark_sha256"] == "DEADBEEF"


def test_submission_never_invents_a_condition():
    # Only tiers actually present in the run may appear in the submission.
    results = {
        "manifest": {"git_sha": "abc", "config": {"seeds": 1, "base_seed": 0}},
        "metrics": {"raw": {"accuracy": 0.1, "accuracy_ci": [0.0, 0.3], "n": 7}, "chance": 0.14},
    }
    sub = make_submission.build_submission(results, "sparse")
    assert set(sub["conditions"]) == {"raw"}


def test_submission_omits_probe_health_when_degenerate():
    # A run without a valid probe_health must not emit a schema-invalid 0 placeholder.
    results = {
        "manifest": {"git_sha": "abc", "config": {"seeds": 1, "base_seed": 0}},
        "metrics": {
            "raw": {"accuracy": 0.1, "accuracy_ci": [0.0, 0.3], "n": 7},
            "probe_health": {"distinct_labels": 0, "n_labels": 7},
        },
    }
    sub = make_submission.build_submission(results, "degenerate")
    assert "probe_health" not in sub  # omitted, not emitted as invalid 0
