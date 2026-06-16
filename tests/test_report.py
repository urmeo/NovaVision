import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import report  # noqa: E402


def _results():
    return {
        "metrics": {
            "raw": {
                "accuracy": 0.14,
                "accuracy_ci": [0.1, 0.2],
                "macro_f1": 0.1,
                "valence_rho": 0.0,
                "arousal_rho": 0.0,
                "clip_t": 0.2,
                "n": 70,
            },
            "affect": {
                "accuracy": 0.55,
                "accuracy_ci": [0.45, 0.65],
                "macro_f1": 0.5,
                "valence_rho": 0.3,
                "arousal_rho": 0.2,
                "clip_t": 0.25,
                "n": 70,
            },
            "chance": 0.143,
        },
        "contrasts": {
            "affect_vs_raw": {"mean_diff": 0.41, "ci_low": 0.3, "ci_high": 0.5, "p_value": 0.001}
        },
    }


def test_render_includes_ci_and_significance():
    out = report.render(_results())
    assert "[0.450, 0.650]" in out  # CI rendered
    assert "+0.410" in out  # contrast delta
    assert "chance" in out.lower()


def test_nan_renders_as_dash():
    assert report._fmt(float("nan")) == "–"
    assert report._fmt(None) == "–"


def test_inject_replaces_between_markers(tmp_path):
    paper = tmp_path / "paper.md"
    paper.write_text("# Title\n\n<!--TABLES-->\nold\n<!--/TABLES-->\n\n## Next\n")
    assert report.inject(paper, "NEW TABLE\n") is True
    text = paper.read_text()
    assert "NEW TABLE" in text and "old" not in text
    assert "## Next" in text
    # Idempotent: a second injection does not duplicate.
    report.inject(paper, "NEWER\n")
    assert paper.read_text().count("<!--TABLES-->") == 1
