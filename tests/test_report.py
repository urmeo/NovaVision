import report  # scripts/ on sys.path via conftest.py


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


def test_nan_renders_as_placeholder():
    assert report._fmt(float("nan")) == "n/a"
    assert report._fmt(None) == "n/a"


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


def test_shuffled_note_survives_partial_control_dict():
    # An older results.json may carry shuffled_control without p_value/null_mean.
    metrics = {
        "emotion": {"shuffled_control": {}},  # present but empty
        "affect": {"shuffled_control": {"p_value": 0.14, "null_mean": 0.142}},
    }
    note = report._shuffled_note(metrics)
    assert "affect p=0.14" in note
    assert "null mean 0.142" in note  # falls back to the tier that has it


def test_shuffled_note_empty_when_no_pvalues():
    assert report._shuffled_note({"emotion": {"shuffled_control": {"null_mean": 0.1}}}) == ""


def test_tables_tolerate_null_bounds():
    # A degenerate (n<2) run writes null CI/contrast bounds via json_safe; the
    # tables must render "n/a", not crash on None.__format__.
    metrics = {
        "raw": {
            "accuracy": 0.14,
            "accuracy_ci": [None, None],
            "macro_f1": None,
            "valence_rho": 0.0,
            "arousal_rho": 0.0,
            "clip_t": None,
            "n": 1,
        },
        "chance": 0.143,
    }
    assert "n/a" in report.metrics_table(metrics)
    contrasts = {
        "emotion_vs_raw": {"ci_low": None, "ci_high": None, "mean_diff": None, "p_value": None}
    }
    assert "n/a" in report.contrasts_table(contrasts)
