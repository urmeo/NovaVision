import correct_recovery as cr  # scripts/ on sys.path via conftest.py

from novavision.taxonomy import EMOTIONS


def _identity_confusion(n_per_class=10):
    # A perfect probe: every true class recovered exactly.
    return [
        [n_per_class if i == j else 0 for j in range(len(EMOTIONS))] for i in range(len(EMOTIONS))
    ]


def _records(recovery_rate):
    # `recovery_rate` correct predictions per class, over 10 items each.
    recs = []
    for e in EMOTIONS:
        for k in range(10):
            pred = e if k < recovery_rate * 10 else "neutral" if e != "neutral" else "anger"
            recs.append({"tier": "emotion", "intended": e, "predicted": pred})
    return recs


def test_perfect_probe_leaves_recovery_unchanged():
    results = {"records": _records(0.5)}
    validation = {"confusion": _identity_confusion(), "model": "perfect"}
    out = cr.correct(results, validation, "emotion")
    # sens=spec=1 -> corrected == apparent.
    assert out["corrected_recovery"] == out["apparent_recovery"]


def test_sensitivity_specificity_from_confusion():
    ss = cr._sensitivity_specificity(_identity_confusion())
    for e in EMOTIONS:
        sens, spec = ss[e]
        assert sens == 1.0 and spec == 1.0


def test_zero_support_class_serializes_null_not_nan():
    import json

    # A class the probe never saw has undefined sensitivity; it must become JSON
    # null, never a bare NaN token (the results.json strict-JSON contract).
    n = len(EMOTIONS)
    conf = [[0] * n for _ in range(n)]
    for i, e in enumerate(EMOTIONS):
        if e != "neutral":
            conf[i][i] = 10  # neutral's true row/col stay empty -> zero support
    out = cr.correct({"records": _records(0.5)}, {"confusion": conf, "model": "x"}, "emotion")
    assert out["per_class"]["neutral"]["sensitivity"] is None
    json.dumps(out, allow_nan=False)  # raises if any bare NaN leaked through


def test_correction_on_committed_pilot_reinforces_null():
    import json
    from pathlib import Path

    root = Path(cr.__file__).resolve().parents[1] / "results" / "paper"
    results = json.loads((root / "results.json").read_text())
    validation = json.loads((root / "probe_validation_scene.json").read_text())
    out = cr.correct(results, validation, "emotion")
    # Correcting for the B/32 probe's measured error does not lift recovery above chance.
    assert out["corrected_recovery"] <= 0.2
