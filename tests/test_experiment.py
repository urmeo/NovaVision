from novavision.eval.probes import Recovery
from novavision.experiments import run
from novavision.taxonomy import EMOTIONS


def _records(tier: str, correct: bool):
    rows = []
    for sk in range(2):
        for i, emotion in enumerate(EMOTIONS):
            predicted = emotion if correct else EMOTIONS[(i + 1) % len(EMOTIONS)]
            rows.append(
                {
                    "probe": "clip:test",
                    "tier": tier,
                    "content": "a city street",
                    "intended": emotion,
                    "seed": sk,
                    "predicted": predicted,
                    "intended_valence": 0.5,
                    "intended_arousal": 0.6,
                    "recovered_valence": 0.4 if correct else -0.4,
                    "recovered_arousal": 0.5,
                    "clip_t": 0.25,
                }
            )
    return rows


def test_summarize_reports_ci_and_chance():
    records = _records("affect", correct=True) + _records("raw", correct=False)
    metrics = run._summarize(records)
    assert metrics["affect"]["accuracy"] == 1.0
    assert metrics["raw"]["accuracy"] == 0.0
    assert len(metrics["affect"]["accuracy_ci"]) == 2
    assert metrics["chance"] == round(1 / len(EMOTIONS), 4)


def test_contrasts_detect_lift():
    records = _records("emotion", correct=True) + _records("raw", correct=False)
    contrasts = run._contrasts(records)
    assert contrasts["emotion_vs_raw"]["mean_diff"] == 1.0
    assert contrasts["emotion_vs_raw"]["p_value"] < 0.05


def test_run_experiment_content_track(tmp_path, monkeypatch):
    class FakeProbe:
        name = "clip:fake"

        def __init__(self, *a, **k):
            pass

        def recover(self, image):
            return Recovery("joy", 0.4, 0.6, {e: 1.0 if e == "joy" else 0.0 for e in EMOTIONS})

        def clip_t(self, image, text):
            return 0.3

    monkeypatch.setattr(run, "CLIPProbe", FakeProbe)
    result = run.run_experiment(backend="null", contents=2, seeds=1, out=str(tmp_path))

    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "figures" / "accuracy.png").exists()
    for tier in ("raw", "emotion", "affect", "scene"):
        assert tier in result["metrics"]
