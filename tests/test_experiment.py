from novavision.experiments import run
from novavision.taxonomy import EMOTIONS, prior


def _records(tier: str, correct: bool):
    rows = []
    for i, emotion in enumerate(EMOTIONS):
        iv, ia = prior(emotion)
        predicted = emotion if correct or i % 2 == 0 else EMOTIONS[(i + 1) % len(EMOTIONS)]
        rows.append(
            {
                "text": f"sample {i}",
                "intended": emotion,
                "tier": tier,
                "predicted": predicted,
                "intended_valence": iv,
                "intended_arousal": ia,
                "recovered_valence": iv * 0.5,
                "recovered_arousal": ia * 0.5,
                "clip_t": 0.25,
            }
        )
    return rows


def test_summarize():
    records = _records("affect", correct=True) + _records("raw", correct=False)
    metrics = run._summarize(records, ("raw", "affect"))
    assert metrics["affect"]["accuracy"] == 1.0
    assert metrics["raw"]["accuracy"] < 1.0
    assert metrics["affect"]["n"] == len(EMOTIONS)


def test_write_creates_outputs(tmp_path):
    records = _records("affect", correct=True)
    metrics = run._summarize(records, ("affect",))
    run._write(str(tmp_path), "null", "artistic", 0, records, metrics, 0.5)
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "figures" / "accuracy.png").exists()
    assert (tmp_path / "figures" / "confusion_affect.png").exists()


def test_run_experiment_with_stubs(tmp_path, monkeypatch):
    from novavision.affect.analyzer import EmotionAnalysis
    from novavision.eval.clip_affect import Recovery

    class FakeAnalyzer:
        def __init__(self, *a, **k):
            pass

        def analyze(self, text):
            return EmotionAnalysis("joy", 0.9, 0.5, 0.6, 1.0, {"joy": 0.9})

    class FakeCLIP:
        def __init__(self, *a, **k):
            pass

        def recover(self, image):
            return Recovery("joy", 0.4, 0.6, {"joy": 0.9})

        def clip_t(self, image, text):
            return 0.3

    monkeypatch.setattr(run, "EmotionAnalyzer", FakeAnalyzer)
    monkeypatch.setattr(run, "CLIPAffect", FakeCLIP)

    result = run.run_experiment(backend="null", limit=4, out=str(tmp_path))
    assert "classification_accuracy" in result
    assert (tmp_path / "results.json").exists()
    for tier in ("raw", "emotion", "affect"):
        assert result["metrics"][tier]["n"] == 4
