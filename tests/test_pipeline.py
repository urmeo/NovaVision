from novavision.affect.analyzer import EmotionAnalysis
from novavision.generation import NullBackend
from novavision.pipeline import NovaVision


class StubAnalyzer:
    def __init__(self, emotion="joy"):
        self.emotion = emotion

    def analyze(self, text):
        return EmotionAnalysis(self.emotion, 0.9, 0.8, 0.7, 1.0, {self.emotion: 0.9})


def test_run_produces_result():
    nv = NovaVision(backend=NullBackend(), analyzer=StubAnalyzer())
    result = nv.run("good news", tier="affect", seed=3)
    assert result.image.size == (512, 512)
    assert result.seed == 3
    assert result.backend == "null"
    assert "warm vibrant palette" in result.prompt


def test_auto_run_skips_neutral():
    nv = NovaVision(backend=NullBackend(), analyzer=StubAnalyzer(emotion="neutral"))
    assert nv.auto_run("the table is wooden").tier == "raw"


def test_auto_run_conditions_emotion():
    nv = NovaVision(backend=NullBackend(), analyzer=StubAnalyzer(emotion="joy"))
    assert nv.auto_run("what a wonderful day").tier == "affect"


def test_build_pipeline_returns_lazy_null_pipeline(monkeypatch):
    monkeypatch.setenv("BACKEND", "null")
    from novavision.pipeline import build_pipeline

    nv = build_pipeline()
    assert nv.backend.name == "null"
    assert nv.analyzer is not None  # constructed, no model loaded yet
