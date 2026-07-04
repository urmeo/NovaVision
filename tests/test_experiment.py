from pathlib import Path

import pytest

from novavision.affect.analyzer import EmotionAnalysis
from novavision.eval.probes import Recovery
from novavision.experiments import run
from novavision.taxonomy import EMOTIONS

FIXTURE = Path(__file__).parent / "fixtures" / "affectbench_sample.csv"


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
                    "classified": None,
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
    metrics = run._summarize(records, run.CONDITIONS["content"])
    assert metrics["affect"]["accuracy"] == 1.0
    assert metrics["raw"]["accuracy"] == 0.0
    assert len(metrics["affect"]["accuracy_ci"]) == 2
    assert metrics["chance"] == round(1 / len(EMOTIONS), 4)


def test_summarize_reports_probe_collapse_and_baseline():
    # A degenerate probe: predicts the WRONG-but-constant label for every item.
    records = _records("raw", correct=False)
    metrics = run._summarize(records, run.CONDITIONS["content"])
    # majority-class baseline is reported next to accuracy so chance is not mistaken for signal
    assert "majority_baseline" in metrics["raw"]
    # per-tier prediction-collapse diagnostic
    assert metrics["raw"]["collapse"]["distinct"] >= 1
    # global probe-health diagnostic over the conditioning tiers
    health = metrics["probe_health"]
    assert health["n_labels"] == len(EMOTIONS)
    assert 0.0 <= health["majority_rate"] <= 1.0
    # VA correlations carry bootstrap CIs, not bare point estimates
    assert len(metrics["raw"]["valence_rho_ci"]) == 2
    assert len(metrics["raw"]["arousal_rho_ci"]) == 2


def test_contrasts_detect_lift():
    records = _records("emotion", correct=True) + _records("raw", correct=False)
    contrasts = run._contrasts(records)
    assert contrasts["emotion_vs_raw"]["mean_diff"] == 1.0
    assert contrasts["emotion_vs_raw"]["p_value"] < 0.05


def test_shuffle_emotion_avoids_gold():
    for e in EMOTIONS:
        assert run._shuffle_emotion(e, 3) != e


class FakeProbe:
    name = "clip:fake"

    def __init__(self, *a, **k):
        pass

    def recover(self, image):
        return Recovery("joy", 0.4, 0.6, {e: 1.0 if e == "joy" else 0.0 for e in EMOTIONS})

    def clip_t(self, image, text):
        return 0.3


def test_run_experiment_content_track(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "CLIPProbe", FakeProbe)
    result = run.run_experiment(backend="null", contents=2, seeds=1, out=str(tmp_path))

    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "figures" / "accuracy.png").exists()
    for tier in ("raw", "emotion", "affect", "scene"):
        assert tier in result["metrics"]
    # results.json must be standards-compliant JSON (no bare NaN tokens).
    import json

    txt = (tmp_path / "results.json").read_text()
    json.loads(txt, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite")))


def test_json_safe_replaces_non_finite():
    out = run.json_safe({"a": float("nan"), "b": [1.0, float("inf")], "c": "x", "d": 2})
    assert out == {"a": None, "b": [1.0, None], "c": "x", "d": 2}
    __import__("json").dumps(out, allow_nan=False)  # must not raise


def test_contents_zero_means_zero_not_all(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "CLIPProbe", FakeProbe)
    result = run.run_experiment(backend="null", contents=0, seeds=1, out=str(tmp_path))
    # 0 subjects -> no conditioning-tier records; only the scene floor remains.
    assert "raw" not in result["metrics"] and "scene" in result["metrics"]


def test_run_experiment_text_track(tmp_path, monkeypatch):
    class FakeAnalyzer:
        def __init__(self, *a, **k):
            pass

        def analyze(self, text):
            return EmotionAnalysis("joy", 0.9, 0.5, 0.6, 1.0, {"joy": 0.9})

    monkeypatch.setattr(run, "CLIPProbe", FakeProbe)
    monkeypatch.setattr(run, "EmotionAnalyzer", FakeAnalyzer)
    result = run.run_experiment(
        backend="null", track="text", benchmark=str(FIXTURE), limit=4, seeds=1, out=str(tmp_path)
    )

    for tier in ("raw", "emotion", "affect", "shuffled"):
        assert tier in result["metrics"]
    assert "classification_accuracy" in result["metrics"]


def test_text_track_requires_benchmark(monkeypatch):
    monkeypatch.setattr(run, "CLIPProbe", FakeProbe)
    with pytest.raises(ValueError):
        run.run_experiment(backend="null", track="text", seeds=1, out="/tmp/x")


def test_seed_injective_within_guarded_domain():
    seen = set()
    for ci in range(run.SEED_MAX_ITEMS):
        for ei in range(len(EMOTIONS)):
            for sk in range(run.SEED_MAX_SEEDS):
                seen.add(run._seed(0, ci, ei, sk))
    assert len(seen) == run.SEED_MAX_ITEMS * len(EMOTIONS) * run.SEED_MAX_SEEDS


def test_seed_domain_guard():
    run._check_seed_domain(run.SEED_MAX_ITEMS, run.SEED_MAX_SEEDS)  # boundary is fine
    for items, seeds in ((run.SEED_MAX_ITEMS + 1, 1), (1, run.SEED_MAX_SEEDS + 1)):
        with pytest.raises(ValueError):
            run._check_seed_domain(items, seeds)


def test_coverage_override_forces_blend_weight(monkeypatch):
    from novavision.affect.analyzer import EmotionAnalyzer
    from novavision.affect.lexicon import AffectScore

    class _Lex:
        def score(self, text):
            return AffectScore(1.0, 1.0, 0.5)  # coverage 0.5 would normally blend

    a = EmotionAnalyzer(lexicon=_Lex(), coverage_override=0.0)  # prior only
    a._classifier = lambda text, **kw: [[{"label": "joy", "score": 0.9}]]
    from novavision.taxonomy import prior

    pv, pa = prior("joy")
    res = a.analyze("anything")
    assert (res.valence, res.arousal) == (round(pv, 4), round(pa, 4))


def test_coverage_override_rejects_out_of_range():
    from novavision.affect.analyzer import EmotionAnalyzer

    with pytest.raises(ValueError, match="coverage_override"):
        EmotionAnalyzer(coverage_override=1.5)


def test_resume_reuses_checkpoint_and_cleans_up(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "CLIPProbe", FakeProbe)
    out = tmp_path / "run"
    run.run_experiment(backend="null", contents=1, seeds=1, out=str(out), resume=True)
    # A completed run removes its checkpoint; results.json supersedes it.
    assert not (out / "records.jsonl").exists()
    assert (out / "results.json").exists()


def test_resume_skips_completed_records(tmp_path):
    import json as _json

    from novavision.experiments.run import _Checkpoint

    path = tmp_path / "records.jsonl"
    rec = {"tier": "raw", "intended": "joy", "index": 0, "seed": 0, "predicted": "joy"}
    path.write_text(_json.dumps(rec) + "\n")
    ckpt = _Checkpoint(path)
    assert ckpt.cached("raw", "joy", 0, 0) == rec
    assert ckpt.cached("raw", "anger", 0, 0) is None
