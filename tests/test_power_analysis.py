import power_analysis as pa  # scripts/ on sys.path via conftest.py


def test_no_effect_is_unreachable():
    # An effect at or below chance can never reach target power.
    assert pa.min_n_for_power(pa.CHANCE, pa.CHANCE, _rng()) is None


def test_stronger_effect_needs_fewer_samples():
    rng = _rng()
    weak = pa.min_n_for_power(0.20, pa.CHANCE, rng)
    strong = pa.min_n_for_power(0.40, pa.CHANCE, rng)
    assert weak is not None and strong is not None
    assert strong < weak


def test_power_rises_with_n():
    rng = _rng()
    assert pa.power_at(30, 0.30, pa.CHANCE, rng) < pa.power_at(400, 0.30, pa.CHANCE, rng)


def test_analyze_reports_every_effect():
    report = pa.analyze(ceiling=0.455, planned_n=420)
    assert report["probe_ceiling"] == 0.455
    assert len(report["rows"]) == len(pa.EFFECTS)
    # The planned run must be well powered for a mid effect against this ceiling.
    mid = next(r for r in report["rows"] if r["effect_strength"] == 0.5)
    assert mid["power_at_planned_n"] >= 0.8


def _rng():
    import numpy as np

    return np.random.default_rng(0)
