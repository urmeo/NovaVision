import json

from novavision.eval import human_study
from novavision.eval.metrics import cohen_kappa
from novavision.taxonomy import EMOTIONS


def test_cohen_kappa_perfect_and_chance():
    assert cohen_kappa(["joy", "anger"], ["joy", "anger"], EMOTIONS) == 1.0
    # Disagreement below chance gives a negative kappa.
    assert cohen_kappa(["joy", "anger"], ["anger", "joy"], EMOTIONS) < 0


def _fake_results(tmp_path):
    records = []
    for sk in range(2):
        for e in EMOTIONS:
            records.append(
                {
                    "probe": "clip:test",
                    "tier": "affect",
                    "content": "a city street with buildings",
                    "intended": e,
                    "seed": sk,
                    "predicted": e,
                    "intended_valence": 0.0,
                    "intended_arousal": 0.5,
                    "recovered_valence": 0.0,
                    "recovered_arousal": 0.5,
                    "clip_t": 0.2,
                }
            )
    payload = {
        "manifest": {
            "config": {
                "backend": "null",
                "diffusion_model": "stabilityai/sd-turbo",
                "base_seed": 0,
                "width": 64,
                "height": 64,
            }
        },
        "records": records,
    }
    out = tmp_path / "results.json"
    out.write_text(json.dumps(payload))
    return tmp_path


def test_build_sheet_and_analyze(tmp_path):
    results_dir = _fake_results(tmp_path)
    study = human_study.build_sheet(results_dir, n=7, seed=0)
    assert (study / "ratings_template.csv").exists()
    assert (study / "key.csv").exists()
    assert len(list((study / "images").glob("*.png"))) == 7

    # Simulate a rater agreeing with the probe on every item.
    key = list(human_study._read_csv(study / "key.csv"))
    rated = study / "rated.csv"
    human_study._write_csv(
        rated, ["id", "image", "emotion"],
        [{"id": r["id"], "image": "", "emotion": r["probe"]} for r in key],
    )
    result = human_study.analyze(rated, study / "key.csv")
    assert result["n_rated"] == 7
    assert result["human_vs_probe_kappa"] == 1.0
