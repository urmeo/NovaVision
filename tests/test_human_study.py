import json

import pytest

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
        rated,
        ["id", "image", "emotion"],
        [{"id": r["id"], "image": "", "emotion": r["probe"]} for r in key],
    )
    result = human_study.analyze(rated, study / "key.csv")
    assert result["n_rated"] == 7
    assert result["human_vs_probe_kappa"] == 1.0


def _text_record(with_index: bool) -> dict:
    r = {
        "tier": "emotion",
        "content": "a terrible storm hit the village",
        "intended": "fear",
        "seed": 0,
        "predicted": "fear",
        "intended_valence": -0.7,
        "intended_arousal": 0.8,
        "recovered_valence": -0.5,
        "recovered_arousal": 0.6,
        "clip_t": 0.2,
    }
    if with_index:
        r["index"] = 3
    return r


def _text_payload(record) -> dict:
    return {
        "manifest": {
            "config": {
                "backend": "null",
                "diffusion_model": "stabilityai/sd-turbo",
                "base_seed": 0,
                "width": 32,
                "height": 32,
            }
        },
        "records": [record],
    }


def test_build_sheet_text_track_with_index(tmp_path):
    # New text-track records carry a row index, so the image is reproducible.
    (tmp_path / "results.json").write_text(json.dumps(_text_payload(_text_record(with_index=True))))
    study = human_study.build_sheet(tmp_path, n=1, seed=0)
    assert len(list((study / "images").glob("*.png"))) == 1


def test_build_sheet_rejects_unreproducible_record(tmp_path):
    # Pre-index text-track record (no index, content not in bank): clear error, not a crash.
    (tmp_path / "results.json").write_text(
        json.dumps(_text_payload(_text_record(with_index=False)))
    )
    with pytest.raises(ValueError, match="Cannot reproduce"):
        human_study.build_sheet(tmp_path, n=1, seed=0)


def test_analyze_skips_out_of_vocab_rating(tmp_path):
    key = tmp_path / "key.csv"
    human_study._write_csv(
        key,
        ["id", "intended", "probe"],
        [
            {"id": 0, "intended": "joy", "probe": "joy"},
            {"id": 1, "intended": "anger", "probe": "anger"},
        ],
    )
    rated = tmp_path / "rated.csv"
    # 'happy' is an alias -> joy (scored); 'banana' is unknown -> skipped, not a crash.
    human_study._write_csv(
        rated,
        ["id", "image", "emotion"],
        [
            {"id": 0, "image": "", "emotion": "happy"},
            {"id": 1, "image": "", "emotion": "banana"},
        ],
    )
    res = human_study.analyze(rated, key)
    assert res["n_rated"] == 1 and res["n_unscored"] == 1 and res["unscored_ids"] == [1]
