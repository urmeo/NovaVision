import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_probes  # noqa: E402


def test_mcnemar_no_discordants_is_one():
    gold = ["joy", "anger"]
    r = compare_probes.mcnemar_exact(gold, ["joy", "anger"], ["joy", "anger"])
    assert r == {"only_a_correct": 0, "only_b_correct": 0, "p_value": 1.0}


def test_mcnemar_one_sided_discordants():
    # b beats a on all five discordant items: p = 2 * C(5,0)/2^5 = 0.0625
    gold = ["joy"] * 5
    r = compare_probes.mcnemar_exact(gold, ["anger"] * 5, ["joy"] * 5)
    assert (r["only_a_correct"], r["only_b_correct"]) == (0, 5)
    assert r["p_value"] == pytest.approx(0.0625)


def test_mcnemar_symmetric_is_capped_at_one():
    gold = ["joy", "joy"]
    r = compare_probes.mcnemar_exact(gold, ["joy", "anger"], ["anger", "joy"])
    assert r["p_value"] == 1.0


def test_committed_probe_reports_are_paired_and_significant():
    # The paper's section-7 claim regenerates from the committed artifacts.
    import json

    root = Path(__file__).resolve().parents[1] / "results" / "paper"
    a = json.loads((root / "probe_validation_scene.json").read_text())
    b = json.loads((root / "probe_validation_scene_l14.json").read_text())
    assert a["gold"] == b["gold"]
    r = compare_probes.mcnemar_exact(a["gold"], a["predictions"], b["predictions"])
    assert r["p_value"] == pytest.approx(0.0375, abs=1e-4)
