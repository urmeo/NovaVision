from novavision.data.build_benchmark import _curate, _interleave
from novavision.taxonomy import EMOTIONS


def _examples():
    rows = []
    for e in EMOTIONS:
        for i in range(5):
            rows.append((f"{e} sentence number {i}", e))
    return rows


def test_curate_dedups_exact():
    rows = [("Same Text", "joy"), ("same   text", "joy"), ("other", "joy")]
    sampled = _curate(rows, n_per_class=10, seed=0)
    assert len(sampled["joy"]) == 2


def test_curate_balances_per_class():
    sampled = _curate(_examples(), n_per_class=3, seed=0)
    assert all(len(sampled[e]) == 3 for e in EMOTIONS)


def test_interleave_prefix_is_stratified():
    sampled = _curate(_examples(), n_per_class=3, seed=0)
    rows = _interleave(sampled)
    # First len(EMOTIONS) rows cover every class once.
    assert {e for _, e in rows[: len(EMOTIONS)]} == set(EMOTIONS)
