import sys

import pytest

from novavision.data.build_benchmark import _curate, _drop_overlap, _interleave, _normalize
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


def test_drop_overlap_removes_cross_split_leakage():
    examples = [("Loved it!", "joy"), ("so scared", "fear"), ("unique line", "joy")]
    train_norms = {_normalize("loved it!"), _normalize("SO   scared")}
    kept = _drop_overlap(examples, train_norms)
    # train-overlapping items dropped, case/space-insensitive
    assert kept == [("unique line", "joy")]
    # No exclusion set -> passthrough.
    assert _drop_overlap(examples, set()) == examples


def test_cli_threads_revision_into_build(monkeypatch):
    # Regression: --revision was accepted by build() but never exposed on the CLI,
    # so the pinned default could not be overridden without editing source.
    from novavision.data import build_benchmark as bb

    captured = {}

    def fake_build(n, out, seed, split, *, revision, drop_train_overlap):
        captured["revision"] = revision
        from pathlib import Path

        return Path(out)

    monkeypatch.setattr(bb, "build", fake_build)
    monkeypatch.setattr(sys, "argv", ["build_benchmark", "--revision", "deadbeef"])
    bb.main()
    assert captured["revision"] == "deadbeef"


def test_build_rejects_nonpositive_n():
    from novavision.data.build_benchmark import build

    for bad in (0, -5):
        with pytest.raises(ValueError, match="positive integer"):
            build(n_per_class=bad, out_path="/tmp/should_not_write.csv")
