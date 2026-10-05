import numpy as np

from novavision.eval import figures


def test_row_normalize_masks_empty_rows():
    matrix = np.array([[2, 0, 0], [0, 0, 0], [1, 0, 1]])
    norm = figures._row_normalize(matrix)
    assert norm[0, 0] == 1.0
    assert np.isnan(norm[1]).all()
    assert norm[2, 0] == 0.5 and norm[2, 2] == 0.5


def test_plot_confusion_renders_empty_row(tmp_path):
    out = tmp_path / "cm.png"
    figures.plot_confusion(np.array([[3, 1], [0, 0]]), ("a", "b"), out, title="t")
    assert out.exists() and out.stat().st_size > 0


def test_accuracy_plot_includes_available_intervals_and_skips_missing_ones(tmp_path, monkeypatch):
    plt = figures._pyplot()
    fig, ax = plt.subplots()
    monkeypatch.setattr(plt, "subplots", lambda **kwargs: (fig, ax))
    out = tmp_path / "accuracy.png"
    figures.plot_accuracy(
        {"raw": 0.14, "emotion": 0.21},
        out,
        chance=1 / 7,
        accuracy_ci={"raw": [0.0, 0.36], "emotion": [None, None]},
    )
    assert len(ax.collections) == 2
    assert np.allclose(ax.collections[0].get_segments()[0], [[0, 0.0], [0, 0.36]])
    assert "95% CI" in ax.get_title()
    assert out.exists()
