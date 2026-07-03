import numpy as np

from novavision.eval import figures


def test_row_normalize_masks_empty_rows():
    matrix = np.array([[2, 0, 0], [0, 0, 0], [1, 0, 1]])
    norm = figures._row_normalize(matrix)
    assert norm[0, 0] == 1.0
    assert np.isnan(norm[1]).all()  # absent class, not "all wrong"
    assert norm[2, 0] == 0.5 and norm[2, 2] == 0.5


def test_plot_confusion_renders_empty_row(tmp_path):
    out = tmp_path / "cm.png"
    figures.plot_confusion(np.array([[3, 1], [0, 0]]), ("a", "b"), out, title="t")
    assert out.exists() and out.stat().st_size > 0
