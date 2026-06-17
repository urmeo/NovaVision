import pytest
from PIL import Image

from novavision.eval import validate_probe
from novavision.eval.probes import Recovery
from novavision.taxonomy import EMOTIONS


class FakeProbe:
    name = "clip:fake"

    def recover(self, image):
        return Recovery("joy", 0.1, 0.5, {e: 0.0 for e in EMOTIONS})


def _make_dataset(root):
    for e in ("joy", "anger"):
        d = root / e
        d.mkdir(parents=True)
        Image.new("RGB", (8, 8), (10, 20, 30)).save(d / "a.png")


def test_load_image_folder(tmp_path):
    _make_dataset(tmp_path)
    pairs = validate_probe.load_image_folder(tmp_path)
    assert len(pairs) == 2
    assert {label for _, label in pairs} == {"joy", "anger"}


def test_load_rejects_empty(tmp_path):
    with pytest.raises(ValueError):
        validate_probe.load_image_folder(tmp_path)


def test_validate_reports_accuracy(tmp_path):
    _make_dataset(tmp_path)
    report = validate_probe.validate(FakeProbe(), validate_probe.load_image_folder(tmp_path))
    assert report["n"] == 2
    assert report["accuracy"] == 0.5  # joy right, anger wrong
    assert len(report["confusion"]) == len(EMOTIONS)


def test_validate_accepts_pil_images():
    pairs = [(Image.new("RGB", (8, 8)), "joy"), (Image.new("RGB", (8, 8)), "fear")]
    report = validate_probe.validate(FakeProbe(), pairs)
    assert report["n"] == 2
    assert report["accuracy"] == 0.5


def test_ekman_aliases_cover_common_labels():
    aliases = validate_probe.EKMAN_ALIASES
    assert aliases["happy"] == "joy" and aliases["happiness"] == "joy"
    assert aliases["awe"] == "surprise" and aliases["sad"] == "sadness"
    assert set(aliases.values()) <= set(EMOTIONS)
