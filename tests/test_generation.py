import pytest

from novavision.generation import NullBackend, get_backend


def test_factory():
    assert isinstance(get_backend("null"), NullBackend)
    with pytest.raises(ValueError):
        get_backend("nope")


def test_size_and_mode():
    img = NullBackend().generate("hi", width=64, height=48, seed=1)
    assert img.size == (64, 48)
    assert img.mode == "RGB"


def test_seed_is_deterministic():
    a = NullBackend().generate("hi", seed=7).tobytes()
    b = NullBackend().generate("hi", seed=7).tobytes()
    c = NullBackend().generate("hi", seed=8).tobytes()
    assert a == b
    assert a != c


def test_negative_seed_does_not_crash():
    # default backend + negative seed previously raised "expected non-negative integer"
    img = NullBackend().generate("hi", width=8, height=8, seed=-1)
    assert img.size == (8, 8)


def test_hf_api_forwards_negative_prompt_for_non_turbo(monkeypatch):
    from novavision.generation.hf_api_backend import HFApiBackend

    monkeypatch.setenv("HF_TOKEN", "x")
    sent = {}

    class FakeClient:
        def text_to_image(self, prompt, **kw):
            sent.update(kw)
            from PIL import Image

            return Image.new("RGB", (8, 8))

    b = HFApiBackend(model_id="stabilityai/stable-diffusion-2-1")
    b._client = FakeClient()
    b.generate("p", width=8, height=8, seed=3, negative_prompt="blurry")
    assert sent["negative_prompt"] == "blurry" and sent["seed"] == 3
    # turbo model: negative prompt is a no-op, must be dropped
    bt = HFApiBackend(model_id="stabilityai/sd-turbo")
    bt._client = FakeClient()
    bt.generate("p", width=8, height=8, seed=3, negative_prompt="blurry")
    assert sent["negative_prompt"] is None
