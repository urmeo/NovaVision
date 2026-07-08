import pytest

from novavision.generation import NullBackend, get_backend


def test_factory():
    assert isinstance(get_backend("null"), NullBackend)
    with pytest.raises(ValueError):
        get_backend("nope")


def test_hf_api_ignores_device_kwarg(monkeypatch):
    # get_backend forwards `device` to every backend uniformly; the hosted API has
    # no local device and must absorb it, not crash (the --backend hf-api --device combo).
    monkeypatch.setenv("HF_TOKEN", "dummy")
    b = get_backend("hf-api", model_id="stabilityai/sd-turbo", device="cpu")
    assert b.name == "hf-api"


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


def test_diffusers_dtype_known_before_first_generation():
    from novavision.generation.diffusers_backend import DiffusersBackend

    # The manifest reads backend.dtype; it must exist without loading the pipe.
    assert DiffusersBackend(device="cpu").dtype == "float32"
    assert DiffusersBackend(device="cuda").dtype == "float16"


def test_null_backend_prompt_sensitivity():
    a = NullBackend().generate("hi", seed=7).tobytes()
    b = NullBackend().generate("bye", seed=7).tobytes()
    assert a != b


def test_hf_api_default_size_matches_diffusers(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "x")
    from novavision.generation.hf_api_backend import HFApiBackend

    sent = {}

    class FakeClient:
        def text_to_image(self, prompt, **kw):
            sent.update(kw)
            from PIL import Image

            return Image.new("RGB", (8, 8))

    b = HFApiBackend(model_id="stabilityai/sd-turbo")
    b._client = FakeClient()
    b.generate("p", seed=1)
    assert (sent["width"], sent["height"]) == (512, 512)
