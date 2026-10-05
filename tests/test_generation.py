import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from novavision.generation import NullBackend, get_backend


def test_factory():
    assert isinstance(get_backend("null"), NullBackend)
    with pytest.raises(ValueError):
        get_backend("nope")


def test_hf_api_ignores_device_kwarg(monkeypatch):

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

    bt = HFApiBackend(model_id="stabilityai/sd-turbo")
    bt._client = FakeClient()
    bt.generate("p", width=8, height=8, seed=3, negative_prompt="blurry")
    assert sent["negative_prompt"] is None


def test_diffusers_dtype_known_before_first_generation():
    from novavision.generation.diffusers_backend import DiffusersBackend

    assert DiffusersBackend(device="cpu").dtype == "float32"
    assert DiffusersBackend(device="cuda").dtype == "float16"


def test_diffusers_serializes_requests_using_the_cached_pipeline(monkeypatch):
    from PIL import Image

    from novavision.generation.diffusers_backend import DiffusersBackend

    class FakeGenerator:
        def __init__(self, **kwargs):
            pass

        def manual_seed(self, seed):
            return self

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(Generator=FakeGenerator))
    first_entered = threading.Event()
    release_first = threading.Event()
    second_attempted = threading.Event()
    second_entered = threading.Event()
    image = Image.new("RGB", (8, 8))

    def fake_pipe(**kwargs):
        if kwargs["prompt"] == "first":
            first_entered.set()
            assert release_first.wait(3), "test did not release the first request"
        else:
            second_entered.set()
        return SimpleNamespace(images=[image])

    backend = DiffusersBackend(device="cpu")
    backend._pipe = fake_pipe

    def second_request():
        second_attempted.set()
        return backend.generate("second", width=8, height=8)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(backend.generate, "first", width=8, height=8)
        try:
            assert first_entered.wait(3)
            second = executor.submit(second_request)
            assert second_attempted.wait(3)
            assert not second_entered.wait(0.1), "shared pipeline ran two requests at once"
        finally:
            release_first.set()
        assert first.result(timeout=3) is image
        assert second.result(timeout=3) is image
        assert second_entered.is_set()


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
