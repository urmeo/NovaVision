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
